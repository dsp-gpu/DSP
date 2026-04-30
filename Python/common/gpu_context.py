"""
gpu_context.py — GPUContextManager Singleton
=============================================

Singleton (GoF):
  Создаёт GPU-контекст один раз для всей сессии.
  Переиспользование контекста критично — создание занимает ~1-2 сек.

  GPU-индекс берётся из configGPU.json (рядом с dsp_core.so),
  секция "gpus" → первый элемент с "is_active": true.

Usage:
    ctx = GPUContextManager.get()            # GPUContext (OpenCL), device из конфига
    ctx_rocm = GPUContextManager.get_rocm() # ROCmGPUContext, device из конфига

    # Использование в тестах:
    ctx = GPUContextManager.get()
    ctx_rocm = GPUContextManager.get_rocm()
"""

from pathlib import Path
from typing import Optional

from .gpu_loader import GPULoader
from .configs import first_active_gpu_id


def _find_config_path() -> Optional[Path]:
    """Найти configGPU.json рядом с загруженной dsp_core.so.

    dsp_core.so лежит в  build/.../python/  (или DSP/Python/libs/ после Phase B CMake)
    configGPU.json копируется в build/.../ (на уровень выше .so)
    """
    loaded = GPULoader.loaded_from()
    if loaded:
        candidate = Path(loaded).parent / "configGPU.json"
        if candidate.exists():
            return candidate
    return None


def _active_device() -> int:
    """Вернуть id первого активного GPU из configGPU.json."""
    cfg = _find_config_path()
    if cfg is None:
        return 0
    return first_active_gpu_id(cfg, default=0)


class GPUContextManager:
    """Singleton — хранит GPUContext и ROCmGPUContext для всей сессии.

    GPU-индекс читается из configGPU.json (рядом с .so бинарником).

    Attributes:
        _context:      единственный GPUContext (OpenCL)
        _rocm_context: единственный ROCmGPUContext
        _device_index: индекс GPU на котором созданы контексты
    """

    _context = None
    _rocm_context = None
    _device_index: int = 0
    _create_attempted: bool = False

    @classmethod
    def get(cls, device: Optional[int] = None):
        """Получить или создать GPUContext (OpenCL).

        Args:
            device: индекс GPU. None = из configGPU.json.

        Returns:
            GPUContext или None если dsp_core недоступен.
        """
        if not cls._create_attempted:
            cls._create_attempted = True
            cls._device_index = device if device is not None else _active_device()
            cls._try_create(cls._device_index)
        return cls._context

    @classmethod
    def get_rocm(cls, device: Optional[int] = None):
        """Получить или создать ROCmGPUContext.

        Args:
            device: индекс GPU. None = из configGPU.json (тот же что и get()).

        Returns:
            ROCmGPUContext или None если ROCm недоступен.
        """
        if not cls._create_attempted:
            cls.get(device)  # инициализируем _device_index

        if cls._rocm_context is None:
            dev = device if device is not None else cls._device_index
            cls._try_create_rocm(dev)
        return cls._rocm_context

    @classmethod
    def _try_create(cls, device: int) -> None:
        """Создать GPUContext (OpenCL) через dsp_core."""
        if not GPULoader.setup_path():
            return
        try:
            import dsp_core
            cls._context = dsp_core.GPUContext(device)
        except Exception as e:
            print(f"[GPUContextManager] dsp_core.GPUContext(device={device}) failed: {e}")
            cls._context = None

    @classmethod
    def _try_create_rocm(cls, device: int) -> None:
        """Создать ROCmGPUContext через dsp_core."""
        if not GPULoader.setup_path():
            return
        try:
            import dsp_core
            if not hasattr(dsp_core, "ROCmGPUContext"):
                return
            cls._rocm_context = dsp_core.ROCmGPUContext(device)
        except Exception as e:
            print(f"[GPUContextManager] dsp_core.ROCmGPUContext(device={device}) failed: {e}")
            cls._rocm_context = None

    @classmethod
    def is_available(cls) -> bool:
        """True если OpenCL-контекст создан."""
        return cls.get() is not None

    @classmethod
    def is_rocm_available(cls) -> bool:
        """True если ROCm-контекст создан."""
        return cls.get_rocm() is not None

    @classmethod
    def device_index(cls) -> int:
        """Индекс GPU на котором созданы контексты."""
        return cls._device_index

    @classmethod
    def reset(cls) -> None:
        """Сбросить контексты (для тестирования GPUContextManager)."""
        cls._context = None
        cls._rocm_context = None
        cls._create_attempted = False
