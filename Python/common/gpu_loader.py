"""
gpu_loader.py — DSPLoader Singleton
=====================================

Singleton (GoF) + Protected Variations (GRASP):
  Находит dsp_core.so/.pyd один раз для всей сессии.
  Все тесты получают модуль через GPULoader.get() вместо хардкода путей.

Порядок поиска (от приоритетного к резервному):
  0. DSP_LIB_DIR (переменная окружения)  ← высший приоритет
  1. DSP/Python/libs/                    ← CMake POST_BUILD auto-deploy (Phase B)
  2. build/python/Release                ← MSVC Release
  3. build/python/Debug                  ← MSVC Debug
  4. build/debian-radeon9070/python      ← Linux ROCm (RDNA4, gfx1201)
  5. build/debian-mi100/python           ← Linux ROCm (CDNA1, gfx908)
  6. build/Release, build/Debug
  7. build/**/dsp_core.*                 ← авто-поиск по build/

После нахождения libs/ добавляет его в sys.path.
Тогда `import dsp_core` (и остальные `dsp_*`) найдёт все 8 .so/.pyd модулей.

Основной API (DSP-GPU):
    GPULoader.setup_path()     # добавляет libs/ в sys.path
    import dsp_core as core
    import dsp_spectrum as spectrum
    ctx = core.ROCmGPUContext()

Legacy (DEPRECATED после Phase A5 — gpuworklib shim удалён):
    gw = GPULoader.get()       # ВСЕГДА None — используй прямой import dsp_core

@author Кодо (AI Assistant)
@date 2026-04-12 (Phase 3b, dsp-gpu модульная архитектура)
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional


# Корень DSP/Python/ (2 уровня вверх от common/)
_PYTHON_ROOT = Path(__file__).parents[1]
# Корень DSP/ (3 уровня вверх от common/)
_DSP_ROOT = Path(__file__).parents[2]

_SEARCH_PATHS = [
    _PYTHON_ROOT / "libs",           # CMake POST_BUILD auto-deploy (Phase B)
    _DSP_ROOT / "build/python/Release",
    _DSP_ROOT / "build/python/Debug",
    _DSP_ROOT / "build/debian-radeon9070/python",
    _DSP_ROOT / "build/debian-mi100/python",
    _DSP_ROOT / "build/Release",
    _DSP_ROOT / "build/Debug",
    _DSP_ROOT / "build/python",
]


class GPULoader:
    """Singleton — настраивает sys.path для dsp_core / dsp_<module>.

    Attributes:
        _gpuworklib:   legacy field, всегда None после Phase A5 (shim удалён)
        _lib_path:     путь откуда загружены модули
        _load_attempted: флаг что поиск уже выполнен
    """

    _gpuworklib = None  # DEPRECATED: всегда None после Phase A5 (gpuworklib shim удалён)
    _lib_path: Optional[Path] = None
    _load_attempted: bool = False

    @classmethod
    def get(cls):
        """DEPRECATED: возвращает None после Phase A5 (gpuworklib shim удалён).

        Используй прямой импорт: `import dsp_core; import dsp_<module>`.
        Метод сохранён для обратной совместимости utility-скриптов.
        """
        if not cls._load_attempted:
            cls._load_attempted = True
            cls._try_load()
        return cls._gpuworklib  # всегда None после A5

    @classmethod
    def setup_path(cls) -> bool:
        """Добавить libs/ в sys.path без возврата модуля.

        Returns:
            True если libs/ найден и добавлен.
        """
        if not cls._load_attempted:
            cls._load_attempted = True
            cls._try_load()
        return cls._lib_path is not None

    @classmethod
    def _try_load(cls) -> None:
        """Найти libs-директорию с dsp_core (после Phase A5 cleanup gpuworklib shim удалён)."""

        def _try_lib_path(path: Path) -> bool:
            """Добавить path в sys.path и попробовать import dsp_core."""
            if not path.exists():
                return False
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                import dsp_core  # noqa: F401
                cls._lib_path = path
                return True
            except ImportError:
                if path_str in sys.path:
                    sys.path.remove(path_str)
                return False

        # 0. Переменная окружения DSP_LIB_DIR (высший приоритет)
        env_dir = os.environ.get("DSP_LIB_DIR")
        if env_dir:
            candidate = Path(env_dir)
            if not candidate.is_absolute():
                candidate = _DSP_ROOT / candidate
            if _try_lib_path(candidate):
                return

        # 1. Уже доступен?
        try:
            import dsp_core  # noqa: F401
            cls._lib_path = Path("(already in sys.path)")
            return
        except ImportError:
            pass

        # 2. Перебрать статические пути
        for candidate in _SEARCH_PATHS:
            if _try_lib_path(candidate):
                return

        # 3. Авто-поиск: найти dsp_core.* где угодно внутри build/
        build_dir = _DSP_ROOT / "build"
        if build_dir.exists():
            pattern = str(build_dir / "**" / "dsp_core*")
            for found in sorted(glob.glob(pattern, recursive=True)):
                fp = Path(found)
                if fp.suffix in (".pyd", ".so") or ".cpython" in fp.name:
                    if _try_lib_path(fp.parent):
                        return

        # Не нашли — _lib_path остаётся None

    @classmethod
    def loaded_from(cls) -> Optional[str]:
        """Вернуть путь откуда загружены модули (для диагностики)."""
        return str(cls._lib_path) if cls._lib_path else None

    @classmethod
    def is_available(cls) -> bool:
        """True если dsp_core доступен (deprecated — используй setup_path() напрямую)."""
        # DEPRECATED: после Phase A5 (удаление gpuworklib shim) этот метод
        # просто проверяет что setup_path() нашёл libs/. Сохранён для обратной
        # совместимости утилит.
        return cls.setup_path()

    @classmethod
    def reset(cls) -> None:
        """Сбросить Singleton (для тестирования GPULoader)."""
        cls._gpuworklib = None  # legacy field, всегда None после Phase A5
        cls._lib_path = None
        cls._load_attempted = False
