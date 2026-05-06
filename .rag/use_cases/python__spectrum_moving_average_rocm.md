---
id: dsp__spectrum_moving_average_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_moving_average_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 428
title: Скользящие средние GPU vs numpy
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.MovingAverageFilterROCm
top_functions:
  - _ema_1ch
  - ema_ref
  - mma_ref
  - _sma_1ch
  - sma_ref
  - dema_ref
  - tema_ref
  - make_complex_signal
synonyms_ru:
  - скользящее среднее
  - фильтр скользящего среднего
  - gpu-фильтр
  - обработка сигналов
  - rocm-фильтр
inherits_block_id: spectrum__moving_average_filter_rocm__class_overview__v1
block_refs:
  - spectrum__moving_average_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_moving_average_rocm__python_test_usecase__v1 -->

# Python use-case: Скользящие средние GPU vs numpy

## Цель

Проверка корректности GPU-реализации скользящих средних против numpy на различных типах фильтров.

## Когда применять

Запускать после изменений в GPU-контексте или реализации фильтров.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.MovingAverageFilterROCm` | spectrum |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print(f"WARNING: dsp_core/dsp_spectrum not found. (searched: {GPULoader.loaded_from()})")

# ============================================================================
# Parameters
# ============================================================================

POINTS   = 4096
CHANNELS = 8
N_WIN    = 10    # window size for EMA / MMA / DEMA / TEMA
N_SMA    = 8     # window size for SMA (must be ≤ 128, GPU ring buffer limit)
ATOL     = 1e-4  # float32 tolerance GPU vs Python reference

# ============================================================================
# Python reference implementations — float32 arithmetic, match GPU kernels
# ============================================================================

def _ema_1ch(data: np.ndarray, alpha: float) -> np.ndarray:
    """EMA on a single 1D complex64 channel, float32 arithmetic.
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__moving_average_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_moving_average_rocm.py`
- **Строк кода**: 428
- **Top-функций**: 8
- **Test runner**: common.runner

<!-- /rag-block -->
