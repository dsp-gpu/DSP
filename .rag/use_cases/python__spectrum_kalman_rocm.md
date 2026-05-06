---
id: dsp__spectrum_kalman_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_kalman_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 459
title: kalman фильтр gpu vs numpy
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'kalman', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.KalmanFilterROCm
top_functions:
  - _kalman_1ch_scalar
  - kalman_ref
  - kalman_steady_state_gain
  - make_complex_signal
  - make_ctx_kalman
  - test_kalman_basic
  - test_kalman_multi_channel
  - test_kalman_channel_independence
  - test_kalman_noise_reduction
  - test_kalman_step_response
  - test_kalman_lfm_radar_demo
  - test_kalman_properties
synonyms_ru:
  - kalman фильтр
  - gpu тест
  - numpy сравнение
  - сигнальный тест
  - фильтрация
synonyms_en:
  - kalman filter
  - gpu python_test
  - numpy comparison
  - signal python_test
  - filtering
inherits_block_id: spectrum__kalman_filter_rocm__class_overview__v1
block_refs:
  - spectrum__kalman_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_kalman_rocm__python_test_usecase__v1 -->

# Python use-case: kalman фильтр gpu vs numpy

## Цель

Проверка корректности gpu-реализации kalman-фильтра против numpy на 1d и многоканальных сигналах

## Когда применять

Запускать после изменений в kalman_filter_rocm.hip или gpu-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.KalmanFilterROCm` | spectrum |

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
    print("WARNING: dsp_core/dsp_spectrum not found. Skipping GPU tests.")

# ============================================================================
# Parameters
# ============================================================================

POINTS   = 4096
CHANNELS = 8
Q_DEF    = 0.1    # process noise variance (default)
R_DEF    = 25.0   # measurement noise variance (default)
X0_DEF   = 0.0    # initial state
P0_DEF   = 25.0   # initial error covariance (= R by default)
ATOL     = 1e-4   # float32 tolerance GPU vs Python reference

# ============================================================================
# Python reference implementation — matches GPU kalman_kernel exactly
# ============================================================================
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__kalman_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_kalman_rocm.py`
- **Строк кода**: 459
- **Top-функций**: 12
- **Test runner**: common.runner

<!-- /rag-block -->
