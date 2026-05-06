---
id: dsp__spectrum_filters_stage1__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_filters_stage1.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['matplotlib', 'matplotlib.pyplot', 'numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 330
title: Тестирование gpu фильтров стадии 1
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'fir', 'iir', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.FirFilterROCm
  - dsp_spectrum.IirFilterROCm
top_functions:
  - generate_test_signal
  - test_fir_gpu_vs_scipy
  - test_fir_basic_properties
  - test_fir_single_channel
  - test_iir_gpu_vs_scipy
  - test_iir_basic_properties
  - plot_filter_results
synonyms_ru:
  - тест gpu фильтры стадия 1
  - тестирование фильтров gpu
  - gpu фильтры проверка
  - стадия 1 фильтры тест
  - gpu сигнальные тесты
synonyms_en:
  - gpu filter stage 1 python_test
  - filter validation gpu
  - stage 1 gpu testing
  - gpu filter pipeline python_test
  - scipy gpu filter python_test
inherits_block_id: spectrum__fir_filter_rocm__class_overview__v1
block_refs:
  - spectrum__fir_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_filters_stage1__python_test_usecase__v1 -->

# Python use-case: Тестирование gpu фильтров стадии 1

## Цель

Проверка корректности gpu-фильтров fir и iir против scipy.signal

## Когда применять

Запускать после изменений в gpu-контексте или реализации фильтров

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.FirFilterROCm` | spectrum |
| `dsp_spectrum.IirFilterROCm` | spectrum |

## Внешние зависимости

matplotlib, matplotlib.pyplot, numpy, scipy.signal

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print("WARNING: dsp_core/dsp_spectrum not found. Only CPU reference tests will run.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Skipping scipy validation tests.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__fir_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_filters_stage1.py`
- **Строк кода**: 330
- **Top-функций**: 7
- **Test runner**: common.runner

<!-- /rag-block -->
