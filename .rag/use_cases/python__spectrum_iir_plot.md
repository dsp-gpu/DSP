---
id: dsp__spectrum_iir_plot__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_iir_plot.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['matplotlib', 'matplotlib.patches', 'matplotlib.pyplot', 'numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 308
title: iir фильтр gpu и визуализация
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.IirFilterROCm
top_functions:
  - sos_to_sections
  - generate_test_signal
  - test_iir_gpu_vs_scipy
  - test_iir_basic_properties
  - plot_iir_results
synonyms_ru:
  - iir_фильтр
  - butterworth
  - gpu_график
  - частотный_ответ
  - полюса_нули
inherits_block_id: spectrum__iir_filter_rocm__class_overview__v1
block_refs:
  - spectrum__iir_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_iir_plot__python_test_usecase__v1 -->

# Python use-case: iir фильтр gpu и визуализация

## Цель

Проверка корректности gpu-реализации butterworth iir-фильтра с визуализацией 4-панельного сравнения

## Когда применять

Запускать после изменений в iir-реализации или gpu-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.IirFilterROCm` | spectrum |

## Внешние зависимости

matplotlib, matplotlib.patches, matplotlib.pyplot, numpy, scipy.signal

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print("WARNING: dsp_core/dsp_spectrum not found.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: matplotlib not found.")
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__iir_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_iir_plot.py`
- **Строк кода**: 308
- **Top-функций**: 5
- **Test runner**: common.runner

<!-- /rag-block -->
