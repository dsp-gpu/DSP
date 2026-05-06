---
id: dsp__heterodyne_heterodyne_comparison__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/heterodyne/t_heterodyne_comparison.py
primary_repo: heterodyne
module: heterodyne
uses_repos: ['core', 'heterodyne']
uses_external: ['matplotlib', 'matplotlib.gridspec', 'matplotlib.pyplot', 'numpy', 'time']
has_test_runner: true
is_opencl: false
line_count: 569
title: сравнение gpu и cpu децирпирования
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'heterodyne', 'dechirp']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_heterodyne.HeterodyneROCm
top_functions:
  - cpu_pipeline
  - gpu_pipeline
  - run_comparison
  - print_comparison_table
  - generate_report_md
  - generate_comparison_plot
  - main
synonyms_ru:
  - тест децирпирования
  - сравнение gpu cpu
  - анализ гетеродина
  - тест rocm
  - сравнение сигналов
synonyms_en:
  - heterodyne dechirp python_test
  - gpu cpu comparison
  - heterodyne analysis
  - rocm python_test
  - signal comparison
inherits_block_id: heterodyne__heterodyne_rocm__class_overview__v1
block_refs:
  - heterodyne__heterodyne_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__heterodyne_heterodyne_comparison__python_test_usecase__v1 -->

# Python use-case: сравнение gpu и cpu децирпирования

## Цель

верификация gpu-реализации децирпирования гетеродина против cpu-реализации с анализом точности и производительности

## Когда применять

запускать после миграции на rocm и обновления gpu-контекста

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_heterodyne.HeterodyneROCm` | heterodyne |

## Внешние зависимости

matplotlib, matplotlib.gridspec, matplotlib.pyplot, numpy, time

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_heterodyne as heterodyne
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None        # type: ignore
    heterodyne = None  # type: ignore

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, plots will be skipped")

# ============================================================================
# Constants
# ============================================================================

FS = 12e6
F_START = 0.0
F_END = 2e6
```

## Connection (C++ ↔ Python)

- C++ class-card: `heterodyne__heterodyne_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/heterodyne/t_heterodyne_comparison.py`
- **Строк кода**: 569
- **Top-функций**: 7
- **Test runner**: common.runner

<!-- /rag-block -->
