---
id: dsp__heterodyne_heterodyne_step_by_step__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/heterodyne/t_heterodyne_step_by_step.py
primary_repo: heterodyne
module: heterodyne
uses_repos: ['core', 'heterodyne']
uses_external: ['matplotlib', 'matplotlib.pyplot', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 716
title: тест дешифрования пошагово
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'heterodyne', 'fft', 'lfm']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_heterodyne.HeterodyneROCm
top_functions:
  - generate_rx_numpy
  - generate_ref_conjugate_numpy
  - parabolic_interp
  - save_plot
  - add_params_banner
  - step01_generate_rx
  - step02_generate_ref_conjugate
  - step03_dechirp
  - step04_fft
  - step05_find_maxima
  - step06_dechirp_correct
  - step07_verify_dc
  - step08_gpu_pipeline
  - print_summary
  - run_full_test
synonyms_ru:
  - тест дешифрования пошагово
  - сравнение gpu и cpu
  - тест дешифрования с графиками
  - проверка алгоритма дешифрования
  - тест обработки сигналов
synonyms_en:
  - step-by-step dechirp python_test
  - gpu cpu comparison
  - dechirp pipeline verification
  - signal processing python_test
  - heterodyne algorithm check
inherits_block_id: heterodyne__heterodyne_rocm__class_overview__v1
block_refs:
  - heterodyne__heterodyne_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__heterodyne_heterodyne_step_by_step__python_test_usecase__v1 -->

# Python use-case: тест дешифрования пошагово

## Цель

Сравнение GPU и CPU для дешифрования сигнала пошагово с визуализацией

## Когда применять

Запускать после обновления GPU-реализации дешифрования или изменений в параметрах

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_heterodyne.HeterodyneROCm` | heterodyne |

## Внешние зависимости

matplotlib, matplotlib.pyplot, numpy

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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, plots will be skipped")

# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6
F_START = 0.0
F_END = 2e6
B = F_END - F_START   # 2 MHz
```

## Connection (C++ ↔ Python)

- C++ class-card: `heterodyne__heterodyne_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/heterodyne/t_heterodyne_step_by_step.py`
- **Строк кода**: 716
- **Top-функций**: 15
- **Test runner**: common.runner

<!-- /rag-block -->
