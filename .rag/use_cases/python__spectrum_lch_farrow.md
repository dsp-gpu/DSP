---
id: dsp__spectrum_lch_farrow__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_lch_farrow.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum', 'signal_generators']
uses_external: ['json', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 327
title: Тесты LchFarrow задержки
tags: ['spectrum', 'rocm', 'gpu', 'python', 'signal_processing', 'filter', 'lfm']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.LchFarrowROCm
  - dsp_signal_generators.LfmAnalyticalDelayROCm
top_functions:
  - load_lagrange_matrix
  - apply_delay_numpy
  - generate_cw_signal
  - test_zero_delay
  - test_integer_delay
  - test_fractional_delay
  - test_multi_antenna
  - test_lch_farrow_vs_analytical
synonyms_ru:
  - тесты задержки
  - LchFarrow проверка
  - дробная задержка
  - Lagrange матрица
  - GPU задержка
synonyms_en:
  - delay tests
  - LchFarrow validation
  - fractional delay
  - Lagrange matrix
  - GPU delay
inherits_block_id: spectrum__lch_farrow_rocm__class_overview__v1
block_refs:
  - spectrum__lch_farrow_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_lch_farrow__python_test_usecase__v1 -->

# Python use-case: Тесты LchFarrow задержки

## Цель

Проверка корректности работы LchFarrow для дробных задержек с использованием матрицы Lagrange 48x5 и сравнение с NumPy.

## Когда применять

Запускать после изменений в LchFarrowROCm или LfmAnalyticalDelayROCm.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.LchFarrowROCm` | spectrum |
| `dsp_signal_generators.LfmAnalyticalDelayROCm` | signal_generators |

## Внешние зависимости

json, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore

try:
    import dsp_signal_generators as sg
    HAS_SG = True
except ImportError:
    HAS_SG = False
    sg = None        # type: ignore


# ════════════════════════════════════════════════════════════════════════════
# Загрузка матрицы Lagrange 48×5
# ════════════════════════════════════════════════════════════════════════════

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'lagrange_matrix_48x5.json')


def load_lagrange_matrix():
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__lch_farrow_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_lch_farrow.py`
- **Строк кода**: 327
- **Top-функций**: 8
- **Test runner**: common.runner

<!-- /rag-block -->
