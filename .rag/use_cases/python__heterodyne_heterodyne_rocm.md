---
id: dsp__heterodyne_heterodyne_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/heterodyne/t_heterodyne_rocm.py
primary_repo: heterodyne
module: heterodyne
uses_repos: ['core', 'heterodyne']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 315
title: heterodyne rocm vs numpy
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'heterodyne', 'lfm', 'core']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_heterodyne.HeterodyneROCm
top_functions:
  - ref_dechirp
  - ref_correct
  - make_random_signal
  - _make_het
  - _check_gpu
  - test_dechirp_vs_numpy
  - test_dechirp_multi_antenna
  - test_correct_zero_beat
  - test_correct_vs_numpy
  - test_params_dict
  - test_dechirp_correct_chain
synonyms_ru:
  - heterodyne processing
  - lfm demodulation
  - gpu signal processing
  - numpy vs rocm
  - rf signal processing
synonyms_en:
  - heterodyne processing
  - lfm demodulation
  - gpu signal processing
  - numpy vs rocm
  - rf signal processing
inherits_block_id: heterodyne__heterodyne_rocm__class_overview__v1
block_refs:
  - heterodyne__heterodyne_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__heterodyne_heterodyne_rocm__python_test_usecase__v1 -->

# Python use-case: heterodyne rocm vs numpy

## Цель

Тестирование GPU-реализации процессора heterodyne для LFM против NumPy

## Когда применять

Запускать после изменений в heterodyne_rocm.hip или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_heterodyne.HeterodyneROCm` | heterodyne |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_heterodyne as heterodyne
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None        # type: ignore
    heterodyne = None  # type: ignore

# ============================================================================
# Default LFM parameters
# ============================================================================

F_START      = 0.0
F_END        = 2_000_000.0   # 2 MHz bandwidth
SAMPLE_RATE  = 12_000_000.0  # 12 MHz
NUM_SAMPLES  = 8_000
NUM_ANTENNAS = 5
ATOL         = 1e-4

# ============================================================================
# NumPy reference formulas
# ============================================================================

def ref_dechirp(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
```

## Connection (C++ ↔ Python)

- C++ class-card: `heterodyne__heterodyne_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/heterodyne/t_heterodyne_rocm.py`
- **Строк кода**: 315
- **Top-функций**: 11
- **Test runner**: common.runner

<!-- /rag-block -->
