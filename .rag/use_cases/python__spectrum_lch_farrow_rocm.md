---
id: dsp__spectrum_lch_farrow_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_lch_farrow_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['json', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 322
title: gpu vs cpu фракционная задержка
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'lagrange', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.LchFarrowROCm
top_functions:
  - load_lagrange_matrix
  - cpu_lch_farrow
  - make_complex_signal
  - make_ctx_lch
  - test_zero_delay
  - test_integer_delay
  - test_fractional_delay_vs_cpu
  - test_multi_antenna
  - test_properties
synonyms_ru:
  - фракционная задержка
  - gpu vs cpu
  - лагранжево интерполяция
  - задержка сигнала
  - gpu тест
synonyms_en:
  - fractional delay
  - gpu vs cpu
  - lagrange interpolation
  - signal delay
  - gpu python_test
inherits_block_id: spectrum__lch_farrow_rocm__class_overview__v1
block_refs:
  - spectrum__lch_farrow_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_lch_farrow_rocm__python_test_usecase__v1 -->

# Python use-case: gpu vs cpu фракционная задержка

## Цель

Тестирование gpu-реализации lch_farrow_rocm против cpu-референса для фракционной задержки сигнала.

## Когда применять

Запускать после изменений в lch_farrow_rocm или rocmgpucontext.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.LchFarrowROCm` | spectrum |

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
    print(f"WARNING: dsp_core/dsp_spectrum not found. (searched: {GPULoader.loaded_from()})")

# ============================================================================
# Load Lagrange matrix 48×5
# ============================================================================

MATRIX_PATH = os.path.join(os.path.dirname(__file__), 'data', 'lagrange_matrix_48x5.json')

def load_lagrange_matrix() -> np.ndarray:
    """Load 48x5 Lagrange matrix from JSON. Returns (48, 5) float32 array."""
    with open(MATRIX_PATH, 'r') as f:
        data = json.load(f)
    arr = np.array(data['data'], dtype=np.float32)
    return arr.reshape(data['rows'], data['columns'])


# ============================================================================
# CPU reference — exact match of C++ ProcessCpu
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__lch_farrow_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_lch_farrow_rocm.py`
- **Строк кода**: 322
- **Top-функций**: 9
- **Test runner**: common.runner

<!-- /rag-block -->
