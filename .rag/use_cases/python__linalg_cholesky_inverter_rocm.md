---
id: dsp__linalg_cholesky_inverter_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/linalg/t_cholesky_inverter_rocm.py
primary_repo: linalg
module: linalg
uses_repos: ['core', 'linalg']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 178
title: Cholesky инвертор GPU vs numpy
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_linalg.CholeskyInverterROCm
  - dsp_linalg.SymmetrizeMode.GpuKernel
  - dsp_linalg.SymmetrizeMode
  - dsp_linalg.SymmetrizeMode.Roundtrip
top_functions:
  - make_positive_definite
  - frobenius_error
synonyms_ru:
  - Тест Cholesky инвертора
  - Сравнение GPU и numpy
  - Инверсия матриц GPU
  - Тесты Cholesky
  - ROCm Cholesky
inherits_block_id: linalg__symmetrize_mode__class_overview__v1
block_refs:
  - linalg__symmetrize_mode__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__linalg_cholesky_inverter_rocm__python_test_usecase__v1 -->

# Python use-case: Cholesky инвертор GPU vs numpy

## Цель

Проверка точности GPU-реализации Cholesky-инвертора против numpy.linalg.inv на HPD-матрицах.

## Когда применять

Запускать после изменений в CholeskyInverterROCm или GPU-контексте.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_linalg.CholeskyInverterROCm` | linalg |
| `dsp_linalg.SymmetrizeMode.GpuKernel` | linalg |
| `dsp_linalg.SymmetrizeMode` | linalg |
| `dsp_linalg.SymmetrizeMode.Roundtrip` | linalg |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_linalg as linalg
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None    # type: ignore
    linalg = None  # type: ignore


# ============================================================================
# Helpers
# ============================================================================


def make_positive_definite(n: int, seed: int = 42) -> np.ndarray:
    """Создать HPD матрицу n×n: A = B*B^H + n*I."""
    rng = np.random.default_rng(seed)
    B = (rng.standard_normal((n, n)) +
         1j * rng.standard_normal((n, n))).astype(np.complex64)
    A = (B @ B.conj().T + n * np.eye(n, dtype=np.complex64)).astype(np.complex64)
    return A


def frobenius_error(A: np.ndarray, A_inv: np.ndarray) -> float:
    """||A * A_inv - I||_F"""
```

## Connection (C++ ↔ Python)

- C++ class-card: `linalg__symmetrize_mode__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/linalg/t_cholesky_inverter_rocm.py`
- **Строк кода**: 178
- **Top-функций**: 2
- **Test runner**: common.runner

<!-- /rag-block -->
