---
id: dsp__linalg_matrix_csv_comparison__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/linalg/t_matrix_csv_comparison.py
primary_repo: linalg
module: linalg
uses_repos: ['core', 'linalg']
uses_external: ['datetime', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 227
title: Сравнение инверсии матриц GPU и CSV
tags: ['linalg', 'rocm', 'gpu', 'python', 'matrix_inversion', 'cross_repo', 'signal_processing']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_linalg.CholeskyInverterROCm
  - dsp_linalg.SymmetrizeMode.GpuKernel
  - dsp_linalg.SymmetrizeMode
top_functions:
  - load_complex_matrix_csv
  - frobenius_diff
  - relative_error
  - generate_report
synonyms_ru:
  - матричный тест
  - проверка инверсии
  - сравнение gpu и csv
  - тест обратной матрицы
  - верификация матрицы
synonyms_en:
  - matrix_inversion_test
  - gpu_vs_csv_comparison
  - matrix_inverse_verification
  - matrix_inversion_validation
  - csv_benchmark
inherits_block_id: linalg__symmetrize_mode__class_overview__v1
block_refs:
  - linalg__symmetrize_mode__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__linalg_matrix_csv_comparison__python_test_usecase__v1 -->

# Python use-case: Сравнение инверсии матриц GPU и CSV

## Цель

Проверка точности вычисления обратной матрицы через CholeskyInverterROCm против эталонных данных из CSV.

## Когда применять

Запускать после изменений в linalg.CholeskyInverterROCm или обработке данных в data/.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_linalg.CholeskyInverterROCm` | linalg |
| `dsp_linalg.SymmetrizeMode.GpuKernel` | linalg |
| `dsp_linalg.SymmetrizeMode` | linalg |

## Внешние зависимости

datetime, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_linalg as linalg
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None    # type: ignore
    linalg = None  # type: ignore

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
# CSV data: скопированы в DSP/Python/linalg/data/ (Phase A2.0 pre-scan)
DATA_DIR = os.path.join(_THIS_DIR, "data")
REPORT_DIR = os.path.join(_REPO_ROOT, "Results", "Reports", "linalg")

# Пороги: float32, реальные данные — допускаем относительную ошибку по результатам прогона
REL_ERR_THRESHOLD_85 = 2e-2   # 85×85 (~1.85e-2 observed)
REL_ERR_THRESHOLD_341 = 5e-2  # 341×341


# ============================================================================
# Helpers
# ============================================================================
```

## Connection (C++ ↔ Python)

- C++ class-card: `linalg__symmetrize_mode__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/linalg/t_matrix_csv_comparison.py`
- **Строк кода**: 227
- **Top-функций**: 4
- **Test runner**: common.runner

<!-- /rag-block -->
