---
id: dsp__stats_compute_all__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/stats/t_compute_all.py
primary_repo: stats
module: stats
uses_repos: []
uses_external: ['matplotlib', 'matplotlib.pyplot', 'numpy', 're', 'subprocess', 'time']
has_test_runner: false
is_opencl: false
line_count: 425
title: Проверка computeall против numpy
tags: ['stats', 'rocm', 'gpu', 'python', 'signal_processing', 'computeall', 'cross_repo']
uses_pybind: []
top_functions:
  - make_complex_data
  - make_float_data
  - ref_statistics
  - ref_median
  - ref_statistics_float
  - ref_median_float
  - test_compute_all_matches_separate
  - test_compute_all_float_matches
  - test_compute_all_float_mean_is_zero
  - test_compute_all_timing_reference
  - run_gpu_binary
  - parse_gpu_output
  - test_gpu_tests_all_pass
  - test_gpu_compute_all_error
synonyms_ru:
  - тестирование computeall
  - валидация статистики
  - сравнение numpy
  - тесты gpu
  - проверка бинарного вывода
synonyms_en:
  - computeall validation
  - numpy comparison
  - gpu testing
  - binary output check
  - statistics verification
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__stats_compute_all__python_test_usecase__v1 -->

# Python use-case: Проверка computeall против numpy

## Цель

Проверяет корректность результатов ComputeAll и ComputeAllFloat против NumPy и C++ бинарного вывода

## Когда применять

Запускать после изменений в ComputeAll или при наличии GPU-бинарного вывода

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

matplotlib, matplotlib.pyplot, numpy, re, subprocess, time

## Solution (фрагмент кода)

```python
def make_complex_data(beam_count: int, n_point: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.uniform(-1.0, 1.0, beam_count * n_point).astype(np.float32)
    imag = rng.uniform(-1.0, 1.0, beam_count * n_point).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def make_float_data(beam_count: int, n_point: int, seed: int = 999) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, beam_count * n_point).astype(np.float32)


def ref_statistics(data: np.ndarray, beam_count: int) -> list[dict]:
    """Compute mean + variance + std + mean_mag per beam (matches C++ Welford)."""
    n = len(data) // beam_count
    results = []
    for b in range(beam_count):
        beam = data[b * n:(b + 1) * n]
        mags = np.abs(beam)
        results.append({
            "beam_id":        b,
            "mean_real":      float(np.mean(beam).real),
            "mean_imag":      float(np.mean(beam).imag),
            "variance":       float(np.var(mags, ddof=0)),
            "std_dev":        float(np.std(mags, ddof=0)),
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/stats/t_compute_all.py`
- **Строк кода**: 425
- **Top-функций**: 14
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
