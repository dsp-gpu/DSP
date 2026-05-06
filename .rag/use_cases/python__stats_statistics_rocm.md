---
id: dsp__stats_statistics_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/stats/t_statistics_rocm.py
primary_repo: stats
module: stats
uses_repos: []
uses_external: ['matplotlib', 'matplotlib.pyplot', 'numpy', 're', 'subprocess', 'time']
has_test_runner: false
is_opencl: false
line_count: 682
title: Проверка статистики ROCm против NumPy
tags: ['stats', 'rocm', 'gpu', 'python', 'signal_processing', 'statistics', 'validation']
uses_pybind: []
top_functions:
  - make_sinusoid
  - make_multi_beam
  - ref_mean
  - ref_mean_mag
  - ref_variance_mag
  - ref_std_mag
  - ref_median_mag
  - test_numpy_mean_single_beam
  - test_numpy_mean_multi_beam
  - test_numpy_welford_statistics
  - test_numpy_median_linear
  - test_numpy_mean_constant
  - run_gpu_binary
  - parse_output
  - _parse_float
  - test_numpy_histogram_median_basic
  - test_numpy_histogram_median_random
  - test_gpu_all_pass
  - test_gpu_benchmark_speedup
  - test_gpu_vs_numpy_welford
  - test_gpu_vs_numpy_median
  - plot_reference_values
synonyms_ru:
  - тестирование статистики gpu
  - валидация rocm
  - сравнение numpy
  - тесты статистики
  - проверка вычислений
synonyms_en:
  - gpu_statistics_test
  - numpy_validation
  - statistical_computation_check
  - test_statistics
  - rocml_validation
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__stats_statistics_rocm__python_test_usecase__v1 -->

# Python use-case: Проверка статистики ROCm против NumPy

## Цель

Проверка результатов обработки статистики на GPU против NumPy

## Когда применять

Запускать после сборки бинарного файла test_stats_rocm и настройки группы рендеринга

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

matplotlib, matplotlib.pyplot, numpy, re, subprocess, time

## Solution (фрагмент кода)

```python
import re
import subprocess
import sys
import time

import numpy as np

# ============================================================================
# Project paths
# ============================================================================

# Python_test/statistics/ -> 2 levels up -> project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# Legacy: GPUWorkLib monolith → DSP-GPU: stats/build/test_stats_rocm
BINARY_PATH = os.path.join(PROJECT_ROOT, "stats", "build", "test_stats_rocm")
HAS_BINARY = os.path.exists(BINARY_PATH)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/stats/t_statistics_rocm.py`
- **Строк кода**: 682
- **Top-функций**: 22
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
