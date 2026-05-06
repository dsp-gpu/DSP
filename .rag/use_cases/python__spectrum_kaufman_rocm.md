---
id: dsp__spectrum_kaufman_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_kaufman_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 551
title: kama фильтр gpu vs numpy
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'kama', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.KaufmanFilterROCm
top_functions:
  - _kaufman_1ch
  - kaufman_ref
  - make_complex_signal
  - make_ctx_kauf
  - test_kaufman_basic
  - test_kaufman_multi_channel
  - test_kaufman_trend_signal
  - test_kaufman_noise_signal
  - test_kaufman_adaptive_transition
  - test_kaufman_channel_independence
  - test_kaufman_step_demo
  - test_kaufman_properties
synonyms_ru:
  - тест кума
  - тест ками
  - тест адапт
  - тест фильтр
  - тест gpu
synonyms_en:
  - kaufman python_test
  - kama python_test
  - adaptive filter python_test
  - gpu filter python_test
  - signal processing python_test
inherits_block_id: spectrum__kaufman_filter_rocm__class_overview__v1
block_refs:
  - spectrum__kaufman_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_kaufman_rocm__python_test_usecase__v1 -->

# Python use-case: kama фильтр gpu vs numpy

## Цель

Проверяет корректность gpu-реализации kama-фильтра против numpy на различных сигналах

## Когда применять

Запускать после обновления gpu-контекста или реализации kama-фильтра

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.KaufmanFilterROCm` | spectrum |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print("WARNING: dsp_core/dsp_spectrum not found. Skipping GPU tests.")

# ============================================================================
# Parameters
# ============================================================================

POINTS      = 4096
CHANNELS    = 8
ER_PERIOD   = 10    # N — Kaufman default
FAST_PERIOD = 2     # fast EMA period (ER=1)
SLOW_PERIOD = 30    # slow EMA period (ER=0)
ATOL        = 1e-4  # float32 tolerance

# Precomputed SCs for reference
FAST_SC = 2.0 / (FAST_PERIOD + 1)   # = 2/3 ≈ 0.6667
SLOW_SC = 2.0 / (SLOW_PERIOD + 1)   # = 2/31 ≈ 0.0645

# ============================================================================
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__kaufman_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_kaufman_rocm.py`
- **Строк кода**: 551
- **Top-функций**: 12
- **Test runner**: common.runner

<!-- /rag-block -->
