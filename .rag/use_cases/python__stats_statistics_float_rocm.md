---
id: dsp__stats_statistics_float_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/stats/t_statistics_float_rocm.py
primary_repo: stats
module: stats
uses_repos: ['core', 'stats', 'spectrum']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 187
title: Statistics Float Rocm
tags: ['stats', 'python_test']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_stats.StatisticsProcessor
  - dsp_spectrum.ComplexToMagROCm
top_functions:
  - make_float_data
  - make_iq
inherits_block_id: stats__statistics_processor__class_overview__v1
block_refs:
  - stats__statistics_processor__class_overview__v1
ai_generated: true
human_verified: false
---

<!-- rag-block: id=dsp__stats_statistics_float_rocm__python_test_usecase__v1 -->

# Python use-case: Statistics Float Rocm

## Цель

Statistics Float ROCm — Python validation test

## Когда применять

Statistics Float ROCm — Python validation test

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_stats.StatisticsProcessor` | stats |
| `dsp_spectrum.ComplexToMagROCm` | spectrum |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    import dsp_stats as stats
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    stats = None     # type: ignore


# ============================================================================
# Helpers
# ============================================================================

RNG = np.random.default_rng(42)


def make_float_data(n_beams: int, n_points: int, lo: float = 0.5, hi: float = 5.0) -> np.ndarray:
    return RNG.uniform(lo, hi, size=(n_beams, n_points)).astype(np.float32)


def make_iq(n: int, amplitude: float = 1.0, fs: float = 44100.0, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / fs
    return (amplitude * np.exp(1j * 2.0 * np.pi * freq * t)).astype(np.complex64)
```

## Connection (C++ ↔ Python)

- C++ class-card: `stats__statistics_processor__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/stats/t_statistics_float_rocm.py`
- **Строк кода**: 187
- **Top-функций**: 2
- **Test runner**: common.runner

<!-- /rag-block -->
