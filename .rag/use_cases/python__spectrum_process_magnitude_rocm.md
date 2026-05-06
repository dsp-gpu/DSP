---
id: dsp__spectrum_process_magnitude_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_process_magnitude_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum', 'stats']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 167
title: Проверка процесса модуля на GPU
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'fft', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.ComplexToMagROCm
  - dsp_stats.StatisticsProcessor
top_functions:
  - make_iq
synonyms_ru:
  - тест модуля
  - проверка gpu
  - сравнение numpy
  - обработка сигнала
  - комплексный модуль
inherits_block_id: spectrum__complex_to_mag_rocm__class_overview__v1
block_refs:
  - spectrum__complex_to_mag_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_process_magnitude_rocm__python_test_usecase__v1 -->

# Python use-case: Проверка процесса модуля на GPU

## Цель

Проверка корректности вычисления модуля комплексного сигнала на GPU с использованием NumPy в качестве эталона.

## Когда применять

Запускать после изменений в ComplexToMagROCm или GPU-контексте.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.ComplexToMagROCm` | spectrum |
| `dsp_stats.StatisticsProcessor` | stats |

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

FS = 44100.0
FREQ = 1000.0


def make_iq(n: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate complex sinusoid IQ signal."""
    t = np.arange(n, dtype=np.float32) / FS
    ph = 2.0 * np.pi * FREQ * t
    return (amplitude * np.exp(1j * ph)).astype(np.complex64)
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__complex_to_mag_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_process_magnitude_rocm.py`
- **Строк кода**: 167
- **Top-функций**: 1
- **Test runner**: common.runner

<!-- /rag-block -->
