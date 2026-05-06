---
id: dsp__heterodyne_heterodyne__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/heterodyne/t_heterodyne.py
primary_repo: heterodyne
module: heterodyne
uses_repos: ['core', 'heterodyne']
uses_external: ['matplotlib', 'matplotlib.gridspec', 'matplotlib.pyplot', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 278
title: Тесты гетеродинного преобразования на GPU
tags: ['heterodyne', 'rocm', 'gpu', 'signal_processing', 'filter', 'fft', 'lfm', 'dechirp', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_heterodyne.HeterodyneROCm
top_functions:
  - generate_lfm_rx
  - generate_lfm_reference
  - heterodyne_pipeline
synonyms_ru:
  - тесты гетеродинного преобразования
  - тесты lfm дехирп
  - тесты rocm
  - тесты gpu обработки сигналов
  - тесты дехирп
synonyms_en:
  - heterodyne tests
  - lfm dechirp tests
  - rocm tests
  - gpu signal processing tests
  - dechirp verification
inherits_block_id: heterodyne__heterodyne_rocm__class_overview__v1
block_refs:
  - heterodyne__heterodyne_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__heterodyne_heterodyne__python_test_usecase__v1 -->

# Python use-case: Тесты гетеродинного преобразования на GPU

## Цель

Проверка корректности гетеродинного преобразования LFM с GPU против legacy версии

## Когда применять

Запускать после изменений в HeterodyneROCm или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_heterodyne.HeterodyneROCm` | heterodyne |

## Внешние зависимости

matplotlib, matplotlib.gridspec, matplotlib.pyplot, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_heterodyne as heterodyne
    HAS_HETERODYNE = hasattr(heterodyne, 'HeterodyneROCm')
except ImportError:
    HAS_HETERODYNE = False
    core = None        # type: ignore
    heterodyne = None  # type: ignore


# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6           # sample rate, Hz
F_START = 0.0       # LFM start frequency, Hz
F_END = 2e6         # LFM end frequency, Hz
N = 8000            # samples per antenna
ANTENNAS = 5
BANDWIDTH = F_END - F_START  # 2 MHz
DURATION = N / FS            # 666.67 us
MU = BANDWIDTH / DURATION    # 3e9 Hz/s  (chirp rate)
C_LIGHT = 3e8                # speed of light, m/s

DELAYS_US = [100, 200, 300, 400, 500]  # delays in microseconds
F_BEAT_TOL_HZ = 5000.0                 # tolerance +/- 5 kHz
```

## Connection (C++ ↔ Python)

- C++ class-card: `heterodyne__heterodyne_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/heterodyne/t_heterodyne.py`
- **Строк кода**: 278
- **Top-функций**: 3
- **Test runner**: common.runner

<!-- /rag-block -->
