---
id: dsp__spectrum_fir_filter_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_fir_filter_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 258
title: GPU FIR фильтр ROCm vs scipy
tags: ['spectrum', 'gpu', 'python', 'signal_processing', 'filter', 'fir', 'fft', 'rocm']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.FirFilterROCm
top_functions:
  - make_complex_signal
  - make_two_tone
  - scipy_fir_ref
  - make_ctx_fir
  - test_fir_single_channel_basic
  - test_fir_multi_channel
  - test_fir_all_pass
  - test_fir_lowpass_attenuation
  - test_fir_properties
synonyms_ru:
  - тест фильтра ROCm
  - сравнение GPU/scipy
  - реализация FIR
  - валидация фильтра
  - тестирование сигналов
synonyms_en:
  - rocm filter python_test
  - gpu vs scipy benchmark
  - fir implementation validation
  - signal processing verification
  - filter comparison
inherits_block_id: spectrum__fir_filter_rocm__class_overview__v1
block_refs:
  - spectrum__fir_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_fir_filter_rocm__python_test_usecase__v1 -->

# Python use-case: GPU FIR фильтр ROCm vs scipy

## Цель

Проверка корректности GPU-реализации FIR-фильтра ROCm против scipy.signal.lfilter

## Когда применять

Запускать после изменений в fir_filter_rocm.hip или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.FirFilterROCm` | spectrum |

## Внешние зависимости

numpy, scipy.signal

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print(f"WARNING: dsp_core/dsp_spectrum not found. Skipping GPU tests. (searched: {GPULoader.loaded_from()})")

try:
    import scipy.signal as ss
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Skipping validation tests.")

# ============================================================================
# Parameters
# ============================================================================

SAMPLE_RATE = 50_000.0    # Hz
POINTS      = 4096
CHANNELS    = 8
FIR_TAPS    = 64
FIR_CUTOFF  = 0.1         # normalized (0-1, Nyquist=1)
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__fir_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_fir_filter_rocm.py`
- **Строк кода**: 258
- **Top-функций**: 9
- **Test runner**: common.runner

<!-- /rag-block -->
