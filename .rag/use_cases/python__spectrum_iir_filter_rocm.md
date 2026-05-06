---
id: dsp__spectrum_iir_filter_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_iir_filter_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 283
title: iir фильтр gpu vs scipy
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.IirFilterROCm
top_functions:
  - make_complex_signal
  - make_two_tone
  - sos_to_sections
  - scipy_iir_ref
  - make_ctx_iir
  - test_iir_single_channel_basic
  - test_iir_multi_channel
  - test_iir_zero_input
  - test_iir_lowpass_attenuation
  - test_iir_properties
synonyms_ru:
  - iir фильтр gpu
  - gpu сравнение scipy
  - scipy проверка
  - бикуадратный фильтр
  - rocm тест
inherits_block_id: spectrum__iir_filter_rocm__class_overview__v1
block_refs:
  - spectrum__iir_filter_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_iir_filter_rocm__python_test_usecase__v1 -->

# Python use-case: iir фильтр gpu vs scipy

## Цель

Тестирование gpu-реализации iir-фильтра (rocm) против эталонного scipy.signal.sosfilt

## Когда применять

Запускать после изменений в iir_filter_rocm.hip или gpu-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.IirFilterROCm` | spectrum |

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
IIR_ORDER   = 2
IIR_CUTOFF  = 0.1         # normalized (0-1, Nyquist=1)
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__iir_filter_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_iir_filter_rocm.py`
- **Строк кода**: 283
- **Top-функций**: 10
- **Test runner**: common.runner

<!-- /rag-block -->
