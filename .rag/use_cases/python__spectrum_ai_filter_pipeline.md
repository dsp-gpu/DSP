---
id: dsp__spectrum_ai_filter_pipeline__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_ai_filter_pipeline.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['groq', 'json', 'matplotlib', 'matplotlib.pyplot', 'numpy', 'ollama', 're', 'scipy.signal', 'traceback']
has_test_runner: true
is_opencl: false
line_count: 916
title: тест ai фильтрации на gpu
tags: ['spectrum', 'ai_filter', 'gpu', 'signal_processing', 'natural_language', 'python']
uses_pybind:
  - dsp_core.GPUContext
  - dsp_spectrum.FirFilter
  - dsp_spectrum.IirFilter
top_functions:
  - _has_ai_backend
  - ai_ask
  - extract_json
  - parse_filter_request
  - sos_to_sections
  - design_filter
  - generate_test_signal
  - apply_filter_gpu
  - apply_filter_scipy
  - validate_results
  - plot_ai_results
  - run_ai_pipeline
  - demo_iir_lowpass
  - demo_fir_lowpass
  - demo_iir_highpass
  - demo_russian_request
synonyms_ru:
  - ai_фильтр
  - gpu_обработка
  - нейросеть_фильтр
  - пайплайн_дсп
  - естественный_язык
synonyms_en:
  - ai_filter
  - gpu_processing
  - neural_network_filter
  - dsp_pipeline
  - natural_language
inherits_block_id: spectrum__fir_filter__class_overview__v1
block_refs:
  - spectrum__fir_filter__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_ai_filter_pipeline__python_test_usecase__v1 -->

# Python use-case: тест ai фильтрации на gpu

## Цель

Проверяет интеграцию ai-парсинга и gpu-фильтров в пайплайне обработки сигналов

## Когда применять

Запускать после изменений в ai-backend или gpu-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.GPUContext` | core |
| `dsp_spectrum.FirFilter` | spectrum |
| `dsp_spectrum.IirFilter` | spectrum |

## Внешние зависимости

groq, json, matplotlib, matplotlib.pyplot, numpy, ollama, re, scipy.signal, traceback

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore
    print("WARNING: dsp_core/dsp_spectrum not found. GPU processing disabled.")

try:
    import scipy.signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("WARNING: matplotlib not found.")
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__fir_filter__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_ai_filter_pipeline.py`
- **Строк кода**: 916
- **Top-функций**: 16
- **Test runner**: common.runner

<!-- /rag-block -->
