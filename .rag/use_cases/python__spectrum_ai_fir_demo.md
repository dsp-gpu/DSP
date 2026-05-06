---
id: dsp__spectrum_ai_fir_demo__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_ai_fir_demo.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['groq', 'json', 'matplotlib', 'matplotlib.pyplot', 'numpy', 'ollama', 're', 'scipy']
has_test_runner: true
is_opencl: false
line_count: 590
title: AI FIR-фильтр dsp_gpu-GPU
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'filter', 'fft', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.FFTProcessorROCm
top_functions:
  - ai_ask
  - extract_json
  - parse_filter_request
  - design_fir_filter
  - apply_filter
  - validate_filter
  - plot_results
  - run_pipeline
synonyms_ru:
  - ai_fir
  - dsp_gpu
  - фильтр_ai
  - gpu_фильтр
  - ai_фильтр
synonyms_en:
  - ai_fir
  - dsp_gpu
  - filter_integration
  - ai_filter
  - gpu_filter
inherits_block_id: spectrum__fft_processor_rocm__class_overview__v1
block_refs:
  - spectrum__fft_processor_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_ai_fir_demo__python_test_usecase__v1 -->

# Python use-case: AI FIR-фильтр dsp_gpu-GPU

## Цель

Проверка интеграции AI-генератора коэффициентов с GPU-реализацией FIR-фильтра и визуализацией результатов.

## Когда применять

Запускать после изменений в ROCmGPUContext, FFTProcessorROCm или AI-интерфейсах.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.FFTProcessorROCm` | spectrum |

## Внешние зависимости

groq, json, matplotlib, matplotlib.pyplot, numpy, ollama, re, scipy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_signal_generators as signal_generators
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None              # type: ignore
    signal_generators = None  # type: ignore
    spectrum = None          # type: ignore

# ════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════════════════════

MODE = "none"           # "groq" | "ollama" | "none"
OLLAMA_MODEL = "qwen2.5-coder:7b"

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Plots', 'filters')

# ── API Key — читается из api_keys.json (в корне проекта) ──────────────────
# Создай файл:  <корень проекта>/api_keys.json  со следующим содержимым:
#
#   {
#       "api": "sm_ВАШ_КЛЮЧ_ЗДЕСЬ"
#   }
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__fft_processor_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_ai_fir_demo.py`
- **Строк кода**: 590
- **Top-функций**: 8
- **Test runner**: common.runner

<!-- /rag-block -->
