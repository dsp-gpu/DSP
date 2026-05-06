---
id: dsp__integration_signal_to_spectrum__cross_repo_pipeline__v1
type: cross_repo_pipeline
source_path: DSP/Python/integration/t_signal_to_spectrum.py
primary_repo: integration
module: integration
uses_repos: ['core', 'spectrum']
uses_external: ['integration.factories', 'matplotlib', 'matplotlib.gridspec', 'matplotlib.pyplot', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 596
title: Сигнал к спектру GPU
tags: ['signal_processing', 'fft', 'gpu', 'python', 'rocm', 'cross_repo', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.FFTProcessorROCm
  - dsp_core.list_gpus
top_functions:
  - print_header
  - _require_gpu
  - test_multichannel_sin_fft
  - test_signal_types
  - test_multibeam_cw
  - test_generators_from_string
  - test_multibeam_from_string
  - test_mag_phase
  - test_generate_from_string
synonyms_ru:
  - тест сигнал спектр
  - fft сигнал
  - gpu спектральный анализ
  - сигнал в спектр
  - dsp_gpu тест
synonyms_en:
  - signal_to_spectrum python_test
  - fft signal processing
  - gpu spectral analysis
  - signal spectrum conversion
  - dsp_gpu signal python_test
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__integration_signal_to_spectrum__cross_repo_pipeline__v1 -->

# Python use-case: Сигнал к спектру GPU

## Цель

Проверка корректности обработки сигналов через GPU-FFT и визуализацию

## Когда применять

Запускать после изменений в SignalGenerator, FFTProcessor или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.FFTProcessorROCm` | spectrum |
| `dsp_core.list_gpus` | core |

## Внешние зависимости

integration.factories, matplotlib, matplotlib.gridspec, matplotlib.pyplot, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore

# NumPy-обёртка для legacy SignalGenerator API (см. factories.py)
from integration.factories import _NumpySignalGenerator as _SignalGenerator


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _require_gpu():
    """Helper: единая точка проверки GPU."""
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_spectrum not found — check build/libs")


# ============================================================================
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/integration/t_signal_to_spectrum.py`
- **Строк кода**: 596
- **Top-функций**: 9
- **Test runner**: common.runner

<!-- /rag-block -->
