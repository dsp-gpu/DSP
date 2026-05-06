---
id: dsp__signal_generators_form_signal__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/signal_generators/t_form_signal.py
primary_repo: signal_generators
module: signal_generators
uses_repos: ['core', 'signal_generators']
uses_external: ['matplotlib', 'matplotlib.gridspec', 'matplotlib.pyplot', 'numpy', 'traceback']
has_test_runner: true
is_opencl: false
line_count: 737
title: тестирование генератора сигналов
tags: ['signal_generators', 'gpu', 'python', 'signal_processing', 'fft', 'cross_repo', 'lfm']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_signal_generators.FormSignalGeneratorROCm
  - dsp_core.list_gpus
top_functions:
  - _require_gpu
  - getX_numpy
  - test_cw_no_noise
  - test_chirp
  - test_window
  - test_multi_channel
  - test_noise_statistics
  - test_string_params
  - test_signal_plus_noise
  - make_plots
  - main
synonyms_ru:
  - тест генератора сигналов
  - сравнение gpu и numpy
  - генерация сигналов
  - тестирование сигналов
  - проверка генератора
inherits_block_id: signal_generators__form_signal_generator_rocm__class_overview__v1
block_refs:
  - signal_generators__form_signal_generator_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__signal_generators_form_signal__python_test_usecase__v1 -->

# Python use-case: тестирование генератора сигналов

## Цель

Проверка корректности GPU-генератора сигналов против NumPy с визуализацией результатов

## Когда применять

Запускать после изменений в FormSignalGenerator или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_signal_generators.FormSignalGeneratorROCm` | signal_generators |
| `dsp_core.list_gpus` | core |

## Внешние зависимости

matplotlib, matplotlib.gridspec, matplotlib.pyplot, numpy, traceback

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_signal_generators as signal_generators
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None              # type: ignore
    signal_generators = None  # type: ignore


def _require_gpu():
    """Helper: единая точка проверки GPU."""
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_signal_generators not found — check build/libs")


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: getX formula
# ════════════════════════════════════════════════════════════════════════════

def getX_numpy(fs, points, f0, amplitude, phase, fdev, norm_val, tau=0.0):
    """
    CPU reference (NumPy) — формула getX без шума.
    """
    dt = 1.0 / fs
    ti = points * dt
```

## Connection (C++ ↔ Python)

- C++ class-card: `signal_generators__form_signal_generator_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/signal_generators/t_form_signal.py`
- **Строк кода**: 737
- **Top-функций**: 11
- **Test runner**: common.runner

<!-- /rag-block -->
