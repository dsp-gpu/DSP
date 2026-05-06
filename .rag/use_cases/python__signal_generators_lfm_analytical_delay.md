---
id: dsp__signal_generators_lfm_analytical_delay__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/signal_generators/t_lfm_analytical_delay.py
primary_repo: signal_generators
module: signal_generators
uses_repos: ['core', 'signal_generators']
uses_external: ['matplotlib', 'matplotlib.pyplot', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 440
title: аналитическая задержка lfm
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'lfm', 'signal_generators', 'filter']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_signal_generators.LfmAnalyticalDelayROCm
top_functions:
  - _require_gpu
  - lfm_analytical_numpy
  - test_zero_delay_vs_standard_lfm
  - test_fractional_delay_boundary
  - test_gpu_vs_cpu
  - test_multi_antenna
  - test_gpu_vs_numpy
  - ensure_plot_dir
  - make_plots
synonyms_ru:
  - задержка lfm
  - аналитический сдвиг
  - lfm задержка
  - дисперсия сигнала
  - временная сдвиг
inherits_block_id: signal_generators__lfm_analytical_delay_rocm__class_overview__v1
block_refs:
  - signal_generators__lfm_analytical_delay_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__signal_generators_lfm_analytical_delay__python_test_usecase__v1 -->

# Python use-case: аналитическая задержка lfm

## Цель

Проверка корректности аналитической задержки LFM на GPU против CPU и NumPy, а также многоканальных сценариев

## Когда применять

Запускать после изменений в LfmGeneratorAnalyticalDelay или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_signal_generators.LfmAnalyticalDelayROCm` | signal_generators |

## Внешние зависимости

matplotlib, matplotlib.pyplot, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_signal_generators as signal_generators
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None              # type: ignore
    signal_generators = None  # type: ignore

# Ленивая инициализация GPU контекста — создаётся в первом тесте, переиспользуется
ctx = None


def _require_gpu():
    """Helper: единая точка проверки GPU. Создаёт ctx при первом вызове."""
    global ctx
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_signal_generators not found — check build/libs")
    if ctx is None:
        ctx = core.ROCmGPUContext(0)
        print(f"GPU: {ctx.device_name}")


# ════════════════════════════════════════════════════════════════════════════
# NumPy reference: analytical LFM with delay
# ════════════════════════════════════════════════════════════════════════════
```

## Connection (C++ ↔ Python)

- C++ class-card: `signal_generators__lfm_analytical_delay_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/signal_generators/t_lfm_analytical_delay.py`
- **Строк кода**: 440
- **Top-функций**: 9
- **Test runner**: common.runner

<!-- /rag-block -->
