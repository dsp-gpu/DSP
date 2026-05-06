---
id: dsp__signal_generators_delayed_form_signal__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/signal_generators/t_delayed_form_signal.py
primary_repo: signal_generators
module: signal_generators
uses_repos: ['core', 'signal_generators']
uses_external: ['json', 'matplotlib', 'matplotlib.pyplot', 'numpy', 'warnings']
has_test_runner: true
is_opencl: false
line_count: 604
title: Тесты задержки сигнала GPU vs NumPy
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'fractional_delay', 'signal_generators', 'cross_repo']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_signal_generators.DelayedFormSignalGeneratorROCm
  - dsp_signal_generators.FormSignalGeneratorROCm
top_functions:
  - _require_gpu
  - load_lagrange_matrix
  - getX_numpy
  - apply_delay_numpy
  - test_integer_delay
  - test_fractional_delay
  - test_multichannel_delay
  - test_zero_delay
  - test_delay_with_noise
  - ensure_plot_dir
  - plot1_integer_delay
  - plot2_fractional_delay
  - plot3_multichannel_waterfall
  - plot4_delay_sweep
synonyms_ru:
  - задержка сигнала
  - дробная задержка
  - интерполяция лагранжа
  - тестирование gpu
  - сигналы с шумом
synonyms_en:
  - signal delay
  - fractional delay
  - lagrange interpolation
  - gpu testing
  - noisy signals
inherits_block_id: signal_generators__delayed_form_signal_generator_rocm__class_overview__v1
block_refs:
  - signal_generators__delayed_form_signal_generator_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__signal_generators_delayed_form_signal__python_test_usecase__v1 -->

# Python use-case: Тесты задержки сигнала GPU vs NumPy

## Цель

Проверка точности и производительности GPU-реализации задержки сигнала с интерполяцией Лагранжа против NumPy.

## Когда применять

Запускать после изменений в DelayedFormSignalGeneratorROCm или ROCmGPUContext.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_signal_generators.DelayedFormSignalGeneratorROCm` | signal_generators |
| `dsp_signal_generators.FormSignalGeneratorROCm` | signal_generators |

## Внешние зависимости

json, matplotlib, matplotlib.pyplot, numpy, warnings

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_signal_generators as signal_generators
    HAS_GPU = hasattr(signal_generators, 'DelayedFormSignalGeneratorROCm')
except ImportError:
    HAS_GPU = False
    core = None              # type: ignore
    signal_generators = None  # type: ignore


def _require_gpu():
    """Helper: единая точка проверки GPU."""
    if not HAS_GPU:
        raise SkipTest(
            "dsp_signal_generators.DelayedFormSignalGeneratorROCm not found — "
            "rebuild with ENABLE_ROCM=ON")


# ════════════════════════════════════════════════════════════════════════════
# Загрузка матрицы Lagrange 48×5 (data/ скопирована из spectrum/src/lch_farrow/)
# ════════════════════════════════════════════════════════════════════════════

MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'lagrange_matrix_48x5.json')
```

## Connection (C++ ↔ Python)

- C++ class-card: `signal_generators__delayed_form_signal_generator_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/signal_generators/t_delayed_form_signal.py`
- **Строк кода**: 604
- **Top-функций**: 14
- **Test runner**: common.runner

<!-- /rag-block -->
