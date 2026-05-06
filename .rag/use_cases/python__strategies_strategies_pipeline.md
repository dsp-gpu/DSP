---
id: dsp__strategies_strategies_pipeline__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_strategies_pipeline.py
primary_repo: strategies
module: strategies
uses_repos: ['core', 'strategies']
uses_external: ['numpy_reference', 'pipeline_step_validator', 'signal_factory']
has_test_runner: true
is_opencl: false
line_count: 256
title: Тест стратегий обработки сигналов
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'pipeline', 'strategies', 'fft']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_strategies.AntennaProcessorTest
synonyms_ru:
  - тест пайплайна
  - стратегии обработки
  - сигналы
  - тестирование
  - проверка
synonyms_en:
  - pipeline python_test
  - signal processing
  - strategy python_test
  - validation
  - python_test suite
inherits_block_id: strategies__antenna_processor_test__class_overview__v1
block_refs:
  - strategies__antenna_processor_test__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_strategies_pipeline__python_test_usecase__v1 -->

# Python use-case: Тест стратегий обработки сигналов

## Цель

Проверка корректности работы пайплайна стратегий обработки сигналов для 5 вариантов входных данных.

## Когда применять

Запускать после изменений в AntennaProcessorTest или GPU-контексте, а также при проверке обработки сигналов в pipeline.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_strategies.AntennaProcessorTest` | strategies |

## Внешние зависимости

numpy_reference, pipeline_step_validator, signal_factory

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_strategies as strategies
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None        # type: ignore
    strategies = None  # type: ignore

from numpy_reference import NumpyReference
from signal_factory import SignalSourceFactory, SignalVariant, SignalConfig
from pipeline_step_validator import PipelineStepValidator


# ── Дефолтные параметры сценария ─────────────────────────────────────────────

_DEFAULT_CFG = SignalConfig(
    n_ant     = 5,
    n_samples = 8000,
    fs        = 12e6,
    f0        = 2e6,
    tau_step  = 100e-6,
    snr_db    = 20.0,
    n_fft     = 8192,
)
```

## Connection (C++ ↔ Python)

- C++ class-card: `strategies__antenna_processor_test__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/strategies/t_strategies_pipeline.py`
- **Строк кода**: 256
- **Top-функций**: 0
- **Test runner**: common.runner

<!-- /rag-block -->
