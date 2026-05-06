---
id: dsp__strategies_params__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_params.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['math', 'signal_factory']
has_test_runner: false
is_opencl: false
line_count: 117
title: параметры тестов антенных стратегий
tags: ['strategies', 'test_params', 'antenna', 'signal_processing', 'python', 'dataclass', 'params']
uses_pybind: []
synonyms_ru:
  - конфигурация тестов
  - параметры антенных стратегий
  - настройки тестов
  - данные тестов
  - опции антенных стратегий
synonyms_en:
  - python_test configuration
  - antenna strategy parameters
  - python_test settings
  - python_test data
  - strategy options
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_params__python_test_usecase__v1 -->

# Python use-case: параметры тестов антенных стратегий

## Цель

предоставляет общие параметры для тестов антенных стратегий, объединяя размеры, частоты и задержки

## Когда применять

запускать после изменений в antenntestparams или при настройке параметров тестов

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

math, signal_factory

## Solution (фрагмент кода)

```python
class AntennaTestParams:
    """Параметры теста антенной стратегии.

    Использование:
        params = AntennaTestParams.small()      # быстрый тест
        params = AntennaTestParams.full_spec()  # полный тест (2500×5000)
    """
    # Размеры
    n_ant:     int   = 100         # число антенн (small по умолчанию)
    n_samples: int   = 5000        # отсчётов на антенну
    n_beams:   int   = 100         # столбцы матрицы W (=n_ant → квадратная)

    # Частоты
    fs:        float = 0.5e6       # частота дискретизации, Гц
    fdev_hz:   float = 90e3        # девиация ЛЧМ, Гц
    f0_hz:     float = 100e3       # целевая частота для валидации

    # Задержки
    tau_step_us: float = 2.0       # шаг задержки на антенну, мкс

    # Вариант сигнала
    signal_variant: SignalVariant = SignalVariant.SIN

    # Опции вывода
    save_to_files: bool = False
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_params.py`
- **Строк кода**: 117
- **Top-функций**: 0
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
