---
id: dsp__strategies_strategies_step_by_step__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_strategies_step_by_step.py
primary_repo: strategies
module: strategies
uses_repos: ['core', 'strategies']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 343
title: Проверка gpu-пайплайна по шагам против numpy
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_strategies.AntennaProcessorTest
top_functions:
  - generate_signal_numpy
  - generate_weight_matrix_numpy
  - hamming_window
  - compute_nFFT
synonyms_ru:
  - тестирование пайплайна
  - сравнение gpu и numpy
  - пошаговая проверка
  - валидация сигналов
  - тестирование антенн
inherits_block_id: strategies__antenna_processor_test__class_overview__v1
block_refs:
  - strategies__antenna_processor_test__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_strategies_step_by_step__python_test_usecase__v1 -->

# Python use-case: Проверка gpu-пайплайна по шагам против numpy

## Цель

Проверяет каждый шаг gpu-пайплайна по сравнению с numpy, определяя точку расхождения.

## Когда применять

Запускать после изменений в antenprocessor_test или gpu-контексте.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_strategies.AntennaProcessorTest` | strategies |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
def generate_signal_numpy(n_ant, n_samples, fs, f0, amplitude, tau_base, tau_step):
    """Generate test signal matching FormSignalGeneratorROCm output."""
    dt = 1.0 / fs
    t = np.arange(n_samples) * dt
    S = np.zeros((n_ant, n_samples), dtype=np.complex64)
    for ant in range(n_ant):
        tau = tau_base + ant * tau_step
        t_delayed = t - tau
        valid = t_delayed >= 0
        S[ant, valid] = amplitude * np.exp(1j * 2 * np.pi * f0 * t_delayed[valid]).astype(np.complex64)
    return S


def generate_weight_matrix_numpy(n_ant, f0, tau_base, tau_step):
    """Generate delay-and-sum weight matrix W[beam][ant]."""
    W = np.zeros((n_ant, n_ant), dtype=np.complex64)
    inv_sqrt_n = 1.0 / np.sqrt(n_ant)
    for beam in range(n_ant):
        for ant in range(n_ant):
            tau = tau_base + ant * tau_step
            W[beam, ant] = inv_sqrt_n * np.exp(-1j * 2 * np.pi * f0 * tau)
    return W


def hamming_window(n):
```

## Connection (C++ ↔ Python)

- C++ class-card: `strategies__antenna_processor_test__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/strategies/t_strategies_step_by_step.py`
- **Строк кода**: 343
- **Top-функций**: 4
- **Test runner**: common.runner

<!-- /rag-block -->
