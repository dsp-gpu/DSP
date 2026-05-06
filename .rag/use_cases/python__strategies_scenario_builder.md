---
id: dsp__strategies_scenario_builder__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_scenario_builder.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['numpy', 'scenario_builder']
has_test_runner: false
is_opencl: false
line_count: 507
title: Проверка модели ULA и генерации сигналов
tags: ['signal_processing', 'antenna_array', 'uca', 'signal_generation', 'validation', 'numpy', 'strategies']
uses_pybind: []
top_functions:
  - fft_peak_freq
  - fft_bandwidth_3dB
synonyms_ru:
  - тестирование антенной решётки
  - проверка генерации сигналов
  - валидация ULA
  - тестирование сигналов
  - проверка модели
synonyms_en:
  - antenna array testing
  - signal generation validation
  - ULA verification
  - signal testing
  - model validation
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_scenario_builder__python_test_usecase__v1 -->

# Python use-case: Проверка модели ULA и генерации сигналов

## Цель

Проверяет корректность физической модели антенной решётки (ULA) и генератора сигналов с использованием NumPy.

## Когда применять

Запускать после изменений в модуле strategies или связанных компонентах.

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

numpy, scenario_builder

## Solution (фрагмент кода)

```python
def fft_peak_freq(signal_1d: np.ndarray, fs: float) -> float:
    """Найти частоту доминирующего пика в спектре (первая половина)."""
    spectrum = np.fft.fft(signal_1d)
    magnitudes = np.abs(spectrum)
    half = len(magnitudes) // 2
    peak_bin = np.argmax(magnitudes[1:half]) + 1
    return peak_bin * fs / len(signal_1d)


def fft_bandwidth_3dB(signal_1d: np.ndarray, fs: float) -> float:
    """Оценить полосу по уровню -3 дБ от пика."""
    spectrum = np.fft.fft(signal_1d)
    magnitudes = np.abs(spectrum)
    half = len(magnitudes) // 2
    mags = magnitudes[:half]

    peak_val = np.max(mags)
    threshold = peak_val / np.sqrt(2.0)  # -3 дБ

    above = np.where(mags > threshold)[0]
    if len(above) < 2:
        return 0.0

    freq_resolution = fs / len(signal_1d)
    return (above[-1] - above[0]) * freq_resolution
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_scenario_builder.py`
- **Строк кода**: 507
- **Top-функций**: 2
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
