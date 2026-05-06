---
id: dsp__strategies_farrow_pipeline__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_farrow_pipeline.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['farrow_delay', 'numpy', 'pipeline_runner', 'scenario_builder', 'tempfile']
has_test_runner: false
is_opencl: false
line_count: 470
title: лчм обработка пайплайн a vs b
tags: ['signal_processing', 'python', 'lfm', 'strategies', 'filter', 'cross_repo', 'pipeline']
uses_pybind: []
synonyms_ru:
  - лчм пайплайн сравнение
  - фарроу задержка тест
  - сигналы лчм обработка
  - пайплайн a vs b
  - временная коррекция лчм
synonyms_en:
  - lfm_pipeline_comparison
  - farrow_delay_test
  - lfm_signal_processing
  - pipeline_a_vs_b
  - temporal_correction
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_farrow_pipeline__python_test_usecase__v1 -->

# Python use-case: лчм обработка пайплайн a vs b

## Цель

Проверяет точность пайплайна b (farrow) для лчм сигналов по сравнению с пайплайном a.

## Когда применять

Запускать после изменений в farrow_delay или pipeline_runner.

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

farrow_delay, numpy, pipeline_runner, scenario_builder, tempfile

## Solution (фрагмент кода)

```python
class TestFarrowDelay:

    def test_farrow_identity(self):
        """delay=0 → сигнал не меняется."""
        farrow = FarrowDelay()
        rng = np.random.default_rng(42)
        signal = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
        signal = signal.reshape(1, -1)

        result = farrow.apply(signal, np.array([0.0]))
        np.testing.assert_allclose(np.abs(result), np.abs(signal), atol=1e-4)

    def test_farrow_integer_delay(self):
        """Целая задержка → точный сдвиг."""
        farrow = FarrowDelay()
        n = 100
        signal = np.zeros((1, n), dtype=np.complex64)
        signal[0, 10] = 1.0 + 0j  # импульс

        delayed = farrow.apply(signal, np.array([5.0]))
        peak_pos = np.argmax(np.abs(delayed[0]))
        assert peak_pos == 15, f"Peak at {peak_pos}, expected 15"

    def test_farrow_compensate(self):
        """compensate() применяет отрицательную задержку — проверяем сдвиг импульса.
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_farrow_pipeline.py`
- **Строк кода**: 470
- **Top-функций**: 0
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
