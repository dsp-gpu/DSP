---
id: dsp__strategies_base_pipeline__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_base_pipeline.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['numpy', 'signal_generators_strategy', 'strategy_base', 't_params']
has_test_runner: false
is_opencl: false
line_count: 163
title: Проверка NumPy-пайплайна без GPU
tags: ['strategies', 'signal_processing', 'numpy', 'pipeline', 'math_check', 'cpu_only', 'strategy_base']
uses_pybind: []
top_functions:
  - _run_variant
  - test_sin_full_pipeline
  - test_lfm_no_delay_pipeline
  - test_lfm_delay_pipeline
  - test_lfm_farrow_pipeline
  - test_all_variants
synonyms_ru:
  - проверка пайплайна
  - тестирование алгоритма
  - numpy тест
  - проверка математики
  - без gpu
synonyms_en:
  - numpy pipeline python_test
  - algorithm validation
  - math check
  - baseline python_test
  - cpu-only python_test
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_base_pipeline__python_test_usecase__v1 -->

# Python use-case: Проверка NumPy-пайплайна без GPU

## Цель

Проверяет правильность алгоритма (GEMM, Hamming, FFT, поиск пика) на NumPy перед запуском на GPU

## Когда применять

Запускать после изменений в pipeline или параметрах стратегии

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

numpy, signal_generators_strategy, strategy_base, t_params

## Solution (фрагмент кода)

```python
class NumpyPipelineTest(StrategyTestBase):
    """Полный NumPy pipeline тест (T1).

    process() запускает _run_numpy_pipeline().
    validate() проверяет peak_freq и dynamic_range.
    """

    MIN_DYNAMIC_RANGE_DB: float = 20.0

    def process(self, data: np.ndarray, ctx) -> dict:
        return self._run_numpy_pipeline(data)

    def validate(self, result: dict, params: AntennaTestParams) -> TestResult:
        tr = TestResult(test_name=self.name)

        peak_freq = result["peak_freq_hz"]
        bin_hz    = params.bin_hz
        freq_err  = abs(peak_freq - params.f0_hz)

        # Проверка 1: частота пика ≈ f0 (только для SIN/CW; LFM без дечирпа не даёт чёткий пик)
        if params.check_peak_freq:
            tr.add(ValidationResult(
                passed    = freq_err < 2.0 * bin_hz,
                metric_name = "peak_freq_error_hz",
                actual_value = freq_err,
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_base_pipeline.py`
- **Строк кода**: 163
- **Top-функций**: 6
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
