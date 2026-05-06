---
id: dsp__strategies_debug_steps__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_debug_steps.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['math', 'numpy', 'signal_generators_strategy', 't_params']
has_test_runner: true
is_opencl: false
line_count: 183
title: Проверка шагов NumPy pipeline
tags: ['signal_processing', 'python', 'fft', 'python_test', 'pipeline', 'strategies', 'debug']
uses_pybind: []
top_functions:
  - _generate_and_run
  - _parabolic_peak
  - test_gemm_shape_and_gain
  - test_fft_peak_location
  - test_one_max_accuracy
  - test_minmax_dynamic_range_loop
  - _test_minmax_dynamic_range_single
synonyms_ru:
  - тестирование шагов
  - проверка этапов
  - анализ пайплайна
  - отладка процесса
  - контроль этапов
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_debug_steps__python_test_usecase__v1 -->

# Python use-case: Проверка шагов NumPy pipeline

## Цель

Проверка отдельных этапов NumPy-пайплайна с числовыми критериями для выявления ошибок.

## Когда применять

Запускать при сбое test_base_pipeline.py для идентификации проблемного шага.

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

math, numpy, signal_generators_strategy, t_params

## Solution (фрагмент кода)

```python
def _generate_and_run(variant: SignalVariant,
                      params: AntennaTestParams | None = None):
    """Генерировать сигнал, запустить NumPy pipeline, вернуть dict."""
    import math
    if params is None:
        params = AntennaTestParams.small(variant)

    strategy = SignalStrategyFactory.create(variant)
    S        = strategy.generate(params)

    # W = identity / sqrt(n_ant)
    W = (np.eye(params.n_ant, dtype=np.complex64) / np.sqrt(params.n_ant))
    X = (W @ S).astype(np.complex64)

    # Hamming + FFT
    n = params.n_samples
    nfft = 2 ** math.ceil(math.log2(n))
    win  = np.hamming(n).astype(np.float32)
    Xw   = (X * win[np.newaxis, :]).astype(np.complex64)
    Xp   = np.zeros((params.n_ant, nfft), dtype=np.complex64)
    Xp[:, :n] = Xw
    spec = np.fft.fft(Xp, axis=1).astype(np.complex64)
    mags = np.abs(spec).astype(np.float32)

    return dict(S=S, W=W, X=X, spectrum=spec, magnitudes=mags,
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_debug_steps.py`
- **Строк кода**: 183
- **Top-функций**: 7
- **Test runner**: common.runner

<!-- /rag-block -->
