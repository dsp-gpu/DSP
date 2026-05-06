---
id: dsp__stats_snr_estimator__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/stats/t_snr_estimator.py
primary_repo: stats
module: stats
uses_repos: []
uses_external: ['cfar_estimator', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 225
title: Тест оценщика SNR
tags: ['stats', 'gpu', 'python', 'signal_processing', 'snr', 'cross_repo', 'rocm']
uses_pybind: []
top_functions:
  - _get_gpu_module_or_skip
  - _import_cfar_or_skip
  - main
synonyms_ru:
  - оценка SNR
  - тестирование оценщика
  - тест SNR
  - проверка SNR
  - оценка уровня сигнала
synonyms_en:
  - SNR estimation python_test
  - SNR estimator verification
  - signal-to-noise ratio python_test
  - SNR evaluation
  - noise signal ratio check
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__stats_snr_estimator__python_test_usecase__v1 -->

# Python use-case: Тест оценщика SNR

## Цель

Проверка полного pipeline оценщика SNR на GPU ROCm против numpy-референса.

## Когда применять

Запускать после изменений в оценщике SNR или GPU-модулях.

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

cfar_estimator, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core  # noqa: E402
    import dsp_stats as stats  # noqa: E402
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None   # type: ignore
    stats = None  # type: ignore


def _get_gpu_module_or_skip():
    """Загрузить dsp_stats или SkipTest. Возвращает (core, stats) tuple."""
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_stats not built — run CMake build first")
    if not hasattr(stats, "StatisticsProcessor"):
        raise SkipTest("StatisticsProcessor not in bindings")
    if not hasattr(stats, "SnrEstimationConfig"):
        raise SkipTest("SNR bindings not built (SNR_07) — rebuild needed")
    return core, stats


def _import_cfar_or_skip():
    """Импорт numpy reference — пропуск если недоступен."""
    try:
        import cfar_estimator  # type: ignore
        return cfar_estimator
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/stats/t_snr_estimator.py`
- **Строк кода**: 225
- **Top-функций**: 3
- **Test runner**: common.runner

<!-- /rag-block -->
