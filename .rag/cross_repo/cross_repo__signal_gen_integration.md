---
id: dsp__integration_signal_gen_integration__cross_repo_pipeline__v1
type: cross_repo_pipeline
source_path: DSP/Python/integration/t_signal_gen_integration.py
primary_repo: integration
module: integration
uses_repos: ['core']
uses_external: ['integration.factories', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 224
title: Интеграционные тесты генераторов сигналов
tags: ['integration', 'signal_processing', 'gpu', 'python', 'fft', 'cross_repo', 'pipeline']
uses_pybind:
  - dsp_core.ROCmGPUContext
top_functions:
  - _make_ctx_and_gen
synonyms_ru:
  - тесты интеграции
  - проверка геннераторов
  - pipeline сигналов
  - тесты rocm
  - тесты gpu
synonyms_en:
  - integration tests
  - signal generator verification
  - signal pipeline testing
  - rocm tests
  - gpu tests
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__integration_signal_gen_integration__cross_repo_pipeline__v1 -->

# Python use-case: Интеграционные тесты генераторов сигналов

## Цель

Проверка корректности работы генераторов сигналов и полного pipeline включая FFT-анализ

## Когда применять

Запускать после изменений в модуле integration или связанных репозиториях

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |

## Внешние зависимости

integration.factories, numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None  # type: ignore

from integration.factories import make_sig_gen, make_fft_proc


def _make_ctx_and_gen():
    if not HAS_GPU:
        raise SkipTest("dsp_core не найден")
    return None, core.ROCmGPUContext(0)  # legacy: tuple (gw, ctx); gw больше не нужен


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Многоканальная генерация с разными частотами
# ─────────────────────────────────────────────────────────────────────────────

class TestMultichannelGeneration:
    """Несколько каналов с разными параметрами."""

    def setUp(self):
        gw, ctx = _make_ctx_and_gen()
        self._sig_gen = make_sig_gen(gw, ctx)
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/integration/t_signal_gen_integration.py`
- **Строк кода**: 224
- **Top-функций**: 1
- **Test runner**: common.runner

<!-- /rag-block -->
