---
id: dsp__integration_fft_integration__cross_repo_pipeline__v1
type: cross_repo_pipeline
source_path: DSP/Python/integration/t_fft_integration.py
primary_repo: integration
module: integration
uses_repos: ['core']
uses_external: ['integration.factories', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 183
title: Тесты fft и signal_generator
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
synonyms_ru:
  - тесты fft
  - интеграция fft
  - тестирование fft
  - fft проверка
  - fft интеграция
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__integration_fft_integration__cross_repo_pipeline__v1 -->

# Python use-case: Тесты fft и signal_generator

## Цель

Проверка корректности интеграции fft-процессора с генератором сигналов

## Когда применять

Запускать после изменений в fft-модуле или signal_generator

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


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: CW → FFT → peak at expected frequency
# ─────────────────────────────────────────────────────────────────────────────

class TestCwFftIntegration:
    """CW + FFT: пик спектра на заданной частоте."""

    def setUp(self):
        if not HAS_GPU:
            raise SkipTest("dsp_core не найден")
        ctx = core.ROCmGPUContext(0)
        self._sig_gen = make_sig_gen()  # NumPy-based (после миграции с GPUWorkLib)
        self._fft_proc = make_fft_proc(ctx=ctx)

    def test_cw_peak_frequency(self):
        """FFT пик CW-сигнала должен быть на частоте f0 ± 1 бин для нескольких частот."""
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/integration/t_fft_integration.py`
- **Строк кода**: 183
- **Top-функций**: 0
- **Test runner**: common.runner

<!-- /rag-block -->
