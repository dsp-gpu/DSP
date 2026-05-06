---
id: dsp__spectrum_ai_pipeline__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/ai_pipeline/t_ai_pipeline.py
primary_repo: core
module: spectrum
uses_repos: ['core']
uses_external: ['filter_designer', 'llm_parser', 'numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 274
title: Тест ai-пайплайна gpu
tags: ['core', 'spectrum', 'gpu', 'python', 'signal_processing', 'filter', 'fft', 'rocm']
uses_pybind:
  - dsp_core.ROCmGPUContext
top_functions:
  - make_noise_signal
synonyms_ru:
  - тест пайплайна
  - gpu фильтрация
  - ai проверка
  - mock парсер
  - pipeline валидация
synonyms_en:
  - ai pipeline python_test
  - gpu filtering validation
  - mock parser testing
  - filter pipeline verification
  - cross-repo integration
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_ai_pipeline__python_test_usecase__v1 -->

# Python use-case: Тест ai-пайплайна gpu

## Цель

Проверка корректности работы AI-пайплайна с GPU-фильтрацией без зависимостей от AI-бэкендов

## Когда применять

Запускать после изменений в GPU-контексте или модуле FilterDesigner

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |

## Внешние зависимости

filter_designer, llm_parser, numpy, scipy.signal

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore

# Phase B 2026-05-04: when run as script, ai_pipeline/ дир добавляется автоматически
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_parser import MockParser, FilterSpec, create_parser
from filter_designer import FilterDesigner, FilterDesign


def make_noise_signal(n: int = 4096, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# MockParser тесты
# ─────────────────────────────────────────────────────────────────────────────

class TestMockParser:
    """Тесты детерминированного regex-парсера."""
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/spectrum/ai_pipeline/t_ai_pipeline.py`
- **Строк кода**: 274
- **Top-функций**: 1
- **Test runner**: common.runner

<!-- /rag-block -->
