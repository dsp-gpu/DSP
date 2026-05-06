---
id: dsp__integration_zero_copy__cross_repo_pipeline__v1
type: cross_repo_pipeline
source_path: DSP/Python/integration/t_zero_copy.py
primary_repo: integration
module: integration
uses_repos: ['core']
uses_external: []
has_test_runner: false
is_opencl: false
line_count: 171
title: ZeroCopy мост OpenCL ROCm
tags: ['integration', 'rocm', 'gpu', 'python', 'signal_processing', 'cross_repo', 'zero_copy', 'bridge', 'opencl']
uses_pybind:
  - dsp_core.HybridGPUContext
top_functions:
  - run_test
  - test_hybrid_context_creates
  - test_device_names_not_empty
  - test_zero_copy_method_detection
  - test_zero_copy_supported_is_bool
  - test_hybrid_repr
  - test_context_manager
  - test_zero_copy_report
synonyms_ru:
  - zero copy проверка
  - zero copy тест
  - opencl rocm мост
  - zero copy доступность
  - zero copy метод
synonyms_en:
  - zero copy python_test
  - zero copy bridge
  - opencl rocm bridge
  - zero copy availability
  - zero copy detection
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__integration_zero_copy__cross_repo_pipeline__v1 -->

# Python use-case: ZeroCopy мост OpenCL ROCm

## Цель

Проверка доступности ZeroCopy и корректности определения метода

## Когда применять

Запускать после изменений в ZeroCopy мосте или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.HybridGPUContext` | core |

## Внешние зависимости

_нет_

## Solution (фрагмент кода)

```python
import dsp_core as core

PASSED = 0
FAILED = 0


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [ZeroCopy] {name}: PASSED")
        PASSED += 1
    except AssertionError as e:
        print(f"  [ZeroCopy] {name}: FAILED — AssertionError: {e}")
        FAILED += 1
    except Exception as e:
        print(f"  [ZeroCopy] {name}: FAILED — {type(e).__name__}: {e}")
        FAILED += 1


# ============================================================================
# Тест 1: HybridGPUContext создаётся успешно
# ============================================================================

def test_hybrid_context_creates():
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/integration/t_zero_copy.py`
- **Строк кода**: 171
- **Top-функций**: 8
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
