---
id: dsp__integration_hybrid_backend__cross_repo_pipeline__v1
type: cross_repo_pipeline
source_path: DSP/Python/integration/t_hybrid_backend.py
primary_repo: integration
module: integration
uses_repos: ['core', 'stats', 'spectrum']
uses_external: ['numpy']
has_test_runner: false
is_opencl: false
line_count: 236
title: Hybrid GPU Backends Тест
tags: ['integration', 'rocm', 'gpu', 'python', 'cross_repo', 'hybrid', 'opencl']
uses_pybind:
  - dsp_core.HybridGPUContext
  - dsp_core.ROCmGPUContext
  - dsp_stats.StatisticsProcessor
  - dsp_core.GPUContext
  - dsp_spectrum.FFTProcessorROCm
top_functions:
  - run_test
  - test_hybrid_init
  - test_hybrid_device_names
  - test_hybrid_rocm_statistics
  - test_hybrid_zero_copy_info
  - test_hybrid_repr
  - test_hybrid_context_manager
  - test_hybrid_parallel_opencl_rocm
  - test_hybrid_status_report
synonyms_ru:
  - тестирование гибридных gpu
  - тесты подсистем
  - тесты opencl rocm
  - тесты контекста gpu
  - тесты межподсистемной работы
synonyms_en:
  - hybrid gpu testing
  - cross-backend tests
  - opencl rocm integration
  - gpu context validation
  - multi-backend verification
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__integration_hybrid_backend__cross_repo_pipeline__v1 -->

# Python use-case: Hybrid GPU Backends Тест

## Цель

Проверка инициализации HybridGPUContext и корректной работы OpenCL/ROCm подсистем на одном GPU

## Когда применять

Запускать после изменений в HybridGPUContext или подсистемах OpenCL/ROCm

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.HybridGPUContext` | core |
| `dsp_core.ROCmGPUContext` | core |
| `dsp_stats.StatisticsProcessor` | stats |
| `dsp_core.GPUContext` | core |
| `dsp_spectrum.FFTProcessorROCm` | spectrum |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
import dsp_core as core
import dsp_spectrum as spectrum
import dsp_stats as stats

PASSED = 0
FAILED = 0


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [Hybrid] {name}: PASSED")
        PASSED += 1
    except AssertionError as e:
        print(f"  [Hybrid] {name}: FAILED — AssertionError: {e}")
        FAILED += 1
    except Exception as e:
        print(f"  [Hybrid] {name}: FAILED — {type(e).__name__}: {e}")
        FAILED += 1


# ============================================================================
# Тест 1: HybridGPUContext создаётся, оба backend инициализированы
# ============================================================================
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/integration/t_hybrid_backend.py`
- **Строк кода**: 236
- **Top-функций**: 9
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
