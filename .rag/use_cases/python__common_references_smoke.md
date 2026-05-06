---
id: dsp__common_references_smoke__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/common/references/t_references_smoke.py
primary_repo: common
module: common
uses_repos: []
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 217
title: Проверка базовых свойств references без GPU
tags: ['signal_processing', 'python', 'references', 'core', 'numpy']
uses_pybind: []
synonyms_ru:
  - тест_базовый
  - проверка_сигналов
  - тест_references
  - тест_без_gpu
  - базовая_проверка
synonyms_en:
  - basic_test
  - signal_check
  - references_test
  - no_gpu_test
  - core_check
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__common_references_smoke__python_test_usecase__v1 -->

# Python use-case: Проверка базовых свойств references без GPU

## Цель

Проверяет тип данных и форму массивов для сигналов CW и LFM с использованием NumPy.

## Когда применять

Запускать после изменений в модуле references или при проверке базовой функциональности.

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
class TestReferencesSmoke:
    """Smoke-тест references -- все проверки без GPU."""

    def test_signal_refs(self):
        from common.references import SignalReferences
        result = TestResult(test_name="signal_refs")

        # CW
        cw = SignalReferences.cw(12e6, 4096, 2e6)
        result.add(ValidationResult(
            passed=cw.dtype == np.complex64 and cw.shape == (4096,),
            metric_name="cw_shape_dtype",
            actual_value=1.0 if cw.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={cw.dtype}, shape={cw.shape}"
        ))

        # LFM
        lfm = SignalReferences.lfm(12e6, 4096, 0.0, 2e6)
        result.add(ValidationResult(
            passed=lfm.dtype == np.complex64 and lfm.shape == (4096,),
            metric_name="lfm_shape_dtype",
            actual_value=1.0 if lfm.dtype == np.complex64 else 0.0,
            threshold=1.0,
            message=f"dtype={lfm.dtype}, shape={lfm.shape}"
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/common/references/t_references_smoke.py`
- **Строк кода**: 217
- **Top-функций**: 0
- **Test runner**: common.runner

<!-- /rag-block -->
