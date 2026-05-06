---
id: dsp__common_smoke__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/common/validators/t_smoke.py
primary_repo: common
module: common
uses_repos: []
uses_external: ['__future__', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 287
title: Smoke-тесты валидаторов
tags: ['common', 'data_validation', 'unit_test', 'validator', 'attribute_check', 'python', 'test_runner']
uses_pybind: []
synonyms_ru:
  - тесты валидаторов
  - проверка атрибутов
  - валидация данных
  - тестирование класса
  - проверка свойств
synonyms_en:
  - validator tests
  - attribute check
  - data validation
  - class testing
  - property verification
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__common_smoke__python_test_usecase__v1 -->

# Python use-case: Smoke-тесты валидаторов

## Цель

Проверяет публичные атрибуты DataValidator на соответствие ожидаемым значениям

## Когда применять

Запускать при проверке корректности атрибутов DataValidator или при изменении класса DataValidator

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

__future__, numpy

## Solution (фрагмент кода)

```python
class TestValidatorsSmoke:
    """Smoke-тесты валидаторов — работают без GPU."""

    # ── Backward compatibility ──────────────────────────────────────────────

    def test_backward_compat_public_attrs(self) -> TestResult:
        """DataValidator должен иметь публичные .tolerance/.metric/.METRICS."""
        from common import DataValidator

        tr = TestResult(test_name="backward_compat_public_attrs")
        v = DataValidator(tolerance=0.01, metric="max_rel")

        # Атрибуты, которые могут использовать внешние тесты
        ok = (
            v.tolerance == 0.01
            and v.metric == "max_rel"
            and DataValidator.METRICS == ("max_rel", "abs", "rmse")
        )
        tr.add(ValidationResult(
            passed=ok,
            metric_name="public_attrs",
            actual_value=1.0 if ok else 0.0,
            threshold=1.0,
            message=f"tolerance={v.tolerance}, metric={v.metric}",
        ))
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/common/validators/t_smoke.py`
- **Строк кода**: 287
- **Top-функций**: 0
- **Test runner**: common.runner

<!-- /rag-block -->
