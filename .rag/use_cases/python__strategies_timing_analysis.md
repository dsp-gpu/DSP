---
id: dsp__strategies_timing_analysis__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/strategies/t_timing_analysis.py
primary_repo: strategies
module: strategies
uses_repos: []
uses_external: ['glob', 'json', 'matplotlib', 'matplotlib.pyplot', 'numpy']
has_test_runner: true
is_opencl: false
line_count: 221
title: анализ времени выполнения шагов из json
tags: ['signal_processing', 'gpu', 'python', 'timing_analysis', 'json_validation', 'rocm', 'strategies']
uses_pybind: []
top_functions:
  - _load_timing_json
  - _find_timing_files
  - _print_timing_table
  - _plot_timing_bars
  - test_timing_files_exist
  - test_timing_json_valid
  - test_timing_sanity
  - test_plot_timing_bars
  - parse_and_report
synonyms_ru:
  - анализ времени выполнения
  - проверка json
  - график шагов
  - таблица времени
  - тестирование производительности
synonyms_en:
  - timing analysis
  - json validation
  - step chart
  - timing table
  - performance testing
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__strategies_timing_analysis__python_test_usecase__v1 -->

# Python use-case: анализ времени выполнения шагов из json

## Цель

Проверяет наличие и корректность JSON-файлов с временем выполнения шагов, строит таблицу и график

## Когда применять

Запускать после выполнения C++ тестов TimingPerStepTest, создающих timing_*.json

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

glob, json, matplotlib, matplotlib.pyplot, numpy

## Solution (фрагмент кода)

```python
def _load_timing_json(path: str) -> dict | None:
    """Загрузить JSON файл timing из C++ TimingPerStepTest."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _find_timing_files() -> list[str]:
    """Найти все timing_*.json в Results/strategies/."""
    pattern = os.path.join(_RESULTS_DIR, "timing_*.json")
    return sorted(glob.glob(pattern))


def _print_timing_table(data: dict) -> None:
    """Вывести таблицу timing в консоль."""
    print(f"\n{'─'*52}")
    print(f"  Timing: signal={data.get('signal','?')}  "
          f"n_ant={data.get('n_ant','?')}  n_samples={data.get('n_samples','?')}")
    print(f"{'─'*52}")
    print(f"  {'Step':<20} {'GPU ms':>8}  {'Wall ms':>8}")
    print(f"{'─'*52}")
    steps = data.get("steps", [])
    for s in steps:
        print(f"  {s['name']:<20} {s['gpu_ms']:>8.3f}  {s['wall_ms']:>8.3f}")
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/strategies/t_timing_analysis.py`
- **Строк кода**: 221
- **Top-функций**: 9
- **Test runner**: common.runner

<!-- /rag-block -->
