# 🤖 CLAUDE — `DSP` (мета-репо)

> Мета-репо: Python/, Doc/, Examples/, Results/, Logs/, общий runner тестов.
> Зависит от: **всех** 8 рабочих модулей. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

```
DSP/
├── Python/                  # Python тесты всех модулей + общий TestRunner
│   ├── common/              # runner.py, result.py, gpu_loader.py
│   ├── spectrum/            # test_*.py для spectrum
│   ├── stats/               # test_*.py
│   ├── ...                  # остальные модули
│   └── README.md
├── Doc/                     # сводная документация
│   ├── Python/              # {module}_api.md — Python API каждого модуля
│   ├── Doxygen/             # cross-repo Doxygen (HTML не в git)
│   └── addition/            # дополнительные гайды (Debian setup и т.п.)
├── Examples/                # C++ / Python примеры использования
├── Results/                 # JSON / Plots / Profiler результаты бенчмарков
├── Logs/DRVGPU_XX/          # plog файлы per-GPU
└── src/main.cpp             # точка входа: вызывает all_test.hpp каждого модуля
```

## ⚠️ Специфика

- **Главный `main.cpp`** — **НЕ вызывает тесты напрямую**. Только через `all_test.hpp` каждого модуля.
- **Python тесты** — **без pytest**. Прямой запуск `python3 DSP/Python/{module}/test_*.py`.
- **TestRunner** (`DSP/Python/common/runner.py`) — переносится во все будущие проекты Alex как стандарт.
- **GPULoader singleton** — один GPU-context на процесс, переиспользование в тестах.
- **configGPU.json** — per-GPU флаги: `is_console`, `is_prof`, `is_log`.

## 📂 Где что искать

| Путь | Что там |
|------|--------|
| `DSP/Python/common/runner.py` | `TestRunner`, `SkipTest`, `TestResult`, `ValidationResult` |
| `DSP/Python/common/gpu_loader.py` | `GPULoader` singleton |
| `DSP/Doc/Python/{module}_api.md` | Python API документация каждого модуля |
| `DSP/Examples/GPUProfiler_SetGPUInfo.md` | Пример подключения ProfilingFacade |
| `DSP/Examples/GetGPU_and_Mellanox/` | Детект GPU + Mellanox |
| `DSP/Results/Profiler/` | JSON + MD отчёты ProfilingFacade |
| `DSP/Results/Plots/{module}/` | Графики из Python тестов |
| `DSP/Logs/DRVGPU_XX/YYYY-MM-DD/` | Логи per-GPU, per-day |

## 🚫 Запреты

- **pytest** — навсегда (см. `.claude/rules/04-testing-python.md`).
- Не создавать новый `runner.py` в другом месте — только `DSP/Python/common/runner.py`.
- Не писать в `DSP/Results/` в коммит — результаты не в git (или в git-lfs).
- Не складывать отчёты бенчмарков в репо сырыми stdout-дампами — только через `ProfilingFacade::Export*`.

## 🔗 Правила (path-scoped автоматически)

- `04-testing-python.md` — TestRunner, NO pytest
- `11-python-bindings.md` — pybind11
- `13-optimization-docs.md` — где лежат гайды
- `14-cpp-style.md` + `15-cpp-testing.md`
