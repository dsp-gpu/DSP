---
id: dsp__radar_range_angle__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/radar/t_range_angle.py
primary_repo: core
module: radar
uses_repos: ['core']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 286
title: тест range_angle
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
top_functions:
  - make_lfm_signal
  - build_antenna_array
synonyms_ru:
  - проверка параметров range_angle
  - тестирование сигнала
  - валидация процессора
  - тестирование lfm
  - проверка методов
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__radar_range_angle__python_test_usecase__v1 -->

# Python use-case: тест range_angle

## Цель

Проверяет параметры, методы и обработку сигналов модуля RangeAngleProcessor

## Когда применять

Запускать после изменений в RangeAngleParams или GPU-процессоре

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_radar as radar
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None   # type: ignore
    radar = None  # type: ignore


# ── Вспомогательные функции ───────────────────────────────────────────────────

def make_lfm_signal(n_samples: int, fs: float, f_start: float, f_end: float) -> np.ndarray:
    """Генерация LFM сигнала через NumPy (эталонный расчёт)."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    B = f_end - f_start
    T = n_samples / fs
    mu = B / T
    return np.exp(1j * np.pi * mu * t ** 2).astype(np.complex64)


def build_antenna_array(rx_single: np.ndarray, n_ant: int) -> np.ndarray:
    """Размножить сигнал одной антенны на n_ant антенн (угол = 0°, фаза одинакова)."""
    return np.tile(rx_single, n_ant).astype(np.complex64)
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/radar/t_range_angle.py`
- **Строк кода**: 286
- **Top-функций**: 2
- **Test runner**: common.runner

<!-- /rag-block -->
