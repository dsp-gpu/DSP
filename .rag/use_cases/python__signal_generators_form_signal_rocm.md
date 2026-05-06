---
id: dsp__signal_generators_form_signal_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/signal_generators/t_form_signal_rocm.py
primary_repo: signal_generators
module: signal_generators
uses_repos: ['core', 'signal_generators']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 284
title: Тестирование генератора сигналов ROCm
tags: ['signal_generators', 'rocm', 'gpu', 'signal_processing', 'lfm', 'filter', 'python', 'fft']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_signal_generators.FormSignalGeneratorROCm
top_functions:
  - generate_cw_numpy
  - generate_lfm_numpy
  - peak_freq
synonyms_ru:
  - тестирование сигналов
  - генерация сигналов
  - проверка ROCm
  - формирование волн
  - тестирование волн
inherits_block_id: signal_generators__form_signal_generator_rocm__class_overview__v1
block_refs:
  - signal_generators__form_signal_generator_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__signal_generators_form_signal_rocm__python_test_usecase__v1 -->

# Python use-case: Тестирование генератора сигналов ROCm

## Цель

Проверка корректности формирования сигналов (CW, LFM, шум) на GPU через ROCm backend

## Когда применять

Запускать после изменений в FormSignalGeneratorROCm или ROCmGPUContext

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_signal_generators.FormSignalGeneratorROCm` | signal_generators |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_signal_generators as signal_generators
    HAS_FORM_ROCM = hasattr(signal_generators, 'FormSignalGeneratorROCm')
except ImportError:
    HAS_FORM_ROCM = False
    core = None              # type: ignore
    signal_generators = None  # type: ignore



# ─── NumPy helpers ────────────────────────────────────────────────────────────

NORM = 0.7071067811865476  # 1/sqrt(2)


def generate_cw_numpy(antennas: int, points: int, fs: float, f0: float,
                      amplitude: float = 1.0, norm: float = NORM,
                      tau_step: float = 0.0) -> np.ndarray:
    """NumPy-референс для CW сигнала с линейной задержкой."""
    t = np.arange(points) / fs
    result = np.zeros((antennas, points), dtype=np.complex64)
    for ant in range(antennas):
        tau = ant * tau_step
        t_shifted = t - tau
        valid = (t_shifted >= 0) & (t_shifted <= (points - 1) / fs)
```

## Connection (C++ ↔ Python)

- C++ class-card: `signal_generators__form_signal_generator_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/signal_generators/t_form_signal_rocm.py`
- **Строк кода**: 284
- **Top-функций**: 3
- **Test runner**: common.runner

<!-- /rag-block -->
