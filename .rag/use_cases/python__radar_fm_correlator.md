---
id: dsp__radar_fm_correlator__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/radar/t_fm_correlator.py
primary_repo: radar
module: radar
uses_repos: ['core', 'radar']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 268
title: тест fm коррелятора
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'fft', 'radar']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_radar.FMCorrelatorROCm
top_functions:
  - generate_msequence_cpu
  - correlate_numpy
synonyms_ru:
  - тест fm коррелятора
  - корреляция сигналов
  - m_последовательность
  - автокорреляция
  - сравнение gpu и numpy
synonyms_en:
  - fm correlator python_test
  - signal correlation
  - m-sequence
  - auto correlation
  - numpy vs gpu
inherits_block_id: radar__fm_correlator_rocm__class_overview__v1
block_refs:
  - radar__fm_correlator_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__radar_fm_correlator__python_test_usecase__v1 -->

# Python use-case: тест fm коррелятора

## Цель

Проверка корректности вычислений FM-коррелятора с использованием NumPy-эталона

## Когда применять

Запускать после изменений в FM-корреляторе или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_radar.FMCorrelatorROCm` | radar |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
def generate_msequence_cpu(n: int, seed: int = 0x12345678,
                           poly: int = 0x00400007) -> np.ndarray:
    """Генерация M-последовательности LFSR на CPU (эталон для C++/GPU).

    Использует LFSR с полиномом poly. Выход: {+1, -1}.
    Совместимо с FMCorrelator::GenerateMSequence().
    """
    seq = np.zeros(n, dtype=np.float32)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        bit = (state >> 31) & 1
        seq[i] = 1.0 if bit else -1.0
        # feedback: XOR bits согласно полиному
        feedback = bin(state & poly).count('1') & 1
        state = ((state << 1) | feedback) & 0xFFFFFFFF
    return seq


def correlate_numpy(ref: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """Кросс-корреляция через FFT (эталон).

    Returns: corr[j] = sum(ref[k] * inp[(k+j) % N]) — циклическая.
    Нормировка: / N (как в FMCorrelator::RunCorrelationPipeline).
    """
    n = len(ref)
```

## Connection (C++ ↔ Python)

- C++ class-card: `radar__fm_correlator_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/radar/t_fm_correlator.py`
- **Строк кода**: 268
- **Top-функций**: 2
- **Test runner**: common.runner

<!-- /rag-block -->
