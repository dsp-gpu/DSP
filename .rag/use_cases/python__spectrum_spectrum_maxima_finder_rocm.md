---
id: dsp__spectrum_spectrum_maxima_finder_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_spectrum_maxima_finder_rocm.py
primary_repo: spectrum
module: spectrum
uses_repos: ['core', 'spectrum']
uses_external: ['numpy', 'scipy.signal']
has_test_runner: true
is_opencl: false
line_count: 269
title: Тесты поиска максимумов спектра на GPU
tags: []
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_spectrum.SpectrumMaximaFinderROCm
top_functions:
  - next_pow2
  - make_single_tone
  - make_two_tone
  - numpy_find_all_maxima
synonyms_ru:
  - тесты поиска максимумов спектра
  - тесты ROCm спектрального анализа
  - тесты обнаружения пиков
  - тесты GPU-спектра
  - тесты анализа сигналов
inherits_block_id: spectrum__spectrum_maxima_finder_rocm__class_overview__v1
block_refs:
  - spectrum__spectrum_maxima_finder_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_spectrum_maxima_finder_rocm__python_test_usecase__v1 -->

# Python use-case: Тесты поиска максимумов спектра на GPU

## Цель

Проверка корректности GPU-реализации поиска максимумов спектра с NumPy/SciPy

## Когда применять

Запускать после изменений в SpectrumMaximaFinderROCm или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_spectrum.SpectrumMaximaFinderROCm` | spectrum |

## Внешние зависимости

numpy, scipy.signal

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_ROCM_MAXIMA = hasattr(spectrum, 'SpectrumMaximaFinderROCm')
except ImportError:
    HAS_ROCM_MAXIMA = False
    core = None      # type: ignore
    spectrum = None  # type: ignore



# ─── NumPy helpers ────────────────────────────────────────────────────────────

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def make_single_tone(n: int, k: int) -> np.ndarray:
    """Комплексный экспоненциальный сигнал на частоте k/N."""
    t = np.arange(n)
    return np.exp(1j * 2 * np.pi * k / n * t).astype(np.complex64)
```

## Connection (C++ ↔ Python)

- C++ class-card: `spectrum__spectrum_maxima_finder_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_spectrum_maxima_finder_rocm.py`
- **Строк кода**: 269
- **Top-функций**: 4
- **Test runner**: common.runner

<!-- /rag-block -->
