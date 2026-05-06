---
id: dsp__spectrum_spectrum_find_all_maxima_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/spectrum/t_spectrum_find_all_maxima_rocm.py
primary_repo: heterodyne
module: spectrum
uses_repos: ['core', 'heterodyne']
uses_external: ['numpy', 'subprocess']
has_test_runner: true
is_opencl: false
line_count: 134
title: Тесты поиска максимумов спектра ROCm
tags: ['rocm', 'gpu', 'python', 'signal_processing', 'fft', 'heterodyne', 'spectrum']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_heterodyne.HeterodyneROCm
top_functions:
  - _is_amd_rocm
  - test_rocm_context_available
  - test_spectrum_via_heterodyne_rocm
synonyms_ru:
  - поиск пиков спектра
  - анализ спектра ROCm
  - тесты обработки сигналов
  - проверка hipFFT
  - тесты SpectrumProcessorROCm
synonyms_en:
  - spectrum peak detection
  - ROCm spectrum analysis
  - signal processing tests
  - hipFFT verification
  - SpectrumProcessorROCm tests
inherits_block_id: heterodyne__heterodyne_rocm__class_overview__v1
block_refs:
  - heterodyne__heterodyne_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__spectrum_spectrum_find_all_maxima_rocm__python_test_usecase__v1 -->

# Python use-case: Тесты поиска максимумов спектра ROCm

## Цель

Проверка корректности обработки спектра через ROCm и hipFFT без прямого Python API

## Когда применять

Запускать после изменений в ROCmGPUContext или GPU-контексте

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_heterodyne.HeterodyneROCm` | heterodyne |

## Внешние зависимости

numpy, subprocess

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_heterodyne as het
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None  # type: ignore
    het = None   # type: ignore


def _is_amd_rocm():
    """Проверить, что ROCm доступен (AMD GPU)."""
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=3)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ============================================================================
# Test 1: ROCm context available
# ============================================================================
def test_rocm_context_available():
    """ROCmGPUContext создаётся на AMD."""
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_heterodyne not found")
```

## Connection (C++ ↔ Python)

- C++ class-card: `heterodyne__heterodyne_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/spectrum/t_spectrum_find_all_maxima_rocm.py`
- **Строк кода**: 134
- **Top-функций**: 3
- **Test runner**: common.runner

<!-- /rag-block -->
