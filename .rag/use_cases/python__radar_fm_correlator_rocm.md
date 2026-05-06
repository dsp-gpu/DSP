---
id: dsp__radar_fm_correlator_rocm__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/radar/t_fm_correlator_rocm.py
primary_repo: radar
module: radar
uses_repos: ['core', 'radar']
uses_external: ['numpy']
has_test_runner: true
is_opencl: false
line_count: 129
title: Тесты FM-коррелятора ROCm
tags: ['radar', 'gpu', 'python', 'signal_processing', 'correlator', 'fft', 'rocm']
uses_pybind:
  - dsp_core.ROCmGPUContext
  - dsp_radar.FMCorrelatorROCm
synonyms_ru:
  - тестирование fm-коррелятора
  - rocm коррелятор
  - gpu коррелятор тесты
  - сравнение numpy и gpu
  - проверка параметров коррелятора
synonyms_en:
  - fm correlator testing
  - rocm correlator tests
  - gpu correlator verification
  - numpy vs gpu comparison
  - correlator parameter check
inherits_block_id: radar__fm_correlator_rocm__class_overview__v1
block_refs:
  - radar__fm_correlator_rocm__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__radar_fm_correlator_rocm__python_test_usecase__v1 -->

# Python use-case: Тесты FM-коррелятора ROCm

## Цель

Проверка корректности GPU-реализации FM-коррелятора против numpy и проверка функциональности.

## Когда применять

Запускать после настройки ROCm и GPU-контекста для тестирования FM-коррелятора.

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_core.ROCmGPUContext` | core |
| `dsp_radar.FMCorrelatorROCm` | radar |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
    import dsp_core as core
    import dsp_radar as radar
    HAS_ROCM = hasattr(radar, 'FMCorrelatorROCm')
except ImportError:
    HAS_ROCM = False
    core = None   # type: ignore
    radar = None  # type: ignore


class TestFMCorrelatorROCm:
    """FM Correlator ROCm tests — skip if ROCm not available."""

    def setUp(self):
        if not HAS_ROCM:
            raise SkipTest("ROCm not available or FMCorrelatorROCm not found")
        self._ctx = core.ROCmGPUContext(0)

    def test_autocorrelation(self):
        """Autocorrelation: ref vs ref -> peak at j=0, SNR > 10."""
        corr = radar.FMCorrelatorROCm(self._ctx)
        corr.set_params(fft_size=4096, num_shifts=1, num_signals=1,
                        num_output_points=200)
        ref = corr.generate_msequence()
        corr.prepare_reference_from_data(ref)
        peaks = corr.process(ref)
```

## Connection (C++ ↔ Python)

- C++ class-card: `radar__fm_correlator_rocm__class_overview__v1`

## Метаданные

- **Source**: `DSP/Python/radar/t_fm_correlator_rocm.py`
- **Строк кода**: 129
- **Top-функций**: 0
- **Test runner**: common.runner

<!-- /rag-block -->
