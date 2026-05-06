---
id: dsp__linalg_capon__python_test_usecase__v1
type: python_test_usecase
source_path: DSP/Python/linalg/t_capon.py
primary_repo: linalg
module: linalg
uses_repos: []
uses_external: ['numpy', 'subprocess']
has_test_runner: true
is_opencl: false
line_count: 680
title: Capon beamformer gpu vs scipy
tags: []
uses_pybind: []
top_functions:
  - make_steering_matrix
  - make_noise
  - add_interference
  - capon_relief_ref
  - capon_beamform_ref
synonyms_ru:
  - тест капон
  - сравнение gpu
  - валидация бемформер
  - тест mvdr
  - тест сигнала
inherits_block_id: null
block_refs: []
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__linalg_capon__python_test_usecase__v1 -->

# Python use-case: Capon beamformer gpu vs scipy

## Цель

Численная верификация gpu-реализации алгоритма capon против numpy/scipy

## Когда применять

Запускать после изменений в capon_processor или gpu-контексте

## Используемые pybind-классы

_pybind-символов не найдено_

## Внешние зависимости

numpy, subprocess

## Solution (фрагмент кода)

```python
def make_steering_matrix(n_channels: int, n_directions: int,
                         theta_min: float, theta_max: float) -> np.ndarray:
    """
    ULA steering matrix: U[p, m] = exp(j * 2π * p * 0.5 * sin(θ_m))

    Returns U of shape [n_channels, n_directions], complex64.
    Column-major order (Fortran-contiguous) to match C++ convention.

    Что такое управляющий вектор (steering vector)?
    ────────────────────────────────────────────────
    Равномерная линейная решётка (ULA): антенны стоят в ряд с шагом d = λ/2.
    Если сигнал приходит под углом θ, между соседними антеннами возникает
    разность хода d·sin(θ). При шаге d=λ/2 это даёт фазовый сдвиг:
        φ = 2π·(d/λ)·sin(θ) = 2π·0.5·sin(θ)
    Антенна p получает сигнал с суммарным фазовым сдвигом p·φ:
        U[p, m] = exp(j · 2π · p · 0.5 · sin(θ_m))
    Если реальный сигнал пришёл именно с θ_m, «скалярное произведение»
    u_m^H·y_p будет максимальным — векторы «совпадут по фазе».
    """
    # M направлений равномерно от theta_min до theta_max
    thetas = np.linspace(theta_min, theta_max, n_directions)

    # p = [0, 1, 2, ..., P-1] — номер антенны, форма [P, 1] для broadcasting
    p = np.arange(n_channels)[:, None]          # [P, 1]
```

## Connection (C++ ↔ Python)

_Связи будут проставлены TASK_RAG_05 (class-card агент) и pybind_extractor._

## Метаданные

- **Source**: `DSP/Python/linalg/t_capon.py`
- **Строк кода**: 680
- **Top-функций**: 5
- **Test runner**: common.runner

<!-- /rag-block -->
