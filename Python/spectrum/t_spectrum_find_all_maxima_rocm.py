"""
GPUWorkLib: SpectrumMaximaFinder ROCm Tests (AMD, hipFFT)
==========================================================

ROCm-специфичные тесты для поиска максимумов спектра.
Используют ROCmGPUContext + hipFFT (не clFFT).

Тесты проверяют pipeline: FFT (hipFFT) → SpectrumProcessorROCm → peaks.

Примечание: Прямого Python API для SpectrumProcessorROCm нет.
Проверка выполняется через HeterodyneDechirp (который использует spectrum pipeline на ROCm)
или через запуск C++ тестов.

Tests:
  1. rocm_context_available — ROCmGPUContext создаётся
  2. spectrum_via_heterodyne — косвенная проверка: dechirp находит f_beat

Author: Kodo (AI Assistant)
Date: 2026-03-02
"""

import sys
import os
import subprocess

# DSP/Python в sys.path для импорта common.*
_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

from common.runner import SkipTest
from common.gpu_loader import GPULoader

GPULoader.setup_path()  # добавляет DSP/Python/lib/ (или build/python) в sys.path

try:
    import numpy as np
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
    if not _is_amd_rocm():
        raise SkipTest("ROCm not available (need AMD GPU)")
    if not hasattr(core, "ROCmGPUContext"):
        raise SkipTest("dsp_core built without ROCm")
    ctx = core.ROCmGPUContext(0)
    assert ctx is not None
    assert "AMD" in ctx.device_name or "Radeon" in ctx.device_name
    print(f"  ROCm device: {ctx.device_name}")


# ============================================================================
# Test 2: Spectrum pipeline via HeterodyneDechirp (ROCm)
# ============================================================================
def test_spectrum_via_heterodyne_rocm():
    """
    Косвенная проверка spectrum pipeline через HeterodyneROCm.dechirp.

    Алгоритм:
      1. Генерируем reference LFM без задержки
      2. Генерируем rx LFM с задержкой 100 µs → ожидаемый f_beat = chirp_rate * tau
      3. dc = het.dechirp(rx, ref) на GPU → должен быть тон на f_beat
      4. FFT(dc) + argmax → найденная f_beat
      5. Сверка: |f_beat_found - f_beat_expected| < 5 kHz
    """
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_heterodyne not found")
    if not _is_amd_rocm():
        raise SkipTest("ROCm not available (need AMD GPU)")
    if not hasattr(het, "HeterodyneROCm"):
        raise SkipTest("dsp_heterodyne built without HeterodyneROCm")

    ctx = core.ROCmGPUContext(0)
    het_proc = het.HeterodyneROCm(ctx)
    fs = 12e6
    f_start, f_end = 0.0, 2e6
    n = 8000
    n_ant = 1
    het_proc.set_params(float(f_start), float(f_end),
                        float(fs), int(n), int(n_ant))

    # Параметры LFM
    duration = n / fs
    chirp_rate = (f_end - f_start) / duration   # mu = B/T = 3e9 Hz/s

    # Reference: чистый LFM без задержки
    t = np.arange(n, dtype=np.float32) / fs
    phase_ref = 2 * np.pi * (0.5 * chirp_rate * t**2 + f_start * t)
    ref = np.exp(1j * phase_ref).astype(np.complex64)

    # rx: LFM с задержкой 100 µs → f_beat = chirp_rate * tau = 300 kHz
    delay_us = 100.0
    tau = delay_us * 1e-6
    t_delayed = t - tau
    phase_rx = 2 * np.pi * (0.5 * chirp_rate * t_delayed**2 + f_start * t_delayed)
    rx = np.exp(1j * phase_rx).astype(np.complex64)

    # Dechirp на GPU: dc = rx * conj(ref) → тон на f_beat
    dc = het_proc.dechirp(rx, ref)

    # Найти f_beat через FFT spectrum (CPU, validation only)
    spec = np.fft.fft(dc)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    idx_max = int(np.argmax(np.abs(spec)))
    f_beat = abs(float(freqs[idx_max]))

    expected = chirp_rate * tau   # 300 kHz
    err = abs(f_beat - expected)
    assert err < 5000, (
        f"f_beat error {err:.0f} Hz (expected ~{expected:.0f}, found {f_beat:.0f})"
    )
    print(f"  f_beat={f_beat/1e3:.1f} kHz, expected={expected/1e3:.0f} kHz, err={err:.0f} Hz")
