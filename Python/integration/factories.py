"""
factories.py — фабрики для DSP/Python/integration/
=============================================================

Предоставляет factory functions для интеграционных тестов.

После миграции с legacy GPUWorkLib на DSP-GPU:
- `SignalGenerator` (OpenCL) → NumPy fallback (`_NumpySignalGenerator`)
- `FFTProcessor` (alias на FFTProcessorROCm) → `dsp_spectrum.FFTProcessorROCm`
- `ScriptGenerator` → SkipTest (нет в DSP-GPU, см. .future/TASK_script_dsl_rocm.md)
"""

import os
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
integration_plot_dir: str = os.path.join(_PROJECT_ROOT, "Results", "Plots", "integration")
os.makedirs(integration_plot_dir, exist_ok=True)


class _NumpySignalGenerator:
    """NumPy-обёртка с API совместимым со старым gw.SignalGenerator.

    Нужно после миграции с GPUWorkLib: dsp_signal_generators не содержит
    общего CW/LFM/Noise-генератора, но эти сигналы тривиально считаются
    через NumPy без GPU.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def generate_cw(self, freq, fs, length, amplitude=1.0):
        t = np.arange(length) / fs
        return (amplitude * np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)

    def generate_lfm(self, f_start, f_end, fs, length, amplitude=1.0):
        t = np.arange(length) / fs
        T = length / fs
        mu = (f_end - f_start) / T
        phase = 2 * np.pi * (f_start * t + 0.5 * mu * t**2)
        return (amplitude * np.exp(1j * phase)).astype(np.complex64)

    def generate_noise(self, fs, length, power=1.0, noise_type="gaussian"):
        # power → amplitude=sqrt(power)
        amp = np.sqrt(power)
        re = self._rng.standard_normal(length)
        im = self._rng.standard_normal(length)
        return (amp * (re + 1j * im) / np.sqrt(2.0)).astype(np.complex64)


def make_sig_gen(_gw=None, _ctx=None):
    """Создаёт NumPy-based SignalGenerator (после миграции с GPUWorkLib).

    Args для обратной совместимости (игнорируются): _gw, _ctx.
    Раньше создавалось через gw.SignalGenerator(ctx) — теперь чистый NumPy.
    """
    return _NumpySignalGenerator()


def make_fft_proc(_gw=None, ctx=None):
    """Создаёт dsp_spectrum.FFTProcessorROCm (требует ROCmGPUContext).

    Args:
        _gw: игнорируется (для обратной совместимости со старым API)
        ctx: ROCmGPUContext (создаётся вызывающим)
    """
    import dsp_spectrum as spectrum
    return spectrum.FFTProcessorROCm(ctx)


def make_script_gen(_gw=None, _ctx=None):
    """ScriptGenerator не портирован в DSP-GPU — SkipTest.

    См. перспективную задачу `MemoryBank/.future/TASK_script_dsl_rocm.md`.
    """
    from common.runner import SkipTest
    raise SkipTest(
        "ScriptGenerator (runtime DSL → kernel compiler) не портирован в DSP-GPU; "
        "см. MemoryBank/.future/TASK_script_dsl_rocm.md"
    )
