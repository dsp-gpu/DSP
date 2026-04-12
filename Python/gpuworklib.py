"""
gpuworklib.py — shim для обратной совместимости с GPUWorkLib.

Реэкспортирует классы из 8 модульных .pyd/.so файлов под старыми именами.
Позволяет старым тестам работать без изменений:
    gw = GPULoader.get()
    ctx = gw.GPUContext()      # → dsp_core.GPUContext
    fft = gw.FFTProcessor(ctx) # → dsp_spectrum.FFTProcessorROCm

Постепенно заменяйте `import gpuworklib` на конкретные импорты:
    import dsp_core as core
    import dsp_spectrum as spectrum
    ctx = core.ROCmGPUContext()
    fft = spectrum.FFTProcessorROCm(ctx)

@author Кодо (AI Assistant)
@date 2026-04-12 (Phase 3b)
"""

import sys as _sys

# ── dsp_core ──────────────────────────────────────────────────────
try:
    from dsp_core import (
        GPUContext,
        get_gpu_count,
        list_gpus,
    )
    try:
        from dsp_core import ROCmGPUContext, HybridGPUContext
    except ImportError:
        pass  # не ROCm-сборка
except ImportError as e:
    raise ImportError(
        f"dsp_core not found: {e}\n"
        "Запустите: cmake --install build --prefix DSP/Python/lib\n"
        "Затем: export PYTHONPATH=DSP/Python/lib:$PYTHONPATH"
    ) from e

# ── dsp_spectrum ──────────────────────────────────────────────────
try:
    from dsp_spectrum import (
        FirFilter,
        IirFilter,
        LchFarrow,
    )
    try:
        from dsp_spectrum import (
            FFTProcessorROCm as FFTProcessor,        # старое имя: FFTProcessor
            SpectrumMaximaFinderROCm,
            ComplexToMagROCm,
            FirFilterROCm,
            IirFilterROCm,
            LchFarrowROCm,
        )
    except ImportError:
        pass
except ImportError:
    pass  # spectrum опционален

# ── dsp_stats ─────────────────────────────────────────────────────
try:
    from dsp_stats import StatisticsProcessor
except ImportError:
    pass

# ── dsp_signal_generators ─────────────────────────────────────────
try:
    from dsp_signal_generators import (
        LfmAnalyticalDelayGenerator,
    )
    try:
        from dsp_signal_generators import (
            FormSignalGeneratorROCm,
            DelayedFormSignalGeneratorROCm,
            LfmAnalyticalDelayGeneratorROCm,
        )
    except ImportError:
        pass
except ImportError:
    pass

# ── dsp_heterodyne ────────────────────────────────────────────────
try:
    from dsp_heterodyne import HeterodyneDechirp
    try:
        from dsp_heterodyne import HeterodyneROCm
    except ImportError:
        pass
except ImportError:
    pass

# ── dsp_linalg ────────────────────────────────────────────────────
try:
    from dsp_linalg import CholeskyInverterROCm
except ImportError:
    pass

# ── dsp_radar ─────────────────────────────────────────────────────
try:
    from dsp_radar import FmCorrelatorROCm, RangeAngleProcessor
except ImportError:
    pass

# ── dsp_strategies ────────────────────────────────────────────────
try:
    from dsp_strategies import AntennaProcessorTest, WeightGenerator
except ImportError:
    pass
