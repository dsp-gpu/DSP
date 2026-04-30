"""
DSP-GPU Signal-to-Spectrum Test Suite (E2E)
============================================
Сквозной тест: NumPy SignalGenerator → GPU FFT → matplotlib визуализация.

Тесты:
  1. Multi-channel sin → FFT → display
  2. Different signal types (CW, LFM, Noise) → FFT
  3. Multi-beam CW with frequency step
  4. Generators from string parameters
  5. Multi-beam from string + FFT analysis
  6. Magnitude + Phase FFT output
  7. Universal generate() from string DSL

Удалено 2026-04-30 (микро-проект t_gpuworklib.py → t_signal_to_spectrum.py):
  - test_script_generator (runtime DSL → OpenCL kernel компилятор) — нет в DSP-GPU
  - test_script_fft_pipeline (то же)
  → Перспективная задача: MemoryBank/.future/TASK_script_dsl_rocm.md

История миграции:
  - SignalGenerator (legacy OpenCL) → NumPy через _NumpySignalGenerator (factories.py)
  - FFTProcessor → dsp_spectrum.FFTProcessorROCm
  - GPUContext → dsp_core.GPUContext (OpenCL) или ROCmGPUContext

Author: Kodo (AI Assistant)
Date: 2026-02-13 (migrated 2026-04-30: GPUWorkLib → DSP-GPU)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# DSP/Python в sys.path
_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)

PROJECT_ROOT = os.path.dirname(_PT_DIR)
PLOT_DIR = os.path.join(PROJECT_ROOT, "Results", "Plots", "integration")
os.makedirs(PLOT_DIR, exist_ok=True)

from common.gpu_loader import GPULoader
from common.runner import SkipTest

GPULoader.setup_path()  # добавляет DSP/Python/libs/ в sys.path

try:
    import dsp_core as core
    import dsp_spectrum as spectrum
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    core = None      # type: ignore
    spectrum = None  # type: ignore

# NumPy-обёртка для legacy SignalGenerator API (см. factories.py)
from integration.factories import _NumpySignalGenerator as _SignalGenerator


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _require_gpu():
    """Helper: единая точка проверки GPU."""
    if not HAS_GPU:
        raise SkipTest("dsp_core/dsp_spectrum not found — check build/libs")


# ============================================================================
# Test 1: Multi-channel sin -> FFT -> display
# ============================================================================
def test_multichannel_sin_fft():
    _require_gpu()
    print_header("Test 1: Multi-channel sin -> FFT")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    print(f"  GPU: {ctx.device_name}")

    fs = 4000.0       # Sample rate 4 kHz
    length = 4096      # Samples per channel
    freqs = [100, 250, 500, 800, 1200]  # Different frequencies

    fig, axes = plt.subplots(len(freqs), 3, figsize=(16, 3 * len(freqs)))
    fig.suptitle("Test 1: Multi-channel CW signals -> FFT", fontsize=14, fontweight='bold')

    for i, f0 in enumerate(freqs):
        # Generate CW signal
        signal = sig.generate_cw(freq=f0, fs=fs, length=length)

        # Compute FFT
        spectrum = fft.process_complex(signal, sample_rate=fs)

        # Time domain
        t = np.arange(length) / fs * 1000  # ms
        axes[i, 0].plot(t[:200], signal[:200].real, 'b-', linewidth=0.7)
        axes[i, 0].set_title(f"Ch {i+1}: CW f={f0} Hz (time)")
        axes[i, 0].set_xlabel("t, ms")
        axes[i, 0].set_ylabel("Re")
        axes[i, 0].grid(True, alpha=0.3)

        # FFT magnitude
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        mag = np.abs(spectrum)
        half = nfft // 2
        axes[i, 1].plot(freq_axis[:half], mag[:half], 'r-', linewidth=0.7)
        axes[i, 1].set_title(f"FFT magnitude (peak @ {freq_axis[np.argmax(mag[:half])]:.1f} Hz)")
        axes[i, 1].set_xlabel("f, Hz")
        axes[i, 1].set_ylabel("|FFT|")
        axes[i, 1].grid(True, alpha=0.3)

        # FFT phase
        phase = np.angle(spectrum)
        axes[i, 2].plot(freq_axis[:half], phase[:half], 'g-', linewidth=0.5, alpha=0.7)
        axes[i, 2].set_title("FFT phase")
        axes[i, 2].set_xlabel("f, Hz")
        axes[i, 2].set_ylabel("phase, rad")
        axes[i, 2].grid(True, alpha=0.3)

        # Print detected peak
        peak_bin = np.argmax(mag[:half])
        detected_freq = freq_axis[peak_bin]
        print(f"  Ch {i+1}: f0={f0:>6} Hz -> detected {detected_freq:>8.2f} Hz  "
              f"(error={abs(detected_freq - f0):.2f} Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test1_multichannel_fft.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 2: Different signal types (CW, LFM, Noise)
# ============================================================================
def test_signal_types():
    _require_gpu()
    print_header("Test 2: Signal types - CW, LFM, Noise")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 8000.0
    length = 8192

    # Generate signals
    cw = sig.generate_cw(freq=500, fs=fs, length=length, amplitude=2.0)
    lfm = sig.generate_lfm(f_start=200, f_end=1500, fs=fs, length=length)
    noise = sig.generate_noise(fs=fs, length=length, power=1.0, noise_type="gaussian")

    # FFT of each
    cw_fft = fft.process_complex(cw, sample_rate=fs)
    lfm_fft = fft.process_complex(lfm, sample_rate=fs)
    noise_fft = fft.process_complex(noise, sample_rate=fs)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)
    fig.suptitle("Test 2: Signal Types - CW, LFM, Noise", fontsize=14, fontweight='bold')

    signals = [
        ("CW (500 Hz)", cw, cw_fft),
        ("LFM (200->1500 Hz)", lfm, lfm_fft),
        ("Gaussian Noise", noise, noise_fft),
    ]

    t = np.arange(length) / fs * 1000  # ms

    for row, (name, signal, spectrum) in enumerate(signals):
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2
        mag = np.abs(spectrum)

        # Time domain (first 500 samples)
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(t[:500], signal[:500].real, 'b-', linewidth=0.5)
        ax1.set_title(f"{name} - Time")
        ax1.set_xlabel("t, ms")
        ax1.grid(True, alpha=0.3)

        # IQ scatter
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(signal[:1000].real, signal[:1000].imag, s=1, alpha=0.3, c='purple')
        ax2.set_title(f"{name} - IQ")
        ax2.set_xlabel("I")
        ax2.set_ylabel("Q")
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # FFT magnitude (dB)
        ax3 = fig.add_subplot(gs[row, 2])
        mag_db = 20 * np.log10(mag[:half] + 1e-10)
        ax3.plot(freq_axis[:half], mag_db, 'r-', linewidth=0.5)
        ax3.set_title(f"{name} - FFT (dB)")
        ax3.set_xlabel("f, Hz")
        ax3.set_ylabel("dB")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test2_signal_types.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode

    # Print statistics
    print(f"  CW:    mean={np.mean(np.abs(cw)):.3f}  std={np.std(cw.real):.3f}")
    print(f"  LFM:   mean={np.mean(np.abs(lfm)):.3f}  std={np.std(lfm.real):.3f}")
    print(f"  Noise: mean={np.mean(noise.real):.4f}  std={np.std(noise.real):.3f}  "
          f"var={np.var(noise.real):.3f}")
    print("  PASS")


# ============================================================================
# Test 3: Multi-beam CW with frequency step
# ============================================================================
def test_multibeam_cw():
    _require_gpu()
    print_header("Test 3: Multi-beam CW (8 beams, freq_step=50 Hz)")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 4000.0
    length = 4096
    beam_count = 8
    f0 = 100.0
    freq_step = 50.0

    # Generate multi-beam signal on GPU
    multi = sig.generate_cw(freq=f0, fs=fs, length=length,
                            beam_count=beam_count, freq_step=freq_step)
    print(f"  Shape: {multi.shape} (beams x samples)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Test 3: Multi-beam CW (8 beams, freq_step=50 Hz)", fontsize=14, fontweight='bold')

    # Spectrogram-like: FFT all beams
    all_spectra = []
    for b in range(beam_count):
        beam_data = multi[b, :]
        spectrum = fft.process_complex(beam_data, sample_rate=fs)
        nfft = len(spectrum)
        mag = np.abs(spectrum)
        all_spectra.append(mag[:nfft // 2])

        expected = f0 + b * freq_step
        peak_bin = np.argmax(mag[:nfft // 2])
        detected = peak_bin * fs / nfft
        print(f"  Beam {b}: expected {expected:>6.0f} Hz -> detected {detected:>8.2f} Hz")

    # Waterfall plot
    spectra_arr = np.array(all_spectra)
    freq_axis = np.arange(spectra_arr.shape[1]) * fs / nfft
    im = axes[0].imshow(spectra_arr, aspect='auto', origin='lower',
                         extent=[0, freq_axis[-1], 0, beam_count],
                         cmap='hot')
    axes[0].set_xlabel("Frequency, Hz")
    axes[0].set_ylabel("Beam index")
    axes[0].set_title("FFT magnitude (waterfall)")
    plt.colorbar(im, ax=axes[0], label="|FFT|")

    # Overlay: all beams FFT
    colors = plt.cm.viridis(np.linspace(0, 1, beam_count))
    for b in range(beam_count):
        label = f"Beam {b} ({f0 + b*freq_step:.0f} Hz)"
        axes[1].plot(freq_axis, all_spectra[b], color=colors[b], linewidth=0.8, label=label)
    axes[1].set_xlabel("Frequency, Hz")
    axes[1].set_ylabel("|FFT|")
    axes[1].set_title("All beams FFT overlay")
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(0, 600)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test3_multibeam.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 4: Create generators from string parameters
# ============================================================================
def test_generators_from_string():
    _require_gpu()
    print_header("Test 4: Generators from string params")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 4000.0
    length = 4096

    # CW from string
    cw_params = [
        "freq=100,amp=1.0",
        "freq=300,amp=0.8,phase=1.57",
        "freq=700,amp=0.5,freq_step=25",
    ]

    # LFM from string
    lfm_params = [
        "f_start=100,f_end=500,amp=1.0",
        "f_start=200,f_end=1800,amp=0.7",
    ]

    fig, axes = plt.subplots(len(cw_params) + len(lfm_params), 2, figsize=(14, 3 * (len(cw_params) + len(lfm_params))))
    fig.suptitle("Test 4: Generators from string parameters", fontsize=14, fontweight='bold')

    row = 0

    # CW from string
    for params_str in cw_params:
        signal = sig.generate_cw_from_string(params_str, fs=fs, length=length)
        spectrum = fft.process_complex(signal, sample_rate=fs)

        nfft = len(spectrum)
        t = np.arange(length) / fs * 1000
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[row, 0].plot(t[:300], signal[:300].real, 'b-', linewidth=0.7)
        axes[row, 0].set_title(f"CW: '{params_str}' (time)")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(freq_axis[:half], np.abs(spectrum[:half]), 'r-', linewidth=0.7)
        axes[row, 1].set_title("FFT")
        axes[row, 1].grid(True, alpha=0.3)

        print(f"  CW  '{params_str}' -> OK")
        row += 1

    # LFM from string
    for params_str in lfm_params:
        signal = sig.generate_lfm_from_string(params_str, fs=fs, length=length)
        spectrum = fft.process_complex(signal, sample_rate=fs)

        nfft = len(spectrum)
        t = np.arange(length) / fs * 1000
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[row, 0].plot(t[:500], signal[:500].real, 'b-', linewidth=0.5)
        axes[row, 0].set_title(f"LFM: '{params_str}' (time)")
        axes[row, 0].grid(True, alpha=0.3)

        mag_db = 20 * np.log10(np.abs(spectrum[:half]) + 1e-10)
        axes[row, 1].plot(freq_axis[:half], mag_db, 'r-', linewidth=0.5)
        axes[row, 1].set_title("FFT (dB)")
        axes[row, 1].grid(True, alpha=0.3)

        print(f"  LFM '{params_str}' -> OK")
        row += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test4_from_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 5: Multi-beam CW from string + FFT analysis
# ============================================================================
def test_multibeam_from_string():
    _require_gpu()
    print_header("Test 5: Multi-beam from string -> FFT")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 8000.0
    length = 8192
    beam_count = 4

    configs = [
        ("freq=200,amp=1.0,freq_step=100", "CW 200+100*n Hz"),
        ("freq=500,amp=0.5,freq_step=200", "CW 500+200*n Hz"),
    ]

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 5 * len(configs)))
    fig.suptitle("Test 5: Multi-beam from string -> FFT", fontsize=14, fontweight='bold')

    for idx, (params_str, title) in enumerate(configs):
        multi = sig.generate_cw_from_string(params_str, fs=fs, length=length, beam_count=beam_count)
        print(f"  '{params_str}' -> shape {multi.shape}")

        # Time domain overlay
        t = np.arange(length) / fs * 1000
        colors = plt.cm.tab10(np.linspace(0, 1, beam_count))
        for b in range(beam_count):
            axes[idx, 0].plot(t[:400], multi[b, :400].real, color=colors[b],
                              linewidth=0.7, label=f"Beam {b}")
        axes[idx, 0].set_title(f"{title} - Time domain")
        axes[idx, 0].set_xlabel("t, ms")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)

        # FFT overlay
        for b in range(beam_count):
            spectrum = fft.process_complex(multi[b, :], sample_rate=fs)
            nfft = len(spectrum)
            freq_axis = np.arange(nfft) * fs / nfft
            half = nfft // 2
            axes[idx, 1].plot(freq_axis[:half], np.abs(spectrum[:half]),
                              color=colors[b], linewidth=0.7, label=f"Beam {b}")

            peak = np.argmax(np.abs(spectrum[:half]))
            print(f"    Beam {b}: peak @ {freq_axis[peak]:.1f} Hz")

        axes[idx, 1].set_title(f"{title} - FFT")
        axes[idx, 1].set_xlabel("f, Hz")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test5_multibeam_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 6: Magnitude + Phase FFT output
# ============================================================================
def test_mag_phase():
    _require_gpu()
    print_header("Test 6: FFT Magnitude + Phase + Frequency")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 4000.0
    length = 4096

    signal = sig.generate_cw(freq=300, fs=fs, length=length)

    # Get mag/phase/freq
    result = fft.process_mag_phase(signal, sample_rate=fs)

    mag = result["magnitude"]
    phase = result["phase"]
    freq = result["frequency"]
    nfft = result["nFFT"]

    print(f"  nFFT = {nfft}")
    print(f"  magnitude shape: {mag.shape}")
    print(f"  phase shape:     {phase.shape}")
    print(f"  frequency shape: {freq.shape}")

    half = nfft // 2
    peak_idx = np.argmax(mag[:half])
    print(f"  Peak @ index {peak_idx}: freq={freq[peak_idx]:.2f} Hz, mag={mag[peak_idx]:.2f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Test 6: Magnitude + Phase + Frequency", fontsize=14, fontweight='bold')

    axes[0].plot(freq[:half], mag[:half], 'r-', linewidth=0.7)
    axes[0].set_title(f"Magnitude (peak @ {freq[peak_idx]:.1f} Hz)")
    axes[0].set_xlabel("f, Hz")
    axes[0].set_ylabel("|FFT|")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freq[:half], phase[:half], 'g-', linewidth=0.5)
    axes[1].set_title("Phase")
    axes[1].set_xlabel("f, Hz")
    axes[1].set_ylabel("rad")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq[:half], 20*np.log10(mag[:half] + 1e-10), 'b-', linewidth=0.7)
    axes[2].set_title("Magnitude (dB)")
    axes[2].set_xlabel("f, Hz")
    axes[2].set_ylabel("dB")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test6_mag_phase.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Test 7: Universal generate() from string (type=... in string)
# ============================================================================
def test_generate_from_string():
    _require_gpu()
    print_header("Test 7: Universal generate() from string")

    # Migration: GPUContext(OpenCL) → ROCmGPUContext (FFTProcessorROCm требует ROCm)
    ctx = core.ROCmGPUContext(0)
    sig = _SignalGenerator()                  # NumPy-обёртка (legacy API)
    fft = spectrum.FFTProcessorROCm(ctx)

    fs = 4000.0
    length = 4096

    # Different signal types from a single generate() method
    configs = [
        ("type=cw,freq=200,amp=1.0", "CW 200 Hz"),
        ("type=sin,freq=500,amp=0.8,phase=0.5", "Sin 500 Hz"),
        ("type=lfm,f_start=100,f_end=800,amp=1.0", "LFM 100->800 Hz"),
        ("type=chirp,f_start=50,f_end=1500", "Chirp 50->1500 Hz"),
        ("type=noise,power=1.0", "Gaussian Noise"),
        ("type=cw,freq=100,amp=1.0,freq_step=75", "CW multi (100+75*n Hz)"),
    ]

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 3 * len(configs)))
    fig.suptitle("Test 7: Universal generate() from string", fontsize=14, fontweight='bold')

    for i, (params_str, title) in enumerate(configs):
        beam_count = 4 if "freq_step" in params_str else 1
        signal = sig.generate(params_str, fs=fs, length=length, beam_count=beam_count)

        # For multi-beam, take first beam
        if signal.ndim == 2:
            plot_signal = signal[0, :]
        else:
            plot_signal = signal

        spectrum = fft.process_complex(plot_signal, sample_rate=fs)

        t = np.arange(len(plot_signal)) / fs * 1000
        nfft = len(spectrum)
        freq_axis = np.arange(nfft) * fs / nfft
        half = nfft // 2

        axes[i, 0].plot(t[:400], plot_signal[:400].real, 'b-', linewidth=0.6)
        axes[i, 0].set_title(f"'{params_str}'")
        axes[i, 0].set_xlabel("t, ms")
        axes[i, 0].grid(True, alpha=0.3)

        mag_db = 20 * np.log10(np.abs(spectrum[:half]) + 1e-10)
        axes[i, 1].plot(freq_axis[:half], mag_db, 'r-', linewidth=0.6)
        axes[i, 1].set_title(f"{title} - FFT (dB)")
        axes[i, 1].set_xlabel("f, Hz")
        axes[i, 1].grid(True, alpha=0.3)

        shape_str = str(signal.shape)
        print(f"  '{params_str}' -> shape {shape_str} OK")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test7_generate_from_string.png"), dpi=150)
    # plt.show()  # uncomment for interactive mode
    print("  PASS")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DSP-GPU Signal-to-Spectrum Test Suite")
    print("=" * 60)

    gpus = core.list_gpus()
    print(f"\n  Found {len(gpus)} GPU(s):")
    for g in gpus:
        print(f"    [{g['index']}] {g['name']} ({g['memory_mb']} MB)")

    test_multichannel_sin_fft()
    test_signal_types()
    test_multibeam_cw()
    test_generators_from_string()
    test_multibeam_from_string()
    test_mag_phase()
    test_generate_from_string()
    # test_script_generator + test_script_fft_pipeline удалены 2026-04-30:
    # ScriptGenerator (runtime DSL→kernel компилятор) не портирован в DSP-GPU.
    # См. перспективную задачу MemoryBank/.future/TASK_script_dsl_rocm.md

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
