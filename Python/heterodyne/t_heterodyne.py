"""
test_heterodyne.py — Tests for HeterodyneROCm (LFM dechirp pipeline)

Tests:
  1. Basic dechirp — single antenna, known delay, verify f_beat
  2. Multiple antennas — 5 antennas, linear delays, verify range
  3. SNR verification — all SNR values > 0 dB
  4. Plot: f_beat vs delay (linear relationship) + SNR bars

Parameters: fs=12MHz, B=2MHz, N=8000, mu=3e9 Hz/s
search_range=5000 => half_range=2500 => left bins [0..2499] (~0..3.66 MHz)

TODO(Debian 2026-05-03+): первый запуск после миграции с HeterodyneDechirp на
HeterodyneROCm.dechirp/correct + np.fft. Подкрутить F_BEAT_TOL_HZ если float32
даст расхождение vs legacy результаты.

@author Kodo (AI Assistant)
@date 2026-02-21 (migrated 2026-04-30: HeterodyneDechirp → HeterodyneROCm + np.fft)
"""

import sys
import os
import numpy as np

_PT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PT_DIR not in sys.path:
    sys.path.insert(0, _PT_DIR)
from common.runner import SkipTest, TestRunner
from common.gpu_loader import GPULoader

GPULoader.setup_path()  # добавляет DSP/Python/libs/ в sys.path

try:
    import dsp_core as core
    import dsp_heterodyne as heterodyne
    HAS_HETERODYNE = hasattr(heterodyne, 'HeterodyneROCm')
except ImportError:
    HAS_HETERODYNE = False
    core = None        # type: ignore
    heterodyne = None  # type: ignore


# ============================================================================
# Constants (match C++ test parameters)
# ============================================================================

FS = 12e6           # sample rate, Hz
F_START = 0.0       # LFM start frequency, Hz
F_END = 2e6         # LFM end frequency, Hz
N = 8000            # samples per antenna
ANTENNAS = 5
BANDWIDTH = F_END - F_START  # 2 MHz
DURATION = N / FS            # 666.67 us
MU = BANDWIDTH / DURATION    # 3e9 Hz/s  (chirp rate)
C_LIGHT = 3e8                # speed of light, m/s

DELAYS_US = [100, 200, 300, 400, 500]  # delays in microseconds
F_BEAT_TOL_HZ = 5000.0                 # tolerance +/- 5 kHz

PLOTS_DIR = os.path.join(_PT_DIR, '..', 'Results', 'Plots', 'heterodyne')


# ============================================================================
# Helpers: LFM signal generation (CPU reference) + dechirp+FFT pipeline
# ============================================================================

def generate_lfm_rx(delays_us, f_start=F_START, f_end=F_END, fs=FS, n=N):
    """Generate delayed LFM signal (complex IQ) for each antenna."""
    t = np.arange(n, dtype=np.float32) / fs
    mu = (f_end - f_start) / (n / fs)
    rx = np.zeros((len(delays_us), n), dtype=np.complex64)
    for i, delay_us in enumerate(delays_us):
        tau = delay_us * 1e-6
        t_delayed = t - tau
        phase = 2 * np.pi * (0.5 * mu * t_delayed**2 + f_start * t_delayed)
        rx[i, :] = np.exp(1j * phase).astype(np.complex64)
    return rx


def generate_lfm_reference(f_start=F_START, f_end=F_END, fs=FS, n=N):
    """Опорный (un-delayed) LFM сигнал — нужен для dechirp = rx * conj(ref)."""
    return generate_lfm_rx([0.0], f_start, f_end, fs, n).ravel()


def heterodyne_pipeline(het, rx, num_antennas, fs=FS, n=N, mu=MU):
    """Полный pipeline после миграции: GPU dechirp + CPU FFT/argmax/SNR.

    Возвращает структуру совместимую с legacy result:
      {
        'success': True,
        'antennas': [
          {'f_beat_hz': float, 'peak_snr_db': float, 'range_m': float},
          ...
        ]
      }
    """
    ref = generate_lfm_reference(fs=fs, n=n)
    # broadcast ref на все антенны
    ref_multi = np.tile(ref, num_antennas).astype(np.complex64)
    rx_flat = rx.ravel().astype(np.complex64)

    # GPU: dechirp = rx * conj(ref)
    dc = het.dechirp(rx_flat, ref_multi)
    dc = dc.reshape(num_antennas, n)

    # CPU: FFT + поиск пика + SNR на каждую антенну
    antennas = []
    for ant_idx in range(num_antennas):
        spec = np.fft.fft(dc[ant_idx])
        mag = np.abs(spec)
        # ищем в первой половине (низкочастотные beats)
        half = n // 2
        peak_bin = int(np.argmax(mag[:half]))
        f_beat = peak_bin * fs / n
        peak_val = mag[peak_bin]

        # SNR: peak vs avg от двух соседей вне guard
        guard = 5
        left  = max(peak_bin - guard - 5, 0)
        right = min(peak_bin + guard + 5, half - 1)
        noise_floor = (mag[left] + mag[right]) / 2.0 + 1e-12
        snr_db = 20.0 * np.log10(peak_val / noise_floor)

        # Range из f_beat: R = c * f_beat / (2 * mu)
        range_m = C_LIGHT * f_beat / (2.0 * mu)

        antennas.append({
            'f_beat_hz': float(f_beat),
            'peak_snr_db': float(snr_db),
            'range_m': float(range_m),
        })
    return {'success': True, 'error_message': '', 'antennas': antennas}


# ============================================================================
# Tests
# ============================================================================

class TestHeterodyne:
    """Tests for HeterodyneROCm (legacy HeterodyneDechirp.process replaced
    with dechirp + np.fft + argmax pipeline — see heterodyne_pipeline helper)."""

    def setUp(self):
        if not HAS_HETERODYNE:
            raise SkipTest("dsp_core/dsp_heterodyne not found — check build/libs")
        self._ctx = core.ROCmGPUContext(0)
        self._het = heterodyne.HeterodyneROCm(self._ctx)
        self._het.set_params(float(F_START), float(F_END), float(FS), int(N), int(ANTENNAS))

    def test_basic_dechirp_single_antenna(self):
        """Single antenna with delay=100us -> f_beat=300kHz."""
        het = heterodyne.HeterodyneROCm(self._ctx)
        het.set_params(float(F_START), float(F_END), float(FS), int(N), 1)

        rx = generate_lfm_rx([100.0])
        result = heterodyne_pipeline(het, rx, num_antennas=1)

        assert result['success'], f"Process failed: {result['error_message']}"
        assert len(result['antennas']) == 1

        expected_f_beat = MU * 100e-6
        actual_f_beat = result['antennas'][0]['f_beat_hz']
        error = abs(actual_f_beat - expected_f_beat)

        print(f"\n  Expected: {expected_f_beat:.0f} Hz  (mu * tau = 3e9 * 100e-6)")
        print(f"  Actual:   {actual_f_beat:.0f} Hz")
        print(f"  Error:    {error:.0f} Hz  (tolerance: {F_BEAT_TOL_HZ:.0f} Hz)")

        assert error < F_BEAT_TOL_HZ, \
            f"f_beat error {error:.0f} Hz exceeds tolerance {F_BEAT_TOL_HZ:.0f} Hz"

    def test_multiple_antennas_range(self):
        """5 antennas with delays [100..500] us, verify f_beat and range."""
        rx = generate_lfm_rx(DELAYS_US)
        result = heterodyne_pipeline(self._het, rx, num_antennas=ANTENNAS)

        assert result['success'], f"Process failed: {result['error_message']}"
        assert len(result['antennas']) == ANTENNAS

        print(f"\n  Formula: f_beat = mu * tau,  R = c * T * f_beat / (2 * B)")
        print(f"  mu={MU:.2e} Hz/s, T={DURATION*1e6:.2f} us, B={BANDWIDTH/1e6:.0f} MHz")
        print(f"  {'Ant':>3} | {'Delay us':>8} | {'f_beat Hz':>11} | {'Expected':>11} | "
              f"{'Error':>8} | {'Range m':>9}")
        print(f"  {'-'*3} | {'-'*8} | {'-'*11} | {'-'*11} | {'-'*8} | {'-'*9}")

        for i, ant in enumerate(result['antennas']):
            delay_us = DELAYS_US[i]
            expected_f = MU * delay_us * 1e-6
            actual_f = ant['f_beat_hz']
            error = abs(actual_f - expected_f)
            print(f"  {i:3d} | {delay_us:8.0f} | {actual_f:11.0f} | {expected_f:11.0f} | "
                  f"{error:8.0f} | {ant['range_m']:9.2f}")
            assert error < F_BEAT_TOL_HZ, \
                f"Antenna {i}: f_beat error {error:.0f} Hz exceeds tolerance"

    def test_snr_positive(self):
        """All SNR values should be > 0 dB for clean LFM signal."""
        rx = generate_lfm_rx(DELAYS_US)
        result = heterodyne_pipeline(self._het, rx, num_antennas=ANTENNAS)
        assert result['success']

        print()
        print(f"  SNR = 20*log10(peak_mag / avg(left_bin, right_bin))")
        for i, ant in enumerate(result['antennas']):
            snr = ant['peak_snr_db']
            tau_us = DELAYS_US[i]
            f_beat = ant['f_beat_hz']
            print(f"  Ant {i}: delay={tau_us} us, f_beat={f_beat/1e3:.1f} kHz, SNR={snr:.1f} dB")
            assert snr > 0.0, f"Antenna {i}: SNR {snr:.1f} dB <= 0"
        print("  All SNR > 0 dB — PASSED")

    def test_plot_f_beat_vs_delay(self):
        """Generate multi-panel report plot for the heterodyne dechirp pipeline."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            return  # matplotlib not available — skip silently

        rx = generate_lfm_rx(DELAYS_US)
        result = heterodyne_pipeline(self._het, rx, num_antennas=ANTENNAS)
        assert result['success']

        delays_arr = np.array(DELAYS_US, dtype=float)
        delays_s   = delays_arr * 1e-6
        expected_f = MU * delays_s
        actual_f   = np.array([a['f_beat_hz'] for a in result['antennas']])
        snr_vals   = np.array([a['peak_snr_db'] for a in result['antennas']])
        ranges     = np.array([a['range_m'] for a in result['antennas']])
        errors_hz  = np.abs(actual_f - expected_f)
        true_r     = C_LIGHT * delays_s / 2.0

        fig = plt.figure(figsize=(14, 11))
        gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.40,
                       top=0.88, bottom=0.09)

        fig.suptitle('LFM Dechirp — GPU результаты (HeterodyneROCm)',
                     fontsize=13, fontweight='bold')
        fig.text(0.5, 0.915,
                 f'fs={FS/1e6:.0f} MHz | B={BANDWIDTH/1e6:.0f} MHz | '
                 f'N={N} | T={DURATION*1e6:.1f} мкс | '
                 f'μ={MU:.2e} Гц/с | {ANTENNAS} антенн',
                 ha='center', fontsize=9, color='gray')

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(delays_arr, expected_f / 1e3, 'b--', lw=2, alpha=0.7,
                 label='Теория: f = μ·τ')
        ax1.plot(delays_arr, actual_f / 1e3, 'ro-', ms=9, lw=1.5,
                 label='GPU результат')
        ax1.set_xlabel('Задержка τ [мкс]')
        ax1.set_ylabel('f_beat [кГц]')
        ax1.set_title('Частота биений f_beat vs задержка', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        bar_c = ['#2ecc71' if s > 10 else '#f39c12' if s > 0 else '#e74c3c'
                  for s in snr_vals]
        ax2.bar(range(ANTENNAS), snr_vals, color=bar_c, alpha=0.85, ec='black', lw=0.5)
        ax2.set_xlabel('Антенна')
        ax2.set_ylabel('SNR [дБ]')
        ax2.set_title('Отношение сигнал/шум по антеннам', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = os.path.join(PLOTS_DIR, 'test_heterodyne_results.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved: {plot_path}")


if __name__ == '__main__':
    runner = TestRunner()
    results = runner.run(TestHeterodyne())
    runner.print_summary(results)
