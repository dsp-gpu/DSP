# Python Bindings (dsp_core)

> pybind11 модуль для GPU-вычислений из Python

**Модуль**: `dsp_core`
**Файл**: `build/python/Release/dsp_core.cp312-win_amd64.pyd`
**Python**: 3.12+
**Зависимости**: pybind11 3.0.1, numpy

---

## Содержание

| Файл | Описание |
|------|----------|
| [API.md](API.md) | Полный Python API Reference |

---

## Установка

```bash
# Сборка
cmake --build build --target gpuworklib_python --config Release

# Путь к модулю
set PYTHONPATH=E:\C++\DSP-GPU\build\python\Release
```

---

## Быстрый старт

```python
import dsp_core
import numpy as np

# GPU контекст (device 0)
ctx = dsp_core.GPUContext(0)
print(ctx.device_name)  # "NVIDIA GeForce RTX 2080 Ti"

# Генерация сигнала
gen = dsp_core.SignalGenerator(ctx)
signal = gen.generate_cw(
    beam_count=16, n_point=4096, sample_rate=1000.0,
    f0=100.0, freq_step=10.0
)
print(signal.shape)  # (16, 4096)

# FFT
fft = dsp_core.FFTProcessor(ctx)
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')

# ScriptGenerator
sg = dsp_core.ScriptGenerator(ctx)
sg.load("[Params]\nANTENNAS=8\nPOINTS=4096\n[Signal]\nres = sin(0.1f*(float)T)")
data = sg.generate()
```

---

## Классы

### GPUContext

```python
ctx = dsp_core.GPUContext(device_index=0)
ctx.device_name       # str: имя GPU
ctx.device_index      # int: индекс
ctx.global_memory_mb  # float: глобальная память (MB)
```

### SignalGenerator

```python
gen = dsp_core.SignalGenerator(ctx)

# CW: A * exp(j*(2*pi*f*t + phase))
data = gen.generate_cw(beam_count, n_point, sample_rate,
                       f0=100.0, phase=0.0, amplitude=1.0, freq_step=0.0)

# LFM: chirp f_start → f_end
data = gen.generate_lfm(beam_count, n_point, sample_rate,
                        f_start=100.0, f_end=500.0, amplitude=1.0)

# Noise: Gaussian / White
data = gen.generate_noise(beam_count, n_point,
                          noise_type='gaussian', power=1.0, seed=0)
```

Все методы возвращают `numpy.ndarray` shape `(beam_count, n_point)`, dtype `complex64`.
Для `beam_count=1` возвращается 1D массив `(n_point,)`.

### FFTProcessor

```python
fft = dsp_core.FFTProcessor(ctx)

# Complex output
spectrum = fft.process(data, sample_rate=1000.0, output_mode='complex')
# spectrum: numpy array, same shape as input, dtype=complex64

# Magnitude + Phase
mag, phase = fft.process(data, sample_rate=1000.0, output_mode='mag_phase')
# mag, phase: numpy arrays, dtype=float32

# Magnitude + Phase + Frequency
mag, phase, freq = fft.process(data, sample_rate=1000.0, output_mode='mag_phase_freq')
```

### ScriptGenerator

```python
sg = dsp_core.ScriptGenerator(ctx)

# Загрузка
sg.load(script_text)         # из строки
sg.load_file(file_path)      # из файла

# Генерация
data = sg.generate()          # numpy array (antennas, points) complex64

# Свойства
sg.antennas       # int
sg.points         # int
sg.kernel_source  # str: сгенерированный OpenCL kernel
sg.is_ready       # bool
```

---

## Типичные pipeline'ы

### Signal → FFT → Analysis

```python
ctx = dsp_core.GPUContext(0)
gen = dsp_core.SignalGenerator(ctx)
fft = dsp_core.FFTProcessor(ctx)

# 16 антенн, каждая с разной частотой
signal = gen.generate_cw(16, 4096, 1000.0, f0=50.0, freq_step=25.0)
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')

# Анализ
magnitudes = np.abs(spectrum)
peak_bins = np.argmax(magnitudes, axis=1)
peak_freqs = peak_bins * 1000.0 / spectrum.shape[1]
```

### ScriptGenerator → FFT

```python
sg = dsp_core.ScriptGenerator(ctx)
sg.load("""
[Params]
ANTENNAS = 16
POINTS = 4096
[Defs]
float freq = 50.0f + (float)ID * 25.0f
[Signal]
float angle = 2.0f * M_PI_F * freq / 1000.0f * (float)T
res_re = cos(angle)
res_im = sin(angle)
""")

signal = sg.generate()
fft = dsp_core.FFTProcessor(ctx)
spectrum = fft.process(signal, sample_rate=1000.0, output_mode='complex')
```

---

## Zero-Copy и GIL

- **GIL release**: Все GPU операции выполняются с `py::gil_scoped_release` для максимальной производительности
- **Zero-copy GPU→Python**: GPU буферы читаются напрямую в numpy через `py::capsule` для автоматического управления памятью
- **Формат данных**: `complex64` (2x float32) = OpenCL `float2`

---

## Файлы

```
python/
└── gpu_worklib_bindings.cpp    # pybind11 модуль (все классы)
```

---

## Тесты

Файл: `D:\Python\С++ to Python\test_gpuworklib.py`

| Тест | Описание |
|------|----------|
| 1 | GPUContext: создание, device info |
| 2 | CW Generator: single + multi-beam |
| 3 | LFM Generator: chirp signal |
| 4 | Noise Generator: Gaussian + White |
| 5 | FFT Complex: spectrum analysis |
| 6 | FFT MagPhase: magnitude + phase |
| 7 | SignalGenerator → FFT Pipeline |
| 8 | ScriptGenerator: 3 примера (real, IQ, conditional) |
| 9 | ScriptGenerator → FFTProcessor Pipeline |

---

*Обновлено: 2026-02-13*
