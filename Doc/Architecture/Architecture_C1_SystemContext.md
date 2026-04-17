# C1 — System Context Diagram

> **Project**: DSP-GPU
> **Date**: 2026-03-28
> **Reference**: [c4model.com](https://c4model.com)
> **Level**: 1 (System Context) — самый высокий уровень абстракции

---

## 1. Описание

**DSP-GPU** — библиотека GPU-вычислений для цифровой обработки сигналов (ЦОС).
Предоставляет модули генерации, FFT, фильтрации, гетеродинирования и спектрального анализа.

---

## 2. System Context Diagram (ASCII)

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                        ПОЛЬЗОВАТЕЛИ                                 │
 │                                                                      │
 │  ┌─────────────┐   ┌─────────────────┐   ┌───────────────────┐     │
 │  │  C++ Dev    │   │  Python Data    │   │  CI/CD Pipeline   │     │
 │  │  (Engineer) │   │  Scientist      │   │  (GitHub Actions) │     │
 │  └──────┬──────┘   └───────┬─────────┘   └────────┬──────────┘     │
 └─────────┼──────────────────┼──────────────────────┼─────────────────┘
           │                  │                      │
           │ C++ API          │ Python API           │ cmake --build
           │ (include/*.hpp)  │ (pybind11/NumPy)     │ + ctest
           │                  │                      │
           ▼                  ▼                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │                    ╔══════════════════════════╗                       │
 │                    ║      DSP-GPU          ║                       │
 │                    ║                          ║                       │
 │                    ║  GPU Signal Processing   ║                       │
 │                    ║  Library                 ║                       │
 │                    ║                          ║                       │
 │                    ║  - Signal Generation     ║                       │
 │                    ║  - FFT / IFFT + Maxima   ║                       │
 │                    ║  - Statistics (Welford)  ║                       │
 │                    ║  - FIR / IIR Filters     ║                       │
 │                    ║  - Heterodyne Dechirp    ║                       │
 │                    ║  - Fractional Delay      ║                       │
 │                    ║  - FM Correlation        ║                       │
 │                    ║  - Digital Beamforming   ║                       │
 │                    ║  - MVDR Capon            ║                       │
 │                    ║  - 3D Range-Angle        ║                       │
 │                    ╚═══════════╤══════════════╝                       │
 │                                │                                     │
 └────────────────────────────────┼─────────────────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
 ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
 │  GPU Hardware   │  │  External Libs  │  │  Host OS / FS        │
 │                 │  │                 │  │                      │
 │  NVIDIA (OpenCL)│  │  clFFT          │  │  Logs/DRVGPU_XX/     │
 │  AMD (ROCm/HIP)│  │  hipFFT         │  │  Results/JSON/       │
 │  Intel (OpenCL) │  │  plog           │  │  Results/Profiler/   │
 │                 │  │  pybind11       │  │  configGPU.json      │
 │  1..N устройств │  │  NumPy          │  │  Kernel cache (.bin) │
 └─────────────────┘  └─────────────────┘  └──────────────────────┘
```

---

## 3. Акторы и системы

### Пользователи (People)

| Актор | Описание | Взаимодействие |
|-------|----------|----------------|
| **C++ Engineer** | Разработчик ЦОС-приложений | Использует C++ API напрямую: `#include <drv_gpu.hpp>` |
| **Python Data Scientist** | Исследователь / аналитик | Использует Python API: `import gpu_worklib` |
| **CI/CD Pipeline** | Автоматизация сборки и тестов | CMake build + ctest (15 C++ тестов, 9+ Python тестов) |

### Внешние системы (External Systems)

| Система | Технология | Назначение |
|---------|-----------|------------|
| **GPU Hardware** | NVIDIA / AMD / Intel | Аппаратное ускорение (OpenCL / ROCm) |
| **clFFT** | C library | БПФ для OpenCL backend |
| **hipFFT** | C library | БПФ для ROCm backend |
| **plog** | C++ header-only | Логирование (per-GPU файлы) |
| **pybind11** | C++ ↔ Python bridge | Python-биндинги (NumPy integration) |
| **Host OS / FS** | Windows / Linux | Файловая система (логи, конфиг, кеш ядер, результаты) |

---

## 4. Границы системы

```
                    ┌─── Граница DSP-GPU ───────────────────────┐
                    │                                               │
  User Code ──────▶ │  core + Modules + Python Bindings         │
                    │                                               │
                    │  Ответственность:                             │
                    │  ✅ Абстракция GPU (multi-backend)            │
                    │  ✅ Управление памятью GPU                    │
                    │  ✅ Генерация сигналов (CW, LFM, Noise, DSL) │
                    │  ✅ FFT/IFFT с пост-обработкой                │
                    │  ✅ Поиск спектральных максимумов             │
                    │  ✅ FIR/IIR фильтрация                       │
                    │  ✅ Гетеродинирование (LFM Dechirp)          │
                    │  ✅ Дробная задержка (Lagrange/Farrow)        │
                    │  ✅ Профилирование и логирование              │
                    │  ✅ Batch-обработка больших данных             │
                    │                                               │
                    │  Вне ответственности:                         │
                    │  ❌ Установка GPU-драйверов                   │
                    │  ❌ Сборка clFFT/hipFFT (внешняя зависимость)│
                    │  ❌ GUI / визуализация (через Python)         │
                    │  ❌ Сетевые протоколы                         │
                    └───────────────────────────────────────────────┘
```

---

## 5. PlantUML (для рендеринга)

```plantuml
@startuml C1_SystemContext
!include <C4/C4_Context>

' Лейаут и тема
LAYOUT_TOP_DOWN()
LAYOUT_WITH_LEGEND()
!theme C4_united from <C4/themes>

title DSP-GPU — C1: System Context Diagram

Person(cpp_dev, "C++ Engineer", "Разработчик ЦОС-приложений")
Person(py_sci, "Python Scientist", "Исследователь / аналитик данных")
Person(ci_cd, "CI/CD Pipeline", "Автоматическая сборка и тесты")

System(dsp_core, "DSP-GPU", "Библиотека GPU-вычислений для ЦОС:\nFFT, фильтры, генераторы, гетеродин")

System_Ext(gpu_hw, "GPU Hardware", "NVIDIA (OpenCL)\nAMD (ROCm/HIP)\nIntel (OpenCL)")
System_Ext(clfft, "clFFT / hipFFT", "Библиотеки БПФ")
System_Ext(plog_lib, "plog", "Логирование (header-only)")
System_Ext(pybind, "pybind11", "C++ ↔ Python bridge")
System_Ext(host_fs, "Host OS / FS", "Логи, конфиг, кеш, результаты")

' Связи от людей к системе – сверху
Rel(cpp_dev, dsp_core, "C++ API", "#include <drv_gpu.hpp>")
Rel(py_sci, dsp_core, "Python API", "import gpu_worklib")
Rel(ci_cd, dsp_core, "Build & Test", "cmake + ctest")

' Внешние зависимости – ниже/сбоку
Rel(dsp_core, gpu_hw, "Compute", "OpenCL / HIP API")
Rel(dsp_core, clfft, "FFT", "clFFT / hipFFT API")
Rel(dsp_core, plog_lib, "Logging", "plog macros")
Rel(dsp_core, pybind, "Bindings", "pybind11 module")
Rel(dsp_core, host_fs, "I/O", "Read/Write files")

SHOW_LEGEND()
@enduml

```

---

*Следующий уровень: [C2 — Container Diagram](Architecture_C2_Container.md)*
