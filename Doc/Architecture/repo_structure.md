# DSP-GPU — Структура каждого репо

> **Дата**: 2026-04-12
> **Согласовано**: kernels в `kernels/rocm/` (не в include/)

---

## 1. Шаблон (применяется ко всем репо)

```
dsp-gpu/{repo}/
├── CMakeLists.txt              ← standalone build
├── CMakePresets.json           ← local-dev + ci presets
├── cmake/
│   ├── {Repo}Config.cmake.in  ← для find_package (с find_dependency!)
│   └── fetch_deps.cmake        ← FetchContent зависимостей (FIND_PACKAGE_ARGS)
│
├── include/                    ← PUBLIC headers (видны потребителям)
│   └── dsp/
│       └── {module}/
│           ├── processor.hpp
│           ├── params.hpp
│           └── types.hpp
│
├── kernels/                    ← PRIVATE kernel sources (НЕ видны потребителям)
│   └── rocm/                   ← HIP inline sources (*_kernels_rocm.hpp)
│       ├── *_kernels_rocm.hpp         ← R"HIP(...)" строки для hiprtc
│       ├── *_kernel_sources_rocm.hpp  ← multi-kernel compilation units
│       └── bin/                       ← compiled HSACO cache (.gitignore)
│
├── src/
│   └── {module}/               ← реализация (.cpp, .hip)
│
├── tests/                      ← C++ тесты
│   ├── all_test.hpp
│   ├── test_*.hpp
│   └── README.md
│
├── python/                     ← pybind11 биндинги (опционально)
│   ├── CMakeLists.txt
│   ├── dsp_{module}_module.cpp
│   └── dsp_{module}.pyi        ← type stubs для IDE
│
├── examples/
│   └── basic_usage.cpp
│
└── README.md
```

### Почему kernels/ отдельно от include/

**Проблема**: `include/` — PUBLIC директория в CMake. Если `*_kernels_rocm.hpp` лежит в `include/`, она **протечёт** к потребителям через `target_include_directories(... PUBLIC include)`.

**Решение** (по аналогии с .cl файлами в DSP-GPU):
- `.cl` файлы уже лежат в `modules/{module}/kernels/` (отдельно от include)
- `.hpp` kernel sources → `kernels/rocm/` (та же логика)
- CMake: `target_include_directories(... PRIVATE kernels/)`

```cmake
# PUBLIC — только dsp/{module}/ (видны downstream)
target_include_directories(DspSpectrum
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
)
```

В `.cpp`:
```cpp
#include "rocm/heterodyne_kernels_rocm.hpp"  // из PRIVATE kernels/
```

---

## 2. core — детальная структура

```
dsp-gpu/core/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   ├── DspCoreConfig.cmake.in    ← find_dependency(hip, OpenCL)
│   └── FindROCm.cmake
│
├── include/
│   └── dsp/
│       ├── drv_gpu.hpp            ← главный интерфейс core
│       ├── backends/
│       │   ├── rocm_backend.hpp
│       │   └── opencl_backend.hpp
│       ├── common/
│       │   ├── gpu_device_info.hpp
│       │   ├── backend_type.hpp
│       │   └── i_backend.hpp
│       ├── profiler/
│       │   └── gpu_profiler.hpp
│       ├── context/
│       │   ├── gpu_context.hpp
│       │   └── gpu_kernel_op.hpp
│       ├── memory/
│       │   ├── batch_manager.hpp
│       │   └── buffer_set.hpp
│       ├── services/
│       │   └── kernel_cache_service.hpp
│       └── config/
│           └── gpu_config.hpp
│
├── src/
│   ├── backends/
│   │   ├── rocm/
│   │   └── opencl/
│   ├── context/
│   ├── config/
│   ├── profiler/
│   └── services/
│
├── test_utils/                    ← ОТДЕЛЬНАЯ target DspCore::TestUtils
│   ├── CMakeLists.txt             ← add_library(DspCoreTestUtils INTERFACE)
│   └── include/dsp/test/
│       ├── test_runner.hpp
│       ├── gpu_test_base.hpp
│       └── gpu_benchmark_base.hpp
│
├── tests/
│   └── test_drv_gpu.hpp
│
├── python/
│   ├── CMakeLists.txt
│   ├── dsp_core_module.cpp
│   └── dsp_core.pyi
│
└── configGPU.json
```

---

## 3. spectrum — детальная структура

```
dsp-gpu/spectrum/
├── CMakeLists.txt
├── cmake/
│   ├── DspSpectrumConfig.cmake.in  ← find_dependency(DspCore, hip, hipfft)
│   └── fetch_deps.cmake
│
├── include/
│   └── dsp/
│       ├── fft/
│       │   ├── fft_processor_rocm.hpp
│       │   ├── complex_to_mag_phase_rocm.hpp
│       │   ├── all_maxima_pipeline_rocm.hpp
│       │   └── factory/
│       │       └── spectrum_processor_factory.hpp
│       ├── filters/
│       │   ├── fir_filter_rocm.hpp
│       │   ├── iir_filter_rocm.hpp
│       │   ├── kalman_filter_rocm.hpp
│       │   ├── kaufman_filter_rocm.hpp
│       │   └── moving_average_filter_rocm.hpp
│       └── lch_farrow/
│           └── lch_farrow_rocm.hpp
│
├── kernels/
│   └── rocm/
│       ├── complex_to_mag_phase_kernels_rocm.hpp
│       ├── fft_processor_kernels_rocm.hpp
│       ├── fft_kernel_sources_rocm.hpp
│       ├── all_maxima_kernel_sources_rocm.hpp
│       ├── fir_kernels_rocm.hpp
│       ├── iir_kernels_rocm.hpp
│       ├── kalman_kernels_rocm.hpp
│       ├── kaufman_kernels_rocm.hpp
│       └── moving_average_kernels_rocm.hpp
│
├── src/
│   ├── fft/
│   ├── filters/
│   └── lch_farrow/
│
├── tests/
│   ├── all_test.hpp
│   ├── test_fft_rocm.hpp
│   ├── test_filters_rocm.hpp
│   └── test_lch_farrow_rocm.hpp
│
└── python/
    ├── CMakeLists.txt
    ├── dsp_spectrum_module.cpp
    └── dsp_spectrum.pyi
```

---

## 4. Остальные репо (краткие структуры)

### stats
```
dsp-gpu/stats/
├── include/dsp/statistics/
│   └── statistics_processor.hpp
├── kernels/rocm/
│   └── (inline kernel sources для welford, medians, radix sort)
├── src/statistics/
└── python/ → dsp_stats.pyd
```

### linalg
```
dsp-gpu/linalg/
├── include/dsp/
│   ├── vector_algebra/
│   │   └── cholesky_inverter_rocm.hpp
│   └── capon/
│       └── capon_processor_rocm.hpp
├── kernels/rocm/
│   ├── symmetrize_kernel_sources_rocm.hpp
│   └── capon_kernels_rocm.hpp
├── src/
│   ├── vector_algebra/
│   └── capon/
└── python/ → dsp_linalg.pyd
```

### signal_generators
```
dsp-gpu/signal_generators/
├── include/dsp/signal_generators/
│   ├── cw_generator_rocm.hpp
│   ├── lfm_generator_rocm.hpp
│   ├── noise_generator_rocm.hpp
│   ├── form_signal_generator_rocm.hpp
│   └── signal_generator_factory.hpp
├── kernels/rocm/
│   ├── cw_kernels_rocm.hpp
│   ├── lfm_kernels_rocm.hpp
│   ├── noise_kernels_rocm.hpp
│   └── form_signal_kernels_rocm.hpp
├── src/signal_generators/
└── python/ → dsp_signal_generators.pyd
```

### heterodyne
```
dsp-gpu/heterodyne/
├── include/dsp/heterodyne/
│   ├── heterodyne_dechirp.hpp
│   └── heterodyne_processor_rocm.hpp
├── kernels/rocm/
│   └── heterodyne_kernels_rocm.hpp
├── src/heterodyne/
└── python/ → dsp_heterodyne.pyd
```

### radar
```
dsp-gpu/radar/
├── include/dsp/
│   ├── range_angle/
│   │   └── range_angle_processor.hpp
│   └── fm_correlator/
│       └── fm_correlator_rocm.hpp
├── kernels/rocm/
│   ├── range_angle_kernels_rocm.hpp
│   └── fm_kernels_rocm.hpp
├── src/
│   ├── range_angle/
│   └── fm_correlator/
└── python/ → dsp_radar.pyd
```

### strategies
```
dsp-gpu/strategies/
├── include/dsp/strategies/
│   └── antenna_processor.hpp
├── kernels/rocm/
│   └── strategies_kernels_rocm.hpp
├── src/strategies/
└── python/ → dsp_strategies.pyd
```

---

*Сгенерировано: 2026-04-12*
