<!-- type:meta_overview repo:DSP source:DSP/Python/ -->

# DSP — Python Modules Index

DSP — мета-репо без C++ кода, содержит Python-биндинги и тесты для всех 8 модулей DSP-GPU.

## Python модули по репо

### `common/`

**Тесты** (4 файлов):
- `common/io/t_smoke.py`
- `common/plotting/t_smoke.py`
- `common/references/t_references_smoke.py`
- `common/validators/t_smoke.py`

**Модули** (8):
- `__init__.py`
- `base.py`
- `configs.py`
- `gpu_context.py`
- `gpu_loader.py`
- `reporters.py`
- `result.py`
- `runner.py`

### `heterodyne/`

**Тесты** (4 файлов):
- `heterodyne/t_heterodyne.py`
- `heterodyne/t_heterodyne_comparison.py`
- `heterodyne/t_heterodyne_rocm.py`
- `heterodyne/t_heterodyne_step_by_step.py`

**Модули** (2):
- `factories.py`
- `heterodyne_base.py`

### `integration/`

**Тесты** (5 файлов):
- `integration/t_fft_integration.py`
- `integration/t_hybrid_backend.py`
- `integration/t_signal_gen_integration.py`
- `integration/t_signal_to_spectrum.py`
- `integration/t_zero_copy.py`

**Модули** (1):
- `factories.py`

### `libs/`

### `linalg/`

**Тесты** (3 файлов):
- `linalg/t_capon.py`
- `linalg/t_cholesky_inverter_rocm.py`
- `linalg/t_matrix_csv_comparison.py`

**Модули** (1):
- `factories.py`

### `radar/`

**Тесты** (3 файлов):
- `radar/t_fm_correlator.py`
- `radar/t_fm_correlator_rocm.py`
- `radar/t_range_angle.py`

**Модули** (1):
- `__init__.py`

### `signal_generators/`

**Тесты** (4 файлов):
- `signal_generators/t_delayed_form_signal.py`
- `signal_generators/t_form_signal.py`
- `signal_generators/t_form_signal_rocm.py`
- `signal_generators/t_lfm_analytical_delay.py`

**Модули** (3):
- `example_form_signal.py`
- `factories.py`
- `signal_base.py`

### `spectrum/`

**Тесты** (15 файлов):
- `spectrum/ai_pipeline/t_ai_pipeline.py`
- `spectrum/t_ai_filter_pipeline.py`
- `spectrum/t_ai_fir_demo.py`
- `spectrum/t_filters_stage1.py`
- `spectrum/t_fir_filter_rocm.py`
- `spectrum/t_iir_filter_rocm.py`
- `spectrum/t_iir_plot.py`
- `spectrum/t_kalman_rocm.py`
- `spectrum/t_kaufman_rocm.py`
- `spectrum/t_lch_farrow.py`
- `spectrum/t_lch_farrow_rocm.py`
- `spectrum/t_moving_average_rocm.py`
- `spectrum/t_process_magnitude_rocm.py`
- `spectrum/t_spectrum_find_all_maxima_rocm.py`
- `spectrum/t_spectrum_maxima_finder_rocm.py`

**Модули** (3):
- `factories.py`
- `filter_base.py`
- `plot_report_filters.py`

### `stats/`

**Тесты** (4 файлов):
- `stats/t_compute_all.py`
- `stats/t_snr_estimator.py`
- `stats/t_statistics_float_rocm.py`
- `stats/t_statistics_rocm.py`

**Модули** (1):
- `factories.py`

### `strategies/`

**Тесты** (8 файлов):
- `strategies/t_base_pipeline.py`
- `strategies/t_debug_steps.py`
- `strategies/t_farrow_pipeline.py`
- `strategies/t_params.py`
- `strategies/t_scenario_builder.py`
- `strategies/t_strategies_pipeline.py`
- `strategies/t_strategies_step_by_step.py`
- `strategies/t_timing_analysis.py`

**Модули** (11):
- `debug_pipeline_steps.py`
- `factories.py`
- `farrow_delay.py`
- `numpy_reference.py`
- `pipeline_runner.py`
- `pipeline_step_validator.py`
- `plot_strategies_results.py`
- `scenario_builder.py`
- `signal_factory.py`
- `signal_generators_strategy.py`
- `strategy_base.py`

## Doxyfile файлы

- `DSP/Doc/Doxygen/Doxyfile`
- `DSP/Doc/Doxygen/DrvGPU/Doxyfile`
- `DSP/Doc/Doxygen/modules/capon/Doxyfile`
- `DSP/Doc/Doxygen/modules/fft_func/Doxyfile`
- `DSP/Doc/Doxygen/modules/filters/Doxyfile`
- `DSP/Doc/Doxygen/modules/fm_correlator/Doxyfile`
- `DSP/Doc/Doxygen/modules/heterodyne/Doxyfile`
- `DSP/Doc/Doxygen/modules/lch_farrow/Doxyfile`
- `DSP/Doc/Doxygen/modules/range_angle/Doxyfile`
- `DSP/Doc/Doxygen/modules/signal_generators/Doxyfile`
- `DSP/Doc/Doxygen/modules/statistics/Doxyfile`
- `DSP/Doc/Doxygen/modules/strategies/Doxyfile`
- `DSP/Doc/Doxygen/modules/vector_algebra/Doxyfile`

