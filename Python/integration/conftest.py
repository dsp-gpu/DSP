"""
conftest.py — proxy на factories.py (legacy compat для старых импортов
`from integration.conftest import ...`).

Реальные factory functions — в `factories.py`.
"""

from integration.factories import (  # noqa: F401
    make_sig_gen,
    make_fft_proc,
    make_script_gen,
    integration_plot_dir,
)
