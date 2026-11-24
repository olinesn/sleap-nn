"""Config for pytests."""

import pytest

import sleap_nn
from tests.fixtures.datasets import *
from tests.fixtures.inference import *
from tests.fixtures.legacy_models import *
from tests.fixtures.model_ckpts import *
from tests.fixtures.legacy_sleap_json_configs import *
from tests.fixtures.predictors import *


@pytest.fixture(autouse=True)
def _restore_log_sink_after_redirect(monkeypatch):
    """Put sleap_nn's log sink back on stdout after a test that moved it to stderr.

    ``--gui`` calls ``redirect_logs_to_stderr()``, which changes loguru's
    process-global sink for the rest of the process. A test running the CLI
    in-process with ``--gui`` would otherwise send every later test's logs (on the
    same pytest-xdist worker) to stderr, and tests that read the CLI's log output
    from stdout (``tests/utils/cli.py::run_cli``) would find it empty.
    """
    redirected = []
    real = sleap_nn.redirect_logs_to_stderr

    def redirect_and_note():
        redirected.append(True)
        real()

    monkeypatch.setattr(sleap_nn, "redirect_logs_to_stderr", redirect_and_note)
    yield
    if redirected:
        from loguru import logger

        logger.remove()
        sleap_nn._add_default_sink("stdout")
