"""Config for pytests."""

import os

import pytest
import torch

import sleap_nn


def _share_cpus_between_xdist_workers():
    """Give each pytest-xdist worker an equal share of the CPUs for torch.

    By default torch in every worker uses all cores, so with one worker per core
    they compete (N workers x N threads on N cores). Respects an explicit
    ``OMP_NUM_THREADS``; the env var also reaches subprocesses the tests start.
    """
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 0)
    if workers > 1 and "OMP_NUM_THREADS" not in os.environ:
        threads = max(1, (os.cpu_count() or 1) // workers)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)


_share_cpus_between_xdist_workers()

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
