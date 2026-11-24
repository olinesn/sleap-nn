"""Spawn the ``sleap-nn`` CLI from a test WITHOUT re-syncing the project environment.

The CLI tests used to shell out to::

    uv run --frozen --no-group gpu --extra torch-cpu sleap-nn ...

``--frozen`` only pins the lockfile; ``uv run`` still **syncs the project's ``.venv``**
to the requested groups/extras before running. On CI that is a no-op (the job already
synced with exactly those flags), but on a developer machine whose default ``gpu``
group installed ``torch+cu130`` it silently uninstalled torch/torchvision and put the
``+cpu`` wheels in their place -- mid-suite, for every other process sharing the venv.
It is the "venv keeps flipping to CPU torch" hazard that cost several GPU sessions.

Run the CLI in *this* interpreter's environment instead: ``sleap_nn.cli`` has a
``__main__`` guard, so ``python -m sleap_nn.cli`` is the console script without ``uv``
(and without depending on ``uv`` being installed at all).
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback


def sleap_nn_cli(*args: str) -> list[str]:
    """Command prefix running ``sleap-nn`` on ``sys.executable``: ``[python, -m, sleap_nn.cli, *args]``."""
    return [sys.executable, "-m", "sleap_nn.cli", *args]


def cpu_only_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a CLI subprocess that must stay off the GPU.

    ``--extra torch-cpu`` used to guarantee a CPU-only torch in the child; hiding the
    devices keeps that intent on a GPU box without touching the installed wheel.
    """
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_cli(
    cmd, *, env=None, check=False, capture_output=True, text=True
) -> subprocess.CompletedProcess:
    """Run a ``sleap_nn_cli(...)`` command in THIS process; return what ``subprocess.run`` would.

    A ``sleap-nn`` subprocess spends seconds importing torch and Lightning before it
    does anything (2-6 s on the CI runners, most of every CLI test). Click's
    ``CliRunner`` runs the same command in-process, capturing stdout and stderr
    (sleap_nn's log sink looks ``sys.stdout`` up per message, so logs are captured
    too). ``env`` is applied as overrides for the duration of the call.

    With a GPU visible, ``CUDA_VISIBLE_DEVICES=""`` cannot hide it from a torch that
    has already initialized, so the command runs as a real subprocess instead.
    ``capture_output`` and ``text`` are accepted for drop-in use; output is always
    captured as text.
    """
    prefix = sleap_nn_cli()
    cmd = [str(c) for c in cmd]
    if cmd[: len(prefix)] != prefix:
        raise ValueError(f"run_cli expects a sleap_nn_cli(...) command, got {cmd[:3]}")

    import torch

    if torch.cuda.is_available():
        return subprocess.run(
            cmd, env=cpu_only_env(env), check=check, capture_output=True, text=True
        )

    from click.testing import CliRunner

    from sleap_nn.cli import cli

    overrides = None
    if env is not None:
        overrides = {k: v for k, v in env.items() if os.environ.get(k) != v}
        overrides.update({k: None for k in os.environ if k not in env})
    result = CliRunner().invoke(cli, cmd[len(prefix) :], env=overrides)
    stderr = result.stderr
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        stderr += "".join(traceback.format_exception(*result.exc_info))
    proc = subprocess.CompletedProcess(cmd, result.exit_code, result.stdout, stderr)
    if check:
        proc.check_returncode()
    return proc
