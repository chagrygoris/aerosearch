"""Fair sharing of a pool of shared accelerator accounts.

Ask which account can run a job right now, then send the job there:

    from gpu_pool import PoolRouter, PoolLauncher, git_run_spec

    router = PoolRouter.from_dir("~/pool_tokens")
    router.probe_all()
    print(router.report())

The current backend is the official Kaggle CLI; see router.py for the commands used.
"""

from .launcher import ACCELERATORS, NotebookSpec, PoolLauncher, git_run_spec
from .router import (
    AccountStatus,
    CliError,
    PoolAccount,
    PoolRouter,
    Quota,
    filter_progress,
    shape_label,
)

__all__ = [
    "ACCELERATORS",
    "AccountStatus",
    "CliError",
    "NotebookSpec",
    "PoolAccount",
    "PoolLauncher",
    "PoolRouter",
    "Quota",
    "filter_progress",
    "git_run_spec",
    "shape_label",
]
