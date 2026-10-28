"""Offline tests for rtx_monitor: no network, no credentials, no CLI needed.

    python3 gpu_pool/test_rtx_monitor.py      # or: pytest gpu_pool/test_rtx_monitor.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_pool.router import AccountStatus, KernelRun, PoolAccount, PoolRouter
from gpu_pool.rtx_monitor import (
    DryRunSink,
    MonitorState,
    RunState,
    new_points,
    parse_sync_lines,
    premium_kernels,
    run_once,
)

SYNC_LOG = """
some ordinary training noise
0%|          | 0/500 [00:00<?, ?it/s]
WANDB_SYNC {"run_name": "r1", "wandb_project": "3D Pose Estimation", "wandb_entity": "clifforders", "step": 0, "final": false, "metrics": {"train_loss": 1.0}}
more noise here
WANDB_SYNC {"run_name": "r1", "wandb_project": "3D Pose Estimation", "wandb_entity": "clifforders", "step": 1, "final": false, "metrics": {"train_loss": 0.8}}
WANDB_SYNC_FINAL {"run_name": "r1", "wandb_project": "3D Pose Estimation", "wandb_entity": "clifforders", "step": 1, "final": true, "metrics": {"final_median_rotation_error": 10.0}}
"""


# -- parsing ------------------------------------------------------------------------------
def test_parse_sync_lines_ignores_surrounding_noise():
    payloads = parse_sync_lines(SYNC_LOG)
    assert len(payloads) == 3
    assert payloads[0]["step"] == 0 and payloads[0]["final"] is False
    assert payloads[2]["final"] is True
    assert payloads[2]["metrics"]["final_median_rotation_error"] == 10.0


def test_parse_sync_lines_skips_malformed_json():
    text = 'WANDB_SYNC {"broken": \nWANDB_SYNC {"run_name": "r1", "step": 0, "metrics": {}}'
    payloads = parse_sync_lines(text)
    assert len(payloads) == 1
    assert payloads[0]["step"] == 0


def test_parse_sync_lines_empty_input():
    assert parse_sync_lines("") == []
    assert parse_sync_lines("no markers here at all") == []


# -- incremental sync -----------------------------------------------------------------------
def test_new_points_first_poll_takes_everything():
    payloads = parse_sync_lines(SYNC_LOG)
    picked = new_points(payloads, last_step=None)
    assert [p["step"] for p in picked] == [0, 1, 1]


def test_new_points_skips_already_synced_step():
    payloads = parse_sync_lines(SYNC_LOG)
    # last poll already got the non-final point at step 1; only the final one is new
    picked = new_points(payloads, last_step=1)
    assert len(picked) == 1
    assert picked[0]["final"] is True


def test_new_points_drops_old_steps_but_keeps_current_final():
    payloads = [
        {"step": 0, "final": False}, {"step": 1, "final": False}, {"step": 2, "final": False},
    ]
    picked = new_points(payloads, last_step=2)
    assert picked == [{"step": 2, "final": False}] or picked == []
    # step 2 == last_step and not final -> already synced last time, correctly dropped
    assert picked == []


def test_new_points_dedupes_identical_step_repeats():
    payloads = [{"step": 5, "final": False}, {"step": 5, "final": False}]
    picked = new_points(payloads, last_step=None)
    assert len(picked) == 1


# -- state persistence ----------------------------------------------------------------------
def test_monitor_state_roundtrips_through_disk():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        state = MonitorState(path)
        assert state.get("acct/kernel").last_step is None
        state.set("acct/kernel", RunState(wandb_run_id="abc123", last_step=7, finished=False))
        state.save()

        reloaded = MonitorState(path)
        row = reloaded.get("acct/kernel")
        assert row.wandb_run_id == "abc123"
        assert row.last_step == 7
        assert row.finished is False


def test_monitor_state_missing_file_is_empty():
    state = MonitorState(os.path.join(tempfile.gettempdir(), "definitely-does-not-exist-abcxyz.json"))
    assert state.get("anything").last_step is None


# -- premium_kernels filtering, and that T4/P100 never show up ------------------------------
def _account(label="acct1"):
    return PoolAccount(label=label, token="tok")


def _status(account, kernels):
    st = AccountStatus(account=account, ok=True)
    st.kernels = kernels
    st.kernels_checked = True
    return st


class _FakeRouter(PoolRouter):
    def __init__(self, statuses):
        self._fake_statuses = statuses
        # bypass PoolRouter.__init__'s CLI/account requirements entirely
        self.accounts = [s.account for s in statuses]

    def probe_all(self, **kwargs):
        return self._fake_statuses

    @property
    def statuses(self):
        return self._fake_statuses


def test_premium_kernels_only_returns_rtx_l4_tpu_not_t4():
    acct = _account()
    kernels = [
        KernelRun(ref="a/rtx-run", status="running", machine_shape="NvidiaRtxPro6000"),
        KernelRun(ref="a/t4-run", status="running", machine_shape="NvidiaTeslaT4"),
        KernelRun(ref="a/l4-run", status="queued", machine_shape="NvidiaL4"),
        KernelRun(ref="a/cpu-run", status="running", machine_shape=""),
        KernelRun(ref="a/done-run", status="complete", machine_shape="NvidiaRtxPro6000"),
    ]
    router = _FakeRouter([_status(acct, kernels)])
    picked = premium_kernels(router, ["rtx6000", "l4", "tpu-v3", "tpu-v5e"])
    refs = sorted(k.ref for _, k in picked)
    assert refs == ["a/l4-run", "a/rtx-run"]     # T4, CPU and the finished run are excluded


def test_run_once_dry_run_updates_state_without_wandb():
    acct = _account()
    kernel = KernelRun(ref="a/rtx-run", status="running", machine_shape="NvidiaRtxPro6000")
    router = _FakeRouter([_status(acct, [kernel])])
    calls = []

    def _fake_kernel_logs(account, ref, follow=False, follow_seconds=30.0):
        calls.append(follow)
        return SYNC_LOG
    router.kernel_logs = _fake_kernel_logs   # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        state = MonitorState(os.path.join(tmp, "state.json"))
        synced = run_once(router, state, DryRunSink(), verbose=False)
        assert synced == 1
        row = state.get("a/rtx-run")
        assert row.last_step == 1
        assert row.finished is True
        # a still-RUNNING kernel must be followed, or plain `kernels logs` returns nothing
        # (see kernel_logs' docstring) and this would never sync anything mid-run.
        assert calls == [True]

        # a second pass with the same log: everything already synced, nothing new
        synced_again = run_once(router, state, DryRunSink(), verbose=False)
        assert synced_again == 0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
