# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When the GPU stall watchdog writes a dump, and what the dump holds.

CUDA events, the clock and cuda-gdb are faked, so this runs without a GPU;
the poll is driven directly instead of from the daemon thread.
"""

from types import SimpleNamespace

import vllm.distributed.device_communicators.b12x_roce_all_reduce as roce_module
import vllm.v1.worker.gpu_stall_watchdog as watchdog_module
from vllm.v1.worker.gpu_stall_watchdog import GpuStallWatchdog


class _Event:
    def __init__(self) -> None:
        self.done = False

    def record(self) -> None:
        pass

    def query(self) -> bool:
        return self.done


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _watchdog(monkeypatch, tmp_path, comms=()):
    clock = _Clock()
    events: list[_Event] = []

    def make_event() -> _Event:
        events.append(_Event())
        return events[-1]

    monkeypatch.setattr(watchdog_module.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(watchdog_module.torch.cuda, "Event", make_event)
    monkeypatch.setattr(watchdog_module.time, "monotonic", clock)
    monkeypatch.setattr(
        GpuStallWatchdog, "_cuda_gdb", staticmethod(lambda: "kernel hung_kernel\n")
    )
    monkeypatch.setattr(roce_module, "live_roce_communicators", lambda: list(comms))
    return GpuStallWatchdog(rank=2, stall_seconds=60), clock, events


def _dumps(tmp_path) -> list[str]:
    directory = tmp_path / "gpu_stall"
    if not directory.exists():
        return []
    return [path.read_text() for path in sorted(directory.iterdir())]


def test_finished_steps_never_dump(monkeypatch, tmp_path):
    watchdog, clock, events = _watchdog(monkeypatch, tmp_path)
    watchdog.mark()
    events[0].done = True
    clock.now += 3600
    watchdog._poll()
    assert _dumps(tmp_path) == []


def test_one_dump_per_stall_and_rearmed_after_it_clears(monkeypatch, tmp_path):
    watchdog, clock, events = _watchdog(monkeypatch, tmp_path)
    watchdog.mark()
    clock.now += 59
    watchdog._poll()
    assert _dumps(tmp_path) == []  # under the threshold
    clock.now += 2
    watchdog._poll()
    clock.now += 600
    watchdog._poll()
    dumps = _dumps(tmp_path)
    assert len(dumps) == 1  # the stall is reported once, not once per poll
    assert "has not finished on rank 2" in dumps[0]
    assert "kernel hung_kernel" in dumps[0]
    assert "== Python threads" in dumps[0]
    events[0].done = True
    watchdog._poll()
    watchdog.mark()
    clock.now += 61
    watchdog._poll()
    assert len(_dumps(tmp_path)) == 2


def test_poisoned_roce_dumps_once_with_its_snapshot(monkeypatch, tmp_path):
    snapshot = {"doorbell": 7, "completed": 6, "failed": 1, "error_peer": 0}
    comm = SimpleNamespace(
        name="ranks 0-1-2-3", poisoned=True, snapshot=lambda: snapshot
    )
    watchdog, _, _ = _watchdog(monkeypatch, tmp_path, comms=[comm])
    watchdog._poll()
    watchdog._poll()
    dumps = _dumps(tmp_path)
    assert len(dumps) == 1
    assert "RoCEnante on rank 2 timed out" in dumps[0]
    assert f"RoCEnante ranks 0-1-2-3: {snapshot}" in dumps[0]
