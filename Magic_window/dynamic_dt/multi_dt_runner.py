"""Launch export / train / rollout across dt values."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from dynamic_dt.device_plan import ALL_DT_DAYS, device_id_for_dt, format_dt_device_plan
from dynamic_dt.dt_config import dt_hours
from dynamic_dt.export_runner import export_one_window
from dynamic_dt.window_catalog import primary_export_component
from dynamic_dt.window_paths import test_prefix, train_prefix, val_prefix
from dynamic_dt.rollout_runner import rollout_one_dt
from dynamic_dt.train_runner import train_one_dt

# Export is CPU-only — sequential is the default (one dt at a time, high n_jobs).
# Train / rollout use GPUs — parallel across dt is recommended.


def _resolve_dt_list(dt_days_list: Sequence[int] | None) -> list[int]:
    if dt_days_list is None:
        return list(ALL_DT_DAYS)
    return list(dt_days_list)


def is_dt_export_complete(
    dt_dir: Path,
    export_key: str,
    dt_days: int,
    *,
    component: str | None = None,
) -> bool:
    """True when train, val, and test exports all report status ``finished``."""
    component = component or primary_export_component(export_key)
    hours = dt_hours(dt_days)
    dt_dir = Path(dt_dir).resolve()
    for prefix_fn in (train_prefix, val_prefix, test_prefix):
        prefix = prefix_fn(dt_dir, export_key, component=component, dt_hours=hours)
        state_path = Path(f"{prefix}_export_state.json")
        if not state_path.is_file():
            return False
        state = json.loads(state_path.read_text())
        if state.get("status") != "finished":
            return False
    return True


def wait_for_export_prefixes(
    prefixes: Sequence[str],
    *,
    poll_sec: float = 30.0,
) -> None:
    """Block until every export ``output_prefix`` reports status ``finished``."""
    pending = {str(p) for p in prefixes}
    while pending:
        finished: set[str] = set()
        for prefix in pending:
            state_path = Path(f"{prefix}_export_state.json")
            if not state_path.is_file():
                continue
            state = json.loads(state_path.read_text())
            if state.get("status") == "finished":
                finished.add(prefix)
        pending -= finished
        if pending:
            print(f"  … waiting on {len(pending)} export(s)", flush=True)
            time.sleep(poll_sec)
    print(f"  ✓ finished {len(prefixes)} export(s)", flush=True)


def launch_export_sequential(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    zarr_path: str,
    log_dir: Path,
    dt_days_list: Sequence[int] | None = None,
    n_jobs_per_export: int = 24,
    detached: bool = True,
    wait_if_detached: bool = True,
    skip_finished: bool = False,
    **export_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Export one dt at a time (CPU-friendly).

    Each dt runs train → val → test, then optionally waits before the next dt.
    GPUs are not used; ``n_jobs_per_export`` controls CPU parallelism.
    """
    days = _resolve_dt_list(dt_days_list)
    print(f"Export plan — sequential, one dt at a time ({len(days)} dt values):")
    for dt in days:
        print(f"  dt={dt}d")

    jobs: list[dict[str, Any]] = []
    for dt_days in days:
        if skip_finished and is_dt_export_complete(dt_dir, export_key, dt_days):
            print(f"\n✓ dt={dt_days}d already complete — skip")
            continue
        print(f"\n{'='*60}\nExport dt={dt_days}d (CPU, n_jobs={n_jobs_per_export})\n{'='*60}")
        summary = export_one_window(
            dt_dir=dt_dir,
            root=root,
            export_key=export_key,
            dt_days=dt_days,
            zarr_path=zarr_path,
            log_dir=log_dir,
            n_jobs=n_jobs_per_export,
            detached=detached,
            device_slot=None,
            **export_kwargs,
        )
        jobs.append(summary)
        if detached and wait_if_detached and summary.get("output_prefixes"):
            wait_for_export_prefixes(summary["output_prefixes"])
    return jobs


def launch_export_parallel(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    zarr_path: str,
    log_dir: Path,
    dt_days_list: Sequence[int] | None = None,
    n_jobs_per_export: int = 8,
    detached: bool = True,
    **export_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Start all dt exports at once (not recommended — CPU/S3 contention).

    Kept for advanced use; prefer ``launch_export_sequential``.
    """
    days = _resolve_dt_list(dt_days_list)
    print("Export plan — parallel (all dt at once; CPU-heavy):")
    print(format_dt_device_plan(days))

    jobs: list[dict[str, Any]] = []
    for dt_days in days:
        print(f"\n--- export dt={dt_days}d ---")
        jobs.append(
            export_one_window(
                dt_dir=dt_dir,
                root=root,
                export_key=export_key,
                dt_days=dt_days,
                zarr_path=zarr_path,
                log_dir=log_dir,
                n_jobs=n_jobs_per_export,
                detached=detached,
                device_slot=device_id_for_dt(dt_days),
                **export_kwargs,
            )
        )
    return jobs


# Back-compat alias
launch_export_all = launch_export_parallel


def launch_train_all(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    log_dir: Path,
    dt_days_list: Sequence[int] | None = None,
    detached: bool = True,
    resume: bool = False,
    **train_kwargs: Any,
) -> list[dict[str, Any]]:
    """Start one training job per dt on cuda:0 … cuda:7."""
    days = _resolve_dt_list(dt_days_list)
    print("Train plan:")
    print(format_dt_device_plan(days))

    jobs: list[dict[str, Any]] = []
    for dt_days in days:
        print(f"\n--- train dt={dt_days}d ---")
        jobs.append(
            train_one_dt(
                dt_dir=dt_dir,
                root=root,
                export_key=export_key,
                dt_days=dt_days,
                log_dir=log_dir,
                detached=detached,
                resume=resume,
                **train_kwargs,
            )
        )
    return jobs


def launch_rollout_all(
    *,
    dt_dir: Path,
    root: Path,
    export_key: str,
    log_dir: Path,
    dt_days_list: Sequence[int] | None = None,
    plot_indices: Sequence[int] | None = None,
    detached: bool = True,
    **rollout_kwargs: Any,
) -> list[dict[str, Any]]:
    """Start one rollout job per dt on cuda:0 … cuda:7."""
    days = _resolve_dt_list(dt_days_list)
    print("Rollout plan:")
    print(format_dt_device_plan(days))

    jobs: list[dict[str, Any]] = []
    for dt_days in days:
        print(f"\n--- rollout dt={dt_days}d ---")
        jobs.append(
            rollout_one_dt(
                dt_dir=dt_dir,
                root=root,
                export_key=export_key,
                dt_days=dt_days,
                log_dir=log_dir,
                plot_indices=plot_indices,
                detached=detached,
                **rollout_kwargs,
            )
        )
    return jobs


def print_job_summary(jobs: list[dict[str, Any]], *, label: str) -> None:
    print(f"\n{label} — {len(jobs)} job(s):")
    for job in jobs:
        dt = job.get("dt_days", "?")
        dev = job.get("device_id", job.get("device_slot", "cpu"))
        line = f"  dt={dt}d"
        if dev is not None and dev != "cpu":
            line += f" gpu={dev}"
        if "pid" in job:
            line += f" PID {job['pid']}"
        elif job.get("pids"):
            line += f" PIDs {job['pids']}"
        if "log_path" in job:
            line += f" | tail -f {job['log_path']}"
        elif job.get("log_files"):
            line += f" | {len(job['log_files'])} log(s)"
        print(line)
