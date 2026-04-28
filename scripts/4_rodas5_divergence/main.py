"""Warp-divergence benchmark: Rodas5 on Robertson, varying batch_size.

This benchmark measures wall time and wasted lane work when Robertson
trajectories with increasingly varied initial conditions share a Rodas5
while-loop batch.

Initial condition design
------------------------
The Robertson system has three species (y1, y2, y3) with conservation law
y1 + y2 + y3 = 1 and rate constants k1=0.04, k2=1e4, k3 (varied).  The
dominant source of stiffness is the fast eigenvalue ~ -k3*y2: when y2 is
large the solver must take tiny steps; when y2 = 0 the fast mode is absent
and large steps are possible.

The hardest possible IC is therefore y = [0, 1, 0] with k3 = k3_max = 3e9:
all mass sits in the fast species at the highest rate constant, so the solver
faces eigenvalue magnitude ~3e9 from t=0.  This is set as _HARD_Y0 and used
as the IC for the identical scenario, with k3=k3_max for every trajectory.

The ic_large scenario generates ICs on the y2=0 edge of the simplex:
y2=0, y1 ~ U(0, 1), y3 = 1 - y1. Starting with y2=0 means the fast
eigenvalue is zero at t=0 regardless of k3, so every ic_large trajectory is
analytically easier than the identical base without needing to run the solver
to verify.  k3 is drawn log-uniformly over [3e4, 3e9], creating spread in how
quickly y2 (and therefore stiffness) builds up during the integration.

Usage:
    uv run python scripts/4_rodas5_divergence/main.py
"""

import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.systems.python import robertson
from solvers.rodas5 import solve as rodas5_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 100_000
_T_SPAN = robertson.TIMES
_N_RUNS = 1
_BATCH_SIZES = (10_000, 25_000, 50_000, 100_000)
_SOLVER_KWARGS = {
    "first_step": 1e-4,
    "rtol": 1e-6,
    "atol": 1e-8,
    "lu_precision": "fp64",
}

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"

_K3_MAX = 3e9
_rng_params = np.random.default_rng(0)
_k3_varied = np.exp(_rng_params.uniform(np.log(3e4), np.log(_K3_MAX), size=_N_TRAJ))

_PARAMS_IDENTICAL = jnp.array(
    np.column_stack([np.full(_N_TRAJ, 0.04), np.full(_N_TRAJ, 1e4), np.full(_N_TRAJ, _K3_MAX)]),
    dtype=jnp.float64,
)
_PARAMS_IC_LARGE = jnp.array(
    np.column_stack([np.full(_N_TRAJ, 0.04), np.full(_N_TRAJ, 1e4), _k3_varied]),
    dtype=jnp.float64,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    color: str
    params: jnp.ndarray


@dataclass(frozen=True)
class Grouping:
    key: str
    label: str
    linestyle: str
    marker: str


_HARD_Y0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

_SCENARIOS = (
    Scenario("identical", "identical", "#2b7be0", _PARAMS_IDENTICAL),
    Scenario("ic_large", "large y0", "#e02b2b", _PARAMS_IC_LARGE),
)

_GROUPINGS = (
    Grouping("random", "random", "-", "o"),
    Grouping("sorted", "sorted", "--", "s"),
)

_CSV_FIELDS = (
    "gpu",
    "case_key",
    "batch_size",
    "solve_time_ms",
    "total_lane_iterations",
    "wasted_lane_iterations",
    "min_batch_loop_iterations",
    "max_batch_loop_iterations",
    "wasted_lane_iteration_ratio",
)


def get_gpu_name() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        if out:
            return out
    except Exception:
        pass
    try:
        devices = jax.devices("gpu")
        if devices:
            return devices[0].device_kind
    except Exception:
        pass
    return "unknown_GPU"


def gpu_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def make_initial_conditions(
    scenario: Scenario, size: int, seed: int = 42
) -> np.ndarray:
    if scenario.key == "identical":
        return np.broadcast_to(_HARD_Y0, (size, robertson.N_VARS)).copy()
    rng = np.random.default_rng(seed)
    y1 = rng.uniform(0.0, 1.0, size)
    y0 = np.column_stack([y1, np.zeros(size), 1.0 - y1])
    return y0


def summarize_stats(stats: dict) -> dict[str, float | int]:
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    batch_loop_iterations = np.asarray(jax.device_get(stats["batch_loop_iterations"]))
    valid_lanes = np.asarray(jax.device_get(stats["valid_lanes"]))
    active_lane_iterations = int(np.sum(accepted_steps + rejected_steps))
    total_lane_iterations = int(np.sum(batch_loop_iterations * valid_lanes))
    wasted_lane_iterations = total_lane_iterations - active_lane_iterations
    wasted_lane_iteration_ratio = (
        wasted_lane_iterations / total_lane_iterations
        if total_lane_iterations > 0
        else 0.0
    )
    return {
        "total_lane_iterations": total_lane_iterations,
        "wasted_lane_iterations": int(wasted_lane_iterations),
        "min_batch_loop_iterations": int(np.min(batch_loop_iterations)),
        "max_batch_loop_iterations": int(np.max(batch_loop_iterations)),
        "wasted_lane_iteration_ratio": float(wasted_lane_iteration_ratio),
    }


def format_stats(row: dict) -> str:
    return (
        f"{row['solve_time_ms']:.1f} ms, "
        f"lanes={row['total_lane_iterations']}, "
        f"wasted_lanes={row['wasted_lane_iterations']}, "
        f"min_batch_steps={row['min_batch_loop_iterations']}, "
        f"max_batch_steps={row['max_batch_loop_iterations']}, "
        f"wasted={row['wasted_lane_iteration_ratio']:.3f}"
    )


def solve_with_stats(y0: np.ndarray, params: jnp.ndarray, batch_size: int | None):
    return rodas5_solve(
        robertson.ode_fn,
        y0=jnp.asarray(y0, dtype=jnp.float64),
        t_span=_T_SPAN,
        params=params,
        batch_size=batch_size,
        return_stats=True,
        **_SOLVER_KWARGS,
    )


def time_solve_with_stats(
    y0: np.ndarray, params: jnp.ndarray, batch_size: int | None
) -> tuple[float, dict]:
    result = solve_with_stats(y0, params, batch_size)
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(_N_RUNS):
        result = solve_with_stats(y0, params, batch_size)
        jax.block_until_ready(result)
    ms = (time.perf_counter() - t0) / _N_RUNS * 1000
    _, stats = result
    return ms, summarize_stats(stats)


def active_attempt_order(y0: np.ndarray, params: jnp.ndarray) -> np.ndarray:
    _, stats = solve_with_stats(y0, params, batch_size=None)
    jax.block_until_ready(stats)
    accepted_steps = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected_steps = np.asarray(jax.device_get(stats["rejected_steps"]))
    attempts = accepted_steps + rejected_steps
    return np.argsort(attempts, kind="stable")


def order_initial_conditions(
    y0: np.ndarray, scenario: Scenario, grouping: Grouping
) -> np.ndarray:
    if grouping.key == "sorted":
        return y0[active_attempt_order(y0, scenario.params)]
    seed = sum(ord(c) for c in f"{scenario.key}:{grouping.key}")
    rng = np.random.default_rng(seed)
    return y0[rng.permutation(y0.shape[0])]


def is_complete_row(value) -> bool:
    return isinstance(value, dict) and all(field in value for field in _CSV_FIELDS)


def iter_cases():
    for scenario in _SCENARIOS:
        for grouping in _GROUPINGS:
            if scenario.key == "identical" and grouping.key == "sorted":
                continue
            yield scenario, grouping


def collect_row(
    gpu_name: str,
    scenario: Scenario,
    grouping: Grouping,
    y0: np.ndarray,
    batch_size: int,
) -> dict | None:
    print(
        f"  {scenario.label:<12} {grouping.label:<6} batch_size={batch_size:>6} ...",
        end=" ",
        flush=True,
    )
    try:
        ms, stats = time_solve_with_stats(y0, scenario.params, batch_size)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    row = {
        "gpu": gpu_name,
        "case_key": f"{scenario.key}_{grouping.key}",
        "batch_size": int(batch_size),
        "solve_time_ms": float(ms),
        **stats,
    }
    print(format_stats(row), flush=True)
    return row


def run_benchmarks(gpu_name: str, cache: dict) -> list[dict]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows: list[dict] = []
    for scenario, grouping in iter_cases():
        base_y0 = make_initial_conditions(scenario, _N_TRAJ)
        case_key = f"{scenario.key}_{grouping.key}"
        case_cache = gpu_cache.setdefault(case_key, {})
        y0 = order_initial_conditions(base_y0, scenario, grouping)
        print(f"{scenario.label} / {grouping.label}:")
        for bs in _BATCH_SIZES:
            bs_key = str(int(bs))
            cached = case_cache.get(bs_key)
            if is_complete_row(cached):
                row = cached
                print(
                    f"  {scenario.label:<12} {grouping.label:<6} "
                    f"batch_size={bs:>6} ... (cached) "
                    f"{format_stats(row)}"
                )
            else:
                row = collect_row(gpu_name, scenario, grouping, y0, int(bs))
                case_cache[bs_key] = row
                save_cache(cache)
            if row is not None:
                rows.append(row)
        print()
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def rows_for_case(
    rows: list[dict], case_key: str
) -> tuple[list[int], list[float], list[float]]:
    case_rows = sorted(
        (row for row in rows if row["case_key"] == case_key),
        key=lambda row: row["batch_size"],
    )
    return (
        [row["batch_size"] for row in case_rows],
        [row["solve_time_ms"] for row in case_rows],
        [row["wasted_lane_iteration_ratio"] for row in case_rows],
    )


def plot(rows: list[dict], gpu_name: str, output_path: Path) -> None:
    fig, ax_time = plt.subplots(figsize=(11, 6))
    ax_waste = ax_time.twinx()

    time_handles = []
    waste_handles = []
    for scenario, grouping in iter_cases():
        case_key = f"{scenario.key}_{grouping.key}"
        xs, times_ms, wasted = rows_for_case(rows, case_key)
        if not xs:
            continue
        label = f"{scenario.label} / {grouping.label}"
        (time_line,) = ax_time.plot(
            xs,
            times_ms,
            color=scenario.color,
            linestyle=grouping.linestyle,
            marker=grouping.marker,
            label=f"time: {label}",
        )
        (waste_line,) = ax_waste.plot(
            xs,
            wasted,
            color=scenario.color,
            linestyle=":",
            marker=grouping.marker,
            alpha=0.85,
            label=f"waste: {label}",
        )
        time_handles.append(time_line)
        waste_handles.append(waste_line)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Batch size")
    ax_time.set_ylabel("Solve time (ms)")
    ax_waste.set_ylabel("Wasted lane-iteration ratio")
    ax_waste.set_ylim(0.0, 1.0)
    ax_time.set_title(f"Rodas5 divergence - Robertson y0 variation - {gpu_name}")
    ax_time.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_time.set_xticks(_BATCH_SIZES)
    ax_time.set_xticklabels([str(bs) for bs in _BATCH_SIZES], rotation=45, ha="right")
    handles = time_handles + waste_handles
    labels = [handle.get_label() for handle in handles]
    ax_time.legend(handles, labels, loc="upper left", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    slug = gpu_slug(gpu_name)
    print(f"GPU: {gpu_name}\n")

    cache = load_cache()
    rows = run_benchmarks(gpu_name, cache)

    csv_path = _SCRIPT_DIR / f"results-{slug}.csv"
    save_csv(rows, csv_path)

    plot_path = _SCRIPT_DIR / f"plot-{slug}.png"
    plot(rows, gpu_name, plot_path)


if __name__ == "__main__":
    main()
