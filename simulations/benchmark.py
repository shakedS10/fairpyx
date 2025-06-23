# benchmark.py  – timing + plots for the two RFAOII algorithms
# --------------------------------------------------------------------
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import io, json, logging, urllib.parse
from contextlib import redirect_stdout, redirect_stderr
from typing import Any
from pathlib import Path
import random, time, logging
from experiments_csv import Experiment, single_plot_results, multi_plot_results
import fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items as rfaoii
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)

# ───────────────────────── helpers ────────────────────────────────────
def random_utils(m: int, seed: int):
    """Generate a 2 × m utility matrix with values in [−5 … 5]."""
    rnd = random.Random(seed)
    return {
        0: {i: rnd.randint(-5, 5) for i in range(m)},
        1: {i: rnd.randint(-5, 5) for i in range(m)},
    }


def algorithm1_timed(k: int, m: int, seed: int):
    """Run Algorithm 1 (always k = 2) – we only need the time."""
    if k != 2:                     # sanity (k is fixed for Alg-1)
        raise ValueError("Algorithm 1 is defined only for k = 2")
    utils = random_utils(m, seed)
    _ = rfaoii.algorithm1(utils)   # ← actual work


def algorithm2_timed(k: int, m: int, seed: int):
    """Run Algorithm 2 for an *even* k (2, 4, 8…)."""
    if k % 2:                      # Algorithm 2 needs even k
        raise ValueError("Algorithm 2 requires an **even** k")
    utils = random_utils(m, seed)
    _ = rfaoii.algorithm2(k, utils)


def run_alg(k: int, m: int, seed: int, algorithm: str = "alg1", **_):
    """
    timing wrapper used by experiments_csv.Experiment.
    Any extra keyword args sent by Experiment.run are ignored (via **_).
    """
    t0 = time.perf_counter()

    if algorithm == "alg1":
        algorithm1_timed(k, m, seed)
    elif algorithm == "alg2":
        algorithm2_timed(k, m, seed)
    else:
        raise ValueError(f"Unknown algorithm flag: {algorithm}")

    return {"elapsed_ms": round((time.perf_counter() - t0) * 1_000, 2)}


# ───────────────────────── experiment grid ────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
BACKUP_DIR  = RESULTS_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

exp = Experiment(RESULTS_DIR, "timing.csv", backup_folder=BACKUP_DIR)

seed1 = random.randint(0, 10000)
seed2 = random.randint(0, 10000)
seed3 = random.randint(0, 10000)
print(f"\nrandom seed1 {seed1}")
print(f"\nrandom seed2 {seed2}")
print(f"\nrandom seed3 {seed3}")






sizes  = [3, 6, 12, 24, 10000]
seeds  = [seed1, seed2, seed3]
rounds = [2, 4, 8]                   # even values only

exp.clear_previous_results()


# Algorithm 1  (fixed k = 2)
exp.run(run_alg, {
    "algorithm": ["alg1"],
    "k"        : [2],
    "m"        : sizes,
    "seed"     : seeds,
})

# Algorithm 2  (k = 2, 4, 8)
exp.run(run_alg, {
    "algorithm": ["alg2"],
    "k"        : rounds,
    "m"        : sizes,
    "seed"     : seeds,
})

print("✓  Benchmark finished; data in", RESULTS_DIR / "timing.csv")

# ───────────────────────── plotting ───────────────────────────────────
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# single plot – Algorithm 1
single_plot_results(
    RESULTS_DIR / "timing.csv",
    filter       = {"algorithm": "alg1"},
    x_field      = "m",
    y_field      = "elapsed_ms",
    z_field      = "seed",
    mean         = True,
    save_to_file = str(PLOTS_DIR / "alg1_runtime.png"),      # ← cast to str
)

#  3-pane plot – Algorithm 2
multi_plot_results(
    RESULTS_DIR / "timing.csv",
    filter         = {"algorithm": "alg2"},
    subplot_rows   = 1,
    subplot_cols   = 3,
    x_field        = "m",
    y_field        = "elapsed_ms",
    z_field        = "seed",
    subplot_field  = "k",
    mean           = True,
    save_to_file   = str(PLOTS_DIR / "alg2_runtime.png"),    # ← cast to str
)


# scatter – Algorithm 2, fixed m, varying k (even values only)
FIXED_M = 12
df = pd.read_csv(RESULTS_DIR / "timing.csv")
df = df[(df["algorithm"] == "alg2") & (df["m"] == FIXED_M)]

plt.figure(figsize=(6, 4))
for seed, grp in df.groupby("seed"):
    plt.scatter(grp["k"], grp["elapsed_ms"],
                label=f"seed={seed}", s=70)

plt.xlabel("number of rounds  k")
plt.ylabel("runtime  [ms]")
plt.title(f"Algorithm 2 — runtime vs rounds  (m = {FIXED_M})")
plt.grid(True, ls="--", alpha=.4)
plt.legend(title="random seed")
plt.tight_layout()

scatter_path = PLOTS_DIR / f"alg2_runtime_m{FIXED_M}_vs_k_scatter.png"
plt.savefig(scatter_path, dpi=150)
plt.show()

print("✓  Plots saved in", PLOTS_DIR)
