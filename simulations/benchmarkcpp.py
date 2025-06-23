# benchmark.py  – timing + plots for the two RFAOII implementations
# --------------------------------------------------------------------
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import io, json, logging, urllib.parse
from contextlib import redirect_stdout, redirect_stderr
from typing import Any
from pathlib import Path
import random, time, logging, importlib
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


def algorithm1_timed(k: int, m: int, seed: int, impl="py"):
    """Run Algorithm 1 (k = 2) and measure time."""
    if k != 2:
        raise ValueError("Algorithm 1 is defined only for k = 2")
    utils = random_utils(m, seed)
    if impl == "py":
        _ = rfaoii.algorithm1(utils)
    elif impl == "cpp":
        _ = rfaoiic.algorithm1(utils)          # filled in later if available
    else:
        raise ValueError("impl must be 'py' or 'cpp'")


def algorithm2_timed(k: int, m: int, seed: int, impl="py"):
    """Run Algorithm 2 (even k) and measure time."""
    if k % 2:
        raise ValueError("Algorithm 2 requires even k")
    utils = random_utils(m, seed)
    if impl == "py":
        _ = rfaoii.algorithm2(k, utils)
    elif impl == "cpp":
        _ = rfaoiic.algorithm2(k, utils)
    else:
        raise ValueError("impl must be 'py' or 'cpp'")


def run_alg(k: int, m: int, seed: int, algorithm: str, **_):
    """
    Generic timing wrapper used by Experiment.run().
    The *algorithm* flag is one of:
        alg1     – pure-Python   Algorithm 1
        alg1_cpp – C++-accelerated Algorithm 1
        alg2     – pure-Python   Algorithm 2
        alg2_cpp – C++-accelerated Algorithm 2
    """
    t0 = time.perf_counter()

    match algorithm:
        case "alg1":     algorithm1_timed(k, m, seed, "py")
        case "alg1_cpp": algorithm1_timed(k, m, seed, "cpp")
        case "alg2":     algorithm2_timed(k, m, seed, "py")
        case "alg2_cpp": algorithm2_timed(k, m, seed, "cpp")
        case _:
            raise ValueError(f"Unknown algorithm flag: {algorithm}")

    return {"elapsed_ms": round((time.perf_counter() - t0) * 1_000, 2)}


# ───────────────────────── experiment grid ────────────────────────────
RESULTS_DIR = Path("resultscmp");       RESULTS_DIR.mkdir(exist_ok=True)
BACKUP_DIR  = RESULTS_DIR / "backups";  BACKUP_DIR.mkdir(exist_ok=True)

exp = Experiment(RESULTS_DIR, "timing.csv", backup_folder=BACKUP_DIR)

seed1 = random.randint(0, 10000)
seed2 = random.randint(0, 10000)
seed3 = random.randint(0, 10000)
print(f"\nrandom seed1 {seed1}")
print(f"\nrandom seed2 {seed2}")
print(f"\nrandom seed3 {seed3}")






sizes  = [3, 6, 12, 24, 10000]
seeds  = [seed1, seed2, seed3]
rounds = [2, 8, 1000]                   # even values only

exp.clear_previous_results()

# — Pure-Python runs —
exp.run(run_alg, {
    "algorithm": ["alg1"],
    "k": [2],
    "m": sizes,
    "seed": seeds,
})
exp.run(run_alg, {
    "algorithm": ["alg2"],
    "k": rounds,
    "m": sizes,
    "seed": seeds,
})

# ────────────────── C++-accelerated benchmarking ────────────────
cpp_enabled = False
try:
    rfaoiic = importlib.import_module("rfaoiic")
    cpp_enabled = True
    logging.info("rfaoiic imported – benchmarking C++ variants …")

    exp.run(run_alg, {
        "algorithm": ["alg1_cpp"],
        "k": [2],
        "m": sizes,
        "seed": seeds,
    })
    exp.run(run_alg, {
        "algorithm": ["alg2_cpp"],
        "k": rounds,
        "m": sizes,
        "seed": seeds,
    })

except ModuleNotFoundError:
    logging.warning("rfaoiic not found ⇒ skipping C++ benchmarks")

print("✓  Benchmark finished; data in", RESULTS_DIR / "timing.csv")

# ───────────────────────── plotting ──────────────
PLOTS_DIR = RESULTS_DIR / "plots";  PLOTS_DIR.mkdir(exist_ok=True)

# Algorithm 1 – single curve per implementation
single_plot_results(
    RESULTS_DIR / "timing.csv",
    filter       = {"algorithm": ["alg1", "alg1_cpp"]} if cpp_enabled else {"algorithm": "alg1"},
    x_field      = "m",
    y_field      = "elapsed_ms",
    z_field      = "algorithm",          
    mean         = True,
    save_to_file = str(PLOTS_DIR / "alg1_runtime.png"),
)

# Algorithm 2 – 3 panes (k = 2/4/8) with both impls if available
multi_plot_results(
    RESULTS_DIR / "timing.csv",
    filter        = {"algorithm": ["alg2", "alg2_cpp"]} if cpp_enabled else {"algorithm": "alg2"},
    subplot_rows  = 1,
    subplot_cols  = 3,
    x_field       = "m",
    y_field       = "elapsed_ms",
    z_field       = "algorithm",
    subplot_field = "k",
    mean          = True,
    save_to_file  = str(PLOTS_DIR / "alg2_runtime.png"),
)

# scatter – Algorithm 2, fixed m, varying k
FIXED_M = 12
df = pd.read_csv(RESULTS_DIR / "timing.csv")
df = df[(df["m"] == FIXED_M) & (df["algorithm"].isin(["alg2", "alg2_cpp"] if cpp_enabled else ["alg2"]))]

plt.figure(figsize=(6,4))
for (alg, seed), grp in df.groupby(["algorithm", "seed"]):
    style = "o" if alg.endswith("cpp") else "x"
    plt.scatter(grp["k"], grp["elapsed_ms"], marker=style, s=70,
                label=f"{alg}  seed={seed}")
plt.xlabel("number of rounds  k")
plt.ylabel("runtime  [ms]")
plt.title(f"Algorithm 2 — runtime vs k rounds   (m = {FIXED_M})")
plt.grid(True, ls="--", alpha=.4)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"alg2_runtime_m{FIXED_M}_vs_k_scatter.png", dpi=150)
plt.show()

# ──────────────── print speed-up table (nice to have) ─────────────────────
if cpp_enabled:
    pure = df[df["algorithm"] == "alg2"].set_index(["k","seed"])["elapsed_ms"]
    fast = df[df["algorithm"] == "alg2_cpp"].set_index(["k","seed"])["elapsed_ms"]
    joined = pd.concat({"pure": pure, "cpp": fast}, axis=1).dropna()
    joined["speedup"] = joined["pure"] / joined["cpp"]
    print("\nSpeed-ups (Algorithm 2, m = 12):")
    for (k,seed), row in joined.iterrows():
        print(f"  k={k:2}  seed={seed}:   {row['speedup']:.1f}×")
else:
    print("\n(no cpp timings available – install cppyy + rfaoiic.py)")

print("✓  Plots saved in", PLOTS_DIR)
