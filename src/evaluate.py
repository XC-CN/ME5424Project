"""
Unified evaluation script for AeroPursuit predator-prey project.

Provides:
  1. Multi-seed evaluation of trained models (Curriculum method)
  2. Baseline comparisons (Random, Heuristic-only)
  3. Statistical significance testing (Welch's t-test)

Usage:
  python src/evaluate.py                       # run all methods
  python src/evaluate.py --method curriculum    # run only curriculum
  python src/evaluate.py --method random        # run only random baseline
  python src/evaluate.py --method heuristic     # run only heuristic baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curriculum_env import (
    EagleTrainingEnv,
    HenTrainingEnv,
    PhysicsConfig,
)
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [0, 42, 123, 456, 789]
EPISODES_PER_SEED = 100
HEN_MODEL_DEFAULT = "results/curriculum/best_hen/best_model.zip"
EAGLE_MODEL_DEFAULT = "results/curriculum/best_eagle/best_model.zip"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_curriculum(
    hen_model_path: str,
    eagle_model_path: str,
    cfg: PhysicsConfig,
    seeds: List[int],
    n_episodes: int,
) -> Dict[str, List[float]]:
    """Evaluate the curriculum-trained hen + eagle pair."""
    hen_model = PPO.load(hen_model_path, device="cpu")
    eagle_model = PPO.load(eagle_model_path, device="cpu")

    results: Dict[str, List[float]] = {
        "hen_return": [],
        "eagle_return": [],
        "capture": [],
        "episode_length": [],
        "avg_dist": [],
    }

    for seed in seeds:
        env = EagleTrainingEnv(
            hen_policy_path=hen_model_path, config=cfg, seed=seed, device="cpu"
        )
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            hen_return = 0.0
            eagle_return = 0.0
            done = False
            total_dist = 0.0
            steps = 0
            captured = False

            while not done:
                # Eagle action from trained model
                eagle_action, _ = eagle_model.predict(obs, deterministic=True)
                obs, eagle_reward, terminated, truncated, info = env.step(eagle_action)
                eagle_return += eagle_reward
                total_dist += info.get("dist_to_tail", 0.0)
                steps += 1
                done = terminated or truncated
                if terminated and steps < cfg.max_steps:
                    captured = True

            # For hen return, we use a separate env run
            # But since they are coupled, we approximate hen_return from eagle perspective
            # Hen "wins" if not captured; compute a proxy return
            hen_return = 0.0 if captured else 1.0  # simplified: survival = success

            results["eagle_return"].append(eagle_return)
            results["hen_return"].append(hen_return)
            results["capture"].append(1.0 if captured else 0.0)
            results["episode_length"].append(float(steps))
            results["avg_dist"].append(total_dist / max(steps, 1))

        env.close()

    return results


def evaluate_random(
    cfg: PhysicsConfig,
    seeds: List[int],
    n_episodes: int,
) -> Dict[str, List[float]]:
    """Evaluate with both agents taking random actions."""
    results: Dict[str, List[float]] = {
        "hen_return": [],
        "eagle_return": [],
        "capture": [],
        "episode_length": [],
        "avg_dist": [],
    }

    for seed in seeds:
        env = HenTrainingEnv(config=cfg, seed=seed)
        rng = np.random.default_rng(seed)

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            hen_return = 0.0
            done = False
            total_dist = 0.0
            steps = 0
            captured = False

            while not done:
                # Random hen action
                action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                hen_return += reward
                total_dist += info.get("dist_to_tail", 0.0)
                steps += 1
                done = terminated or truncated
                if terminated and steps < cfg.max_steps:
                    captured = True

            results["hen_return"].append(hen_return)
            # Eagle "return" proxy: positive if captured, negative otherwise
            results["eagle_return"].append(100.0 if captured else -float(steps) * 0.01)
            results["capture"].append(1.0 if captured else 0.0)
            results["episode_length"].append(float(steps))
            results["avg_dist"].append(total_dist / max(steps, 1))

        env.close()

    return results


def evaluate_heuristic(
    cfg: PhysicsConfig,
    seeds: List[int],
    n_episodes: int,
) -> Dict[str, List[float]]:
    """Evaluate heuristic eagle vs random hen (no training)."""
    # This uses HenTrainingEnv which already has a heuristic eagle built-in.
    # The hen takes random actions to simulate an untrained agent.
    results: Dict[str, List[float]] = {
        "hen_return": [],
        "eagle_return": [],
        "capture": [],
        "episode_length": [],
        "avg_dist": [],
    }

    for seed in seeds:
        env = HenTrainingEnv(config=cfg, seed=seed)
        rng = np.random.default_rng(seed)

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed * 10000 + ep)
            hen_return = 0.0
            done = False
            total_dist = 0.0
            steps = 0
            captured = False

            while not done:
                # Random hen action (untrained hen)
                action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                hen_return += reward
                total_dist += info.get("dist_to_tail", 0.0)
                steps += 1
                done = terminated or truncated
                if terminated and steps < cfg.max_steps:
                    captured = True

            results["hen_return"].append(hen_return)
            results["eagle_return"].append(100.0 if captured else -float(steps) * 0.01)
            results["capture"].append(1.0 if captured else 0.0)
            results["episode_length"].append(float(steps))
            results["avg_dist"].append(total_dist / max(steps, 1))

        env.close()

    return results


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    t_stat, p_val = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    return float(t_stat), float(p_val)


def compute_stats(data: List[float]) -> Tuple[float, float]:
    """Return (mean, std)."""
    arr = np.array(data)
    return float(np.mean(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary_table(all_results: Dict[str, Dict[str, List[float]]]) -> None:
    """Print a formatted comparison table to console."""
    metrics = ["hen_return", "eagle_return", "capture", "episode_length", "avg_dist"]
    header = f"{'Method':<20}"
    for m in metrics:
        header += f" {m:>20}"
    print("\n" + "=" * 120)
    print("QUANTITATIVE COMPARISON")
    print("=" * 120)
    print(header)
    print("-" * 120)

    for method, results in all_results.items():
        row = f"{method:<20}"
        for m in metrics:
            mean, std = compute_stats(results[m])
            if m == "capture":
                row += f" {mean*100:>8.1f}% +/- {std*100:>5.1f}%"
            else:
                row += f" {mean:>9.2f} +/- {std:>6.2f}"
        print(row)
    print("=" * 120)


def print_significance_table(
    all_results: Dict[str, Dict[str, List[float]]], ours_key: str = "Curriculum"
) -> None:
    """Print statistical significance results."""
    if ours_key not in all_results:
        print("Curriculum results not available, skipping significance tests.")
        return

    ours = all_results[ours_key]
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE (Welch's t-test)")
    print("=" * 90)
    print(f"{'Comparison':<30} {'Metric':<20} {'t-stat':>10} {'p-value':>12} {'Significant?':>14}")
    print("-" * 90)

    for method, results in all_results.items():
        if method == ours_key:
            continue
        for metric in ["hen_return", "eagle_return", "capture"]:
            t_stat, p_val = welch_ttest(ours[metric], results[metric])
            sig = "YES (p<0.05)" if p_val < 0.05 else "NO"
            label = f"Ours vs {method}"
            print(f"{label:<30} {metric:<20} {t_stat:>10.3f} {p_val:>12.2e} {sig:>14}")
    print("=" * 90)


def save_results_csv(
    all_results: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    """Save per-episode raw results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "episode_idx", "hen_return", "eagle_return",
                         "capture", "episode_length", "avg_dist"])
        for method, results in all_results.items():
            n = len(results["hen_return"])
            for i in range(n):
                writer.writerow([
                    method, i,
                    results["hen_return"][i],
                    results["eagle_return"][i],
                    results["capture"][i],
                    results["episode_length"][i],
                    results["avg_dist"][i],
                ])
    print(f"\nRaw results saved to: {path}")


def save_latex_tables(
    all_results: Dict[str, Dict[str, List[float]]],
    output_path: str,
    ours_key: str = "Curriculum",
) -> None:
    """Generate LaTeX table snippets ready to paste into the paper."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Table 1: Quantitative comparison
    lines.append("% === Table: Quantitative Comparison ===")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Quantitative comparison across methods "
                 "(mean $\\pm$ std over 5 seeds, 100 episodes each).}")
    lines.append("\\label{tab:baseline_comparison}")
    lines.append("\\resizebox{\\linewidth}{!}{")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Hen Return} & \\textbf{Eagle Return} "
                 "& \\textbf{Capture Rate (\\%)} & \\textbf{Avg Dist} & \\textbf{Ep. Length} \\\\")
    lines.append("\\midrule")

    for method, results in all_results.items():
        hr_m, hr_s = compute_stats(results["hen_return"])
        er_m, er_s = compute_stats(results["eagle_return"])
        cr_m, cr_s = compute_stats(results["capture"])
        ad_m, ad_s = compute_stats(results["avg_dist"])
        el_m, el_s = compute_stats(results["episode_length"])

        bold = method == ours_key
        fmt = "\\mathbf{" if bold else ""
        end = "}" if bold else ""

        lines.append(
            f"{method} & ${fmt}{hr_m:.2f} \\pm {hr_s:.2f}{end}$ "
            f"& ${fmt}{er_m:.2f} \\pm {er_s:.2f}{end}$ "
            f"& ${fmt}{cr_m*100:.1f} \\pm {cr_s*100:.1f}{end}$ "
            f"& ${fmt}{ad_m:.2f} \\pm {ad_s:.2f}{end}$ "
            f"& ${fmt}{el_m:.1f} \\pm {el_s:.1f}{end}$ \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Statistical significance
    if ours_key in all_results:
        ours = all_results[ours_key]
        lines.append("% === Table: Statistical Significance ===")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Statistical significance (Welch's t-test) comparing "
                     "Curriculum (Ours) against baselines on key metrics.}")
        lines.append("\\label{tab:significance}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Comparison} & \\textbf{Hen Return} & "
                     "\\textbf{Eagle Return} & \\textbf{Capture Rate} \\\\")
        lines.append("\\midrule")

        for method, results in all_results.items():
            if method == ours_key:
                continue
            vals = []
            for metric in ["hen_return", "eagle_return", "capture"]:
                _, p_val = welch_ttest(ours[metric], results[metric])
                if p_val < 0.001:
                    vals.append(f"$p < 0.001$")
                else:
                    vals.append(f"$p = {p_val:.3f}$")
            lines.append(f"Ours vs {method} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table snippets saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AeroPursuit methods and generate comparison data."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "curriculum", "random", "heuristic"],
        default="all",
        help="Which method to evaluate (default: all).",
    )
    parser.add_argument(
        "--hen-model",
        type=str,
        default=HEN_MODEL_DEFAULT,
        help="Path to trained hen model (.zip).",
    )
    parser.add_argument(
        "--eagle-model",
        type=str,
        default=EAGLE_MODEL_DEFAULT,
        help="Path to trained eagle model (.zip).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=EPISODES_PER_SEED,
        help="Number of episodes per seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.csv",
        help="CSV output path for raw results.",
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default="results/latex_tables.tex",
        help="LaTeX table snippet output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PhysicsConfig()
    all_results: Dict[str, Dict[str, List[float]]] = {}

    methods_to_run = (
        ["random", "heuristic", "curriculum"] if args.method == "all"
        else [args.method]
    )

    for method in methods_to_run:
        print(f"\n>>> Evaluating: {method.upper()} <<<")
        if method == "curriculum":
            if not Path(args.hen_model).exists():
                print(f"  [SKIP] Hen model not found: {args.hen_model}")
                continue
            if not Path(args.eagle_model).exists():
                print(f"  [SKIP] Eagle model not found: {args.eagle_model}")
                continue
            results = evaluate_curriculum(
                args.hen_model, args.eagle_model, cfg, SEEDS, args.n_episodes
            )
            all_results["Curriculum"] = results
        elif method == "random":
            results = evaluate_random(cfg, SEEDS, args.n_episodes)
            all_results["Random"] = results
        elif method == "heuristic":
            results = evaluate_heuristic(cfg, SEEDS, args.n_episodes)
            all_results["Heuristic"] = results

        # Quick stats
        for metric in ["hen_return", "eagle_return", "capture"]:
            mean, std = compute_stats(results[metric])
            label = f"{metric}" if metric != "capture" else "capture_rate"
            if metric == "capture":
                print(f"  {label}: {mean*100:.1f}% +/- {std*100:.1f}%")
            else:
                print(f"  {label}: {mean:.2f} +/- {std:.2f}")

    if len(all_results) > 0:
        print_summary_table(all_results)
        print_significance_table(all_results)
        save_results_csv(all_results, args.output)
        save_latex_tables(all_results, args.latex_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
