"""
Evaluation pipeline for AeroPursuit.

Design choices:
  1. Compare only task-level metrics: capture rate, average eagle-tail distance,
     and episode length.
  2. Keep the true-random baseline and the heuristic-eagle/random-hen baseline
     separate.
  3. Aggregate raw rollouts into per-evaluation-seed summaries before running
     statistical tests.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curriculum_env import EagleTrainingEnv, HenTrainingEnv, PhysicsConfig
from stable_baselines3 import PPO


DEFAULT_EVAL_SEEDS = [0, 42, 123, 456, 789]
DEFAULT_EPISODES_PER_SEED = 100
DEFAULT_CURRICULUM_HEN = Path("results/curriculum/best_hen/best_model.zip")
DEFAULT_CURRICULUM_EAGLE = Path("results/curriculum/best_eagle/best_model.zip")
METHOD_ORDER = ["True Random", "Heuristic + Random", "Curriculum"]
METRICS = ["capture_rate", "avg_tail_dist", "episode_length"]
SIGNIFICANCE_BASELINES = ["True Random", "Heuristic + Random"]


@dataclass(frozen=True)
class PolicyPair:
    method: str
    run_id: str
    hen_model_path: Path
    eagle_model_path: Path


class RandomVsRandomEnv(HenTrainingEnv):
    """Sanity-check environment where both agents act randomly."""

    def step(self, action: np.ndarray):
        self._apply_action(self.hen, action, self.cfg.hen_max_speed, self.cfg.max_force)
        eagle_action = self.rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        self._apply_action(
            self.eagle,
            eagle_action,
            self.cfg.eagle_max_speed,
            self.cfg.eagle_max_force,
        )

        self.world.Step(self.cfg.dt, 6, 2)
        self._enforce_bounds()
        self._handle_eagle_bounce_physics()
        self.step_count += 1

        reward, done = self._compute_reward_done()
        obs = self._get_obs(role="hen")
        terminated = done and (self.step_count < self.cfg.max_steps)
        truncated = self.step_count >= self.cfg.max_steps
        info = {"dist_to_tail": float((self.eagle.position - self.chicks[-1].position).length)}
        return obs, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AeroPursuit task-level metrics.")
    parser.add_argument(
        "--method",
        choices=["all", "curriculum", "heuristic", "random"],
        default="all",
        help="Subset of methods to evaluate.",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="*",
        type=int,
        default=DEFAULT_EVAL_SEEDS,
        help="Evaluation seeds used to build per-seed summaries.",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=DEFAULT_EPISODES_PER_SEED,
        help="Number of episodes evaluated for each evaluation seed.",
    )
    parser.add_argument(
        "--curriculum-pair",
        action="append",
        metavar="HEN::EAGLE",
        help="Model pair for a curriculum run. Repeat the flag for multiple training runs.",
    )
    parser.add_argument(
        "--episode-output",
        type=str,
        default="results/eval_episode.csv",
        help="CSV path for episode-level records.",
    )
    parser.add_argument(
        "--seed-output",
        type=str,
        default="results/eval_seed_summary.csv",
        help="CSV path for evaluation-seed summaries.",
    )
    parser.add_argument(
        "--run-output",
        type=str,
        default="results/eval_run_summary.csv",
        help="CSV path for independent-run summaries.",
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default="results/latex_tables.tex",
        help="LaTeX table path generated from summaries.",
    )
    return parser.parse_args()


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def parse_pair_specs(
    raw_specs: Sequence[str] | None,
    default_pair: tuple[Path, Path] | None,
    method_name: str,
) -> list[PolicyPair]:
    specs = list(raw_specs or [])
    if not specs and default_pair is not None:
        if default_pair[0].exists() and default_pair[1].exists():
            specs = [f"{default_pair[0].as_posix()}::{default_pair[1].as_posix()}"]

    pairs: list[PolicyPair] = []
    for idx, raw_spec in enumerate(specs):
        if "::" not in raw_spec:
            raise ValueError(f"Expected pair spec in HEN::EAGLE format, got: {raw_spec}")
        hen_str, eagle_str = raw_spec.split("::", maxsplit=1)
        hen_path = Path(hen_str)
        eagle_path = Path(eagle_str)
        if not hen_path.exists():
            raise FileNotFoundError(f"Hen model not found: {hen_path}")
        if not eagle_path.exists():
            raise FileNotFoundError(f"Eagle model not found: {eagle_path}")

        if hen_path.parent == eagle_path.parent and hen_path.parent.name:
            run_id = hen_path.parent.name
        else:
            run_id = f"run_{idx}"

        pairs.append(
            PolicyPair(
                method=method_name,
                run_id=run_id,
                hen_model_path=hen_path,
                eagle_model_path=eagle_path,
            )
        )
    return pairs


def run_policy_episode(env: EagleTrainingEnv, eagle_model: PPO, reset_seed: int) -> dict:
    obs, _ = env.reset(seed=reset_seed)
    total_dist = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        eagle_action, _ = eagle_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(eagle_action)
        total_dist += info.get("dist_to_tail", 0.0)
        steps += 1

    return {
        "capture": 1.0 if terminated else 0.0,
        "episode_length": float(steps),
        "avg_tail_dist": total_dist / max(steps, 1),
    }


def run_random_hen_episode(env: HenTrainingEnv, reset_seed: int) -> dict:
    _, _ = env.reset(seed=reset_seed)
    hen_rng = np.random.default_rng(reset_seed)
    total_dist = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        hen_action = hen_rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        _, _, terminated, truncated, info = env.step(hen_action)
        total_dist += info.get("dist_to_tail", 0.0)
        steps += 1

    return {
        "capture": 1.0 if terminated else 0.0,
        "episode_length": float(steps),
        "avg_tail_dist": total_dist / max(steps, 1),
    }


def evaluate_policy_pair(
    spec: PolicyPair,
    cfg: PhysicsConfig,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    print(f"Evaluating {spec.method} run '{spec.run_id}'...")
    env = EagleTrainingEnv(
        hen_policy_path=str(spec.hen_model_path),
        config=cfg,
        seed=0,
        device="cpu",
    )
    eagle_model = PPO.load(spec.eagle_model_path, device="cpu")

    rows: list[dict] = []
    try:
        for eval_seed in eval_seeds:
            for episode_idx in range(episodes_per_seed):
                reset_seed = eval_seed * 10000 + episode_idx
                metrics = run_policy_episode(env, eagle_model, reset_seed)
                rows.append(
                    {
                        "method": spec.method,
                        "run_id": spec.run_id,
                        "eval_seed": eval_seed,
                        "episode_idx": episode_idx,
                        **metrics,
                    }
                )
    finally:
        env.close()

    return rows


def evaluate_random_baseline(
    cfg: PhysicsConfig,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    print("Evaluating True Random baseline...")
    env = RandomVsRandomEnv(config=cfg, seed=0)
    rows: list[dict] = []
    try:
        for eval_seed in eval_seeds:
            for episode_idx in range(episodes_per_seed):
                reset_seed = eval_seed * 10000 + episode_idx
                metrics = run_random_hen_episode(env, reset_seed)
                rows.append(
                    {
                        "method": "True Random",
                        "run_id": "analytic_baseline",
                        "eval_seed": eval_seed,
                        "episode_idx": episode_idx,
                        **metrics,
                    }
                )
    finally:
        env.close()
    return rows


def evaluate_heuristic_baseline(
    cfg: PhysicsConfig,
    eval_seeds: Sequence[int],
    episodes_per_seed: int,
) -> list[dict]:
    print("Evaluating Heuristic + Random baseline...")
    env = HenTrainingEnv(config=cfg, seed=0)
    rows: list[dict] = []
    try:
        for eval_seed in eval_seeds:
            for episode_idx in range(episodes_per_seed):
                reset_seed = eval_seed * 10000 + episode_idx
                metrics = run_random_hen_episode(env, reset_seed)
                rows.append(
                    {
                        "method": "Heuristic + Random",
                        "run_id": "analytic_baseline",
                        "eval_seed": eval_seed,
                        "episode_idx": episode_idx,
                        **metrics,
                    }
                )
    finally:
        env.close()
    return rows


def aggregate_seed_rows(episode_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in episode_rows:
        key = (row["method"], row["run_id"], int(row["eval_seed"]))
        grouped.setdefault(key, []).append(row)

    seed_rows: list[dict] = []
    for (method, run_id, eval_seed), rows in sorted(grouped.items()):
        captures = [float(row["capture"]) for row in rows]
        episode_lengths = [float(row["episode_length"]) for row in rows]
        avg_tail_dists = [float(row["avg_tail_dist"]) for row in rows]
        seed_rows.append(
            {
                "method": method,
                "run_id": run_id,
                "eval_seed": eval_seed,
                "capture_rate": float(np.mean(captures)),
                "episode_length": float(np.mean(episode_lengths)),
                "avg_tail_dist": float(np.mean(avg_tail_dists)),
                "episode_count": len(rows),
            }
        )
    return seed_rows


def aggregate_run_rows(seed_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in seed_rows:
        key = (row["method"], row["run_id"])
        grouped.setdefault(key, []).append(row)

    run_rows: list[dict] = []
    for (method, run_id), rows in sorted(grouped.items()):
        run_rows.append(
            {
                "method": method,
                "run_id": run_id,
                "capture_rate": float(np.mean([float(row["capture_rate"]) for row in rows])),
                "episode_length": float(np.mean([float(row["episode_length"]) for row in rows])),
                "avg_tail_dist": float(np.mean([float(row["avg_tail_dist"]) for row in rows])),
                "eval_seed_count": len(rows),
            }
        )
    return run_rows


def aggregate_method_seed_rows(seed_rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in seed_rows:
        key = (row["method"], int(row["eval_seed"]))
        grouped.setdefault(key, []).append(row)

    method_seed_rows: list[dict] = []
    for (method, eval_seed), rows in sorted(grouped.items()):
        method_seed_rows.append(
            {
                "method": method,
                "eval_seed": eval_seed,
                "capture_rate": float(np.mean([float(row["capture_rate"]) for row in rows])),
                "episode_length": float(np.mean([float(row["episode_length"]) for row in rows])),
                "avg_tail_dist": float(np.mean([float(row["avg_tail_dist"]) for row in rows])),
                "source_runs": len(rows),
            }
        )
    return method_seed_rows


def summarize_methods(method_seed_rows: Sequence[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in method_seed_rows:
        method = row["method"]
        grouped.setdefault(method, {metric: [] for metric in METRICS})
        for metric in METRICS:
            grouped[method][metric].append(float(row[metric]))

    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for method, metric_values in grouped.items():
        summary[method] = {
            metric: mean_std(values) for metric, values in metric_values.items()
        }
    return summary


def paired_ttest(
    method_seed_rows: Sequence[dict],
    method_a: str,
    method_b: str,
    metric: str,
) -> tuple[float, float] | None:
    values_a = {
        int(row["eval_seed"]): float(row[metric])
        for row in method_seed_rows
        if row["method"] == method_a
    }
    values_b = {
        int(row["eval_seed"]): float(row[metric])
        for row in method_seed_rows
        if row["method"] == method_b
    }
    common_eval_seeds = sorted(set(values_a).intersection(values_b))
    if len(common_eval_seeds) < 2:
        return None

    arr_a = np.asarray([values_a[seed] for seed in common_eval_seeds], dtype=float)
    arr_b = np.asarray([values_b[seed] for seed in common_eval_seeds], dtype=float)
    t_stat, p_value = stats.ttest_rel(arr_a, arr_b)
    return float(t_stat), float(p_value)


def save_csv(rows: Sequence[dict], fieldnames: Sequence[str], path_str: str) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def print_method_summary(method_seed_rows: Sequence[dict]) -> None:
    if not method_seed_rows:
        print("No seed summaries available.")
        return

    summary = summarize_methods(method_seed_rows)
    print("\nMethod-level summary (mean +/- std over evaluation-seed summaries)")
    print("-" * 96)
    print(f"{'Method':<24} {'Capture Rate (%)':>18} {'Avg Tail Dist (m)':>20} {'Episode Length':>18}")
    print("-" * 96)
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        capture_mean, capture_std = summary[method]["capture_rate"]
        dist_mean, dist_std = summary[method]["avg_tail_dist"]
        length_mean, length_std = summary[method]["episode_length"]
        print(
            f"{method:<24} "
            f"{capture_mean * 100:>8.2f} +/- {capture_std * 100:<7.2f} "
            f"{dist_mean:>9.2f} +/- {dist_std:<8.2f} "
            f"{length_mean:>9.2f} +/- {length_std:<8.2f}"
        )
    print("-" * 96)


def print_significance(method_seed_rows: Sequence[dict]) -> None:
    print("\nCurriculum vs baseline significance")
    print("-" * 86)
    print(f"{'Baseline':<22} {'Metric':<18} {'t-stat':>12} {'p-value':>14} {'Status':>14}")
    print("-" * 86)
    for baseline in SIGNIFICANCE_BASELINES:
        for metric in METRICS:
            test_result = paired_ttest(method_seed_rows, "Curriculum", baseline, metric)
            if test_result is None:
                print(f"{baseline:<22} {metric:<18} {'N/A':>12} {'N/A':>14} {'need >=2 seeds':>14}")
                continue
            t_stat, p_value = test_result
            status = "p < 0.05" if p_value < 0.05 else "not significant"
            print(f"{baseline:<22} {metric:<18} {t_stat:>12.3f} {p_value:>14.3e} {status:>14}")
    print("-" * 86)


def save_latex_tables(method_seed_rows: Sequence[dict], output_path: str) -> None:
    summary = summarize_methods(method_seed_rows)
    lines: list[str] = []

    lines.append("% Generated by src/evaluate.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Quantitative comparison across the proposed curriculum method and non-learning baselines. Metrics are reported as mean $\\pm$ std over evaluation-seed summaries.}"
    )
    lines.append("\\label{tab:baseline_comparison}")
    lines.append("\\resizebox{\\linewidth}{!}{")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Method} & \\textbf{Capture Rate (\\%)} $\\downarrow$ & \\textbf{Avg Tail Dist (m)} $\\uparrow$ & \\textbf{Episode Length} $\\uparrow$ \\\\"
    )
    lines.append("\\midrule")
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        capture_mean, capture_std = summary[method]["capture_rate"]
        dist_mean, dist_std = summary[method]["avg_tail_dist"]
        length_mean, length_std = summary[method]["episode_length"]
        bold_open = "\\mathbf{" if method == "Curriculum" else ""
        bold_close = "}" if method == "Curriculum" else ""
        lines.append(
            f"{method} & ${bold_open}{capture_mean * 100:.2f} \\pm {capture_std * 100:.2f}{bold_close}$ & "
            f"${bold_open}{dist_mean:.2f} \\pm {dist_std:.2f}{bold_close}$ & "
            f"${bold_open}{length_mean:.2f} \\pm {length_std:.2f}{bold_close}$ \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Paired t-tests on per-evaluation-seed summaries for the curriculum method against non-learning baselines.}"
    )
    lines.append("\\label{tab:significance}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Baseline} & \\textbf{Metric} & \\textbf{p-value} & \\textbf{Status} \\\\")
    lines.append("\\midrule")
    for baseline in SIGNIFICANCE_BASELINES:
        for metric in METRICS:
            test_result = paired_ttest(method_seed_rows, "Curriculum", baseline, metric)
            if test_result is None:
                lines.append(f"{baseline} & {metric.replace('_', ' ')} & N/A & need at least 2 eval seeds \\\\")
                continue
            _, p_value = test_result
            status = "significant" if p_value < 0.05 else "not significant"
            lines.append(f"{baseline} & {metric.replace('_', ' ')} & ${p_value:.3e}$ & {status} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output}")


def evaluate_methods(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    cfg = PhysicsConfig()
    episode_rows: list[dict] = []
    methods_to_run = [args.method] if args.method != "all" else ["random", "heuristic", "curriculum"]

    curriculum_pairs = parse_pair_specs(
        args.curriculum_pair,
        (DEFAULT_CURRICULUM_HEN, DEFAULT_CURRICULUM_EAGLE),
        "Curriculum",
    )

    for method in methods_to_run:
        if method == "random":
            episode_rows.extend(
                evaluate_random_baseline(cfg, args.eval_seeds, args.episodes_per_seed)
            )
        elif method == "heuristic":
            episode_rows.extend(
                evaluate_heuristic_baseline(cfg, args.eval_seeds, args.episodes_per_seed)
            )
        elif method == "curriculum":
            if not curriculum_pairs:
                print("Skipping Curriculum: no valid model pair available.")
                continue
            for pair in curriculum_pairs:
                episode_rows.extend(
                    evaluate_policy_pair(pair, cfg, args.eval_seeds, args.episodes_per_seed)
                )

    seed_rows = aggregate_seed_rows(episode_rows)
    run_rows = aggregate_run_rows(seed_rows)
    return episode_rows, seed_rows, run_rows


def main() -> None:
    args = parse_args()
    episode_rows, seed_rows, run_rows = evaluate_methods(args)

    if not episode_rows:
        print("No evaluation records were produced.")
        return

    method_seed_rows = aggregate_method_seed_rows(seed_rows)
    print_method_summary(method_seed_rows)
    print_significance(method_seed_rows)

    save_csv(
        episode_rows,
        ["method", "run_id", "eval_seed", "episode_idx", "capture", "episode_length", "avg_tail_dist"],
        args.episode_output,
    )
    save_csv(
        seed_rows,
        ["method", "run_id", "eval_seed", "capture_rate", "avg_tail_dist", "episode_length", "episode_count"],
        args.seed_output,
    )
    save_csv(
        run_rows,
        ["method", "run_id", "capture_rate", "avg_tail_dist", "episode_length", "eval_seed_count"],
        args.run_output,
    )
    save_latex_tables(method_seed_rows, args.latex_output)

    print("\nDone.")


if __name__ == "__main__":
    main()
