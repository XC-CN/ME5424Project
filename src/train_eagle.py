import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from curriculum_env import EagleTrainingEnv, PhysicsConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: train eagle against frozen hen policy.")
    parser.add_argument("--hen-model", type=str, default="results/curriculum/hen_stage_1.zip", help="Path to trained hen policy.")
    parser.add_argument("--total-steps", type=int, default=300_000, help="Total timesteps for PPO.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency in steps.")
    parser.add_argument("--save-dir", type=str, default="results/curriculum", help="Directory to store models/logs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhysicsConfig()
    env = EagleTrainingEnv(hen_policy_path=args.hen_model, config=cfg, seed=args.seed)
    eval_env = EagleTrainingEnv(hen_policy_path=args.hen_model, config=cfg, seed=args.seed + 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
        batch_size=256,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.995,
    )
    model.learn(total_timesteps=args.total_steps, callback=eval_cb)
    model.save(save_dir / "eagle_stage_1")


if __name__ == "__main__":
    main()

