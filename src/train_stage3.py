import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from curriculum_env import (
    EagleTrainingEnv,
    HenTrainingEnv,
    HenVsModelEnv,
    PhysicsConfig,
    resolve_model_path,
)

PPO_BATCH_SIZE = 512
PPO_N_STEPS = 256
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.995


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: alternating co-training for hen and eagle")
    parser.add_argument(
        "--hen-model",
        type=str,
        default="results/curriculum/best_hen",
        help="Path to a pre-trained hen model. Ignored when --init-from-scratch is set.",
    )
    parser.add_argument(
        "--eagle-model",
        type=str,
        default="results/curriculum/best_eagle",
        help="Path to a pre-trained eagle model. Ignored when --init-from-scratch is set.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/curriculum/stage3",
        help="Directory used to store alternating co-training checkpoints.",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of alternating rounds.")
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=50000,
        help="Timesteps trained for each agent per round.",
    )
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu/cuda/auto).")
    parser.add_argument(
        "--init-from-scratch",
        action="store_true",
        help="Start both PPO policies from fresh random initialization instead of warm-starting from stage 1/2 checkpoints.",
    )
    return parser.parse_args()


def build_fresh_model(env, seed: int, device: str, tensorboard_log: str | None) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        batch_size=PPO_BATCH_SIZE,
        n_steps=PPO_N_STEPS,
        learning_rate=PPO_LEARNING_RATE,
        gamma=PPO_GAMMA,
        tensorboard_log=tensorboard_log,
        device=device,
    )


def initialize_temp_models(
    args: argparse.Namespace,
    cfg: PhysicsConfig,
    temp_hen_path: Path,
    temp_eagle_path: Path,
) -> None:
    if args.init_from_scratch:
        print("Initializing alternating co-training from scratch.")
        hen_bootstrap_env = HenTrainingEnv(config=cfg, seed=args.seed)
        eagle_bootstrap_env = HenTrainingEnv(config=cfg, seed=args.seed + 1000)
        try:
            fresh_hen_model = build_fresh_model(hen_bootstrap_env, args.seed, args.device, None)
            fresh_hen_model.save(temp_hen_path)

            fresh_eagle_model = build_fresh_model(
                eagle_bootstrap_env, args.seed + 1000, args.device, None
            )
            fresh_eagle_model.save(temp_eagle_path)
        finally:
            hen_bootstrap_env.close()
            eagle_bootstrap_env.close()
        return

    hen_path = resolve_model_path(args.hen_model, "best_model")
    eagle_path = resolve_model_path(args.eagle_model, "best_model")

    if not hen_path.exists():
        raise FileNotFoundError(f"Hen model not found at {hen_path}")
    if not eagle_path.exists():
        raise FileNotFoundError(f"Eagle model not found at {eagle_path}")

    print("Warm-starting alternating co-training from pre-trained checkpoints.")
    print(f"Hen init: {hen_path}")
    print(f"Eagle init: {eagle_path}")

    hen_model = PPO.load(hen_path, device=args.device)
    hen_model.save(temp_hen_path)

    eagle_model = PPO.load(eagle_path, device=args.device)
    eagle_model.save(temp_eagle_path)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Stage 3 alternating co-training...")
    print(f"Rounds: {args.rounds}, Steps/Round: {args.steps_per_round}")
    print(f"Budget per agent: {args.rounds * args.steps_per_round} timesteps")

    temp_hen_path = save_dir / "temp_hen.zip"
    temp_eagle_path = save_dir / "temp_eagle.zip"
    tensorboard_log = str(save_dir / "tb")

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    cfg = PhysicsConfig()

    initialize_temp_models(args, cfg, temp_hen_path, temp_eagle_path)

    print("Creating hen alternating-training environment...")
    hen_env = make_vec_env(
        HenVsModelEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={
            "config": cfg,
            "eagle_policy_path": str(temp_eagle_path),
            "device": "cpu",
        },
    )

    print("Creating eagle alternating-training environment...")
    eagle_env = make_vec_env(
        EagleTrainingEnv,
        n_envs=args.n_envs,
        seed=args.seed + 1000,
        vec_env_cls=vec_env_cls,
        env_kwargs={
            "config": cfg,
            "hen_policy_path": str(temp_hen_path),
            "device": "cpu",
        },
    )

    print("Loading policies into alternating-training environments...")
    hen_model = PPO.load(
        temp_hen_path,
        env=hen_env,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )
    eagle_model = PPO.load(
        temp_eagle_path,
        env=eagle_env,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )

    for round_idx in range(1, args.rounds + 1):
        print(f"\n=== Round {round_idx}/{args.rounds} ===")

        print("Training hen against the latest eagle checkpoint...")
        if round_idx > 1:
            hen_env.env_method("load_opponent", str(temp_eagle_path))

        hen_model.learn(total_timesteps=args.steps_per_round, reset_num_timesteps=False)
        hen_model.save(temp_hen_path)
        hen_model.save(save_dir / f"hen_round_{round_idx}")
        print(f"Hen round {round_idx} saved.")

        print("Training eagle against the latest hen checkpoint...")
        eagle_env.env_method("load_opponent", str(temp_hen_path))

        eagle_model.learn(total_timesteps=args.steps_per_round, reset_num_timesteps=False)
        eagle_model.save(temp_eagle_path)
        eagle_model.save(save_dir / f"eagle_round_{round_idx}")
        print(f"Eagle round {round_idx} saved.")

    hen_model.save(save_dir / "hen_stage_3_final")
    eagle_model.save(save_dir / "eagle_stage_3_final")

    print("\nAlternating co-training complete.")
    hen_env.close()
    eagle_env.close()


if __name__ == "__main__":
    main()
