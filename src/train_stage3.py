import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from curriculum_env import HenVsModelEnv, EagleTrainingEnv, PhysicsConfig, resolve_model_path

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Co-training (Fine-tuning) Hen and Eagle")
    parser.add_argument("--hen-model", type=str, default="results/curriculum/best_hen", help="Path to pre-trained hen model")
    parser.add_argument("--eagle-model", type=str, default="results/curriculum/best_eagle", help="Path to pre-trained eagle model")
    parser.add_argument("--save-dir", type=str, default="results/curriculum/stage3", help="Directory to save results")
    parser.add_argument("--rounds", type=int, default=10, help="Number of co-training rounds")
    parser.add_argument("--steps-per-round", type=int, default=50000, help="Timesteps per agent per round")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu/cuda)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve initial model paths
    hen_path = resolve_model_path(args.hen_model, "best_model")
    eagle_path = resolve_model_path(args.eagle_model, "best_model")

    if not hen_path.exists():
        print(f"Error: Hen model not found at {hen_path}")
        return
    if not eagle_path.exists():
        print(f"Error: Eagle model not found at {eagle_path}")
        return

    print(f"Starting Stage 3 Co-training...")
    print(f"Hen init: {hen_path}")
    print(f"Eagle init: {eagle_path}")
    print(f"Rounds: {args.rounds}, Steps/Round: {args.steps_per_round}")

    # Create temporary paths for model exchange
    temp_hen_path = save_dir / "temp_hen.zip"
    temp_eagle_path = save_dir / "temp_eagle.zip"

    # Initial copy to temp paths
    # We load and save to ensure format consistency and create the files
    print("Initializing temp models...")
    hen_model = PPO.load(hen_path, device=args.device)
    hen_model.save(temp_hen_path)
    
    eagle_model = PPO.load(eagle_path, device=args.device)
    eagle_model.save(temp_eagle_path)

    # Create Environments
    # We use SubprocVecEnv for parallelism
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    
    cfg = PhysicsConfig() # Use default config

    print("Creating Hen Training Environment (vs Eagle Model)...")
    hen_env = make_vec_env(
        HenVsModelEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={
            "config": cfg,
            "eagle_policy_path": str(temp_eagle_path), # Initially load the temp eagle
            "device": "cpu" # Opponent inference on CPU usually
        }
    )
    
    print("Creating Eagle Training Environment (vs Hen Model)...")
    eagle_env = make_vec_env(
        EagleTrainingEnv,
        n_envs=args.n_envs,
        seed=args.seed + 1000, # Different seed
        vec_env_cls=vec_env_cls,
        env_kwargs={
            "config": cfg,
            "hen_policy_path": str(temp_hen_path), # Initially load the temp hen
            "device": "cpu"
        }
    )

    # Reload models with the new environments attached to handle n_envs mismatch
    # Also disable tensorboard logging for this stage as requested
    print("Loading models into environments...")
    hen_model = PPO.load(temp_hen_path, env=hen_env, device=args.device, tensorboard_log=None)
    eagle_model = PPO.load(temp_eagle_path, env=eagle_env, device=args.device, tensorboard_log=None)

    # Co-training Loop
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} ===")
        
        # --- Train Hen ---
        print(f"Training Hen (vs Eagle)...")
        # Update opponent in hen_env to latest eagle
        # Note: The eagle model on disk (temp_eagle_path) was updated at the end of last round
        if r > 1:
             # Reload the eagle opponent in all parallel envs
             hen_env.env_method("load_opponent", str(temp_eagle_path))
        
        hen_model.learn(total_timesteps=args.steps_per_round, reset_num_timesteps=False)
        hen_model.save(temp_hen_path)
        hen_model.save(save_dir / f"hen_round_{r}")
        print(f"Hen round {r} finished. Saved to {save_dir / f'hen_round_{r}'}")

        # --- Train Eagle ---
        print(f"Training Eagle (vs Hen)...")
        # Update opponent in eagle_env to latest hen (just saved)
        eagle_env.env_method("load_opponent", str(temp_hen_path))
        
        eagle_model.learn(total_timesteps=args.steps_per_round, reset_num_timesteps=False)
        eagle_model.save(temp_eagle_path)
        eagle_model.save(save_dir / f"eagle_round_{r}")
        print(f"Eagle round {r} finished. Saved to {save_dir / f'eagle_round_{r}'}")

    # Final Save
    hen_model.save(save_dir / "hen_stage_3_final")
    eagle_model.save(save_dir / "eagle_stage_3_final")
    
    print("\nCo-training complete!")
    hen_env.close()
    eagle_env.close()

if __name__ == "__main__":
    main()

