import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from tqdm import tqdm

from curriculum_env import HenTrainingEnv, PhysicsConfig


class ProgressBarCallback(BaseCallback):
    """
    简单的训练进度条回调。

    使用 tqdm 显示当前已训练步数，所有控制台提示信息为中文。
    """

    def __init__(self, total_timesteps: int, description: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.description = description
        self._pbar = None

    def _on_training_start(self) -> None:
        print(f"{self.description}：开始训练，总步数 = {self.total_timesteps}")
        self._pbar = tqdm(
            total=self.total_timesteps,
            desc=self.description,
            unit="步",
            ascii=True,
        )
        return None

    def _on_step(self) -> bool:
        if self._pbar is not None:
            # 使用 SB3 内部记录的总步数刷新进度条
            current = min(int(self.model.num_timesteps), self.total_timesteps)
            self._pbar.n = current
            self._pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        print(f"{self.description}：训练结束。")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段一：训练母鸡，对手为启发式老鹰。")
    parser.add_argument("--total-steps", type=int, default=300_000, help="PPO 总训练步数。")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="评估频率（按环境步数计）。")
    parser.add_argument("--save-dir", type=str, default="results/curriculum", help="模型和日志保存目录。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhysicsConfig()
    env = HenTrainingEnv(config=cfg, seed=args.seed)
    eval_env = HenTrainingEnv(config=cfg, seed=args.seed + 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    progress_cb = ProgressBarCallback(
        total_timesteps=args.total_steps,
        description="母鸡训练进度",
    )

    callbacks = CallbackList([eval_cb, progress_cb])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # 保留 SB3 自身的英文日志输出
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
        batch_size=256,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.995,
    )
    model.learn(total_timesteps=args.total_steps, callback=callbacks)
    model.save(save_dir / "hen_stage_1")


if __name__ == "__main__":
    main()

