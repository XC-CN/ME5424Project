import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from curriculum_env import EagleTrainingEnv, PhysicsConfig


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
    parser = argparse.ArgumentParser(description="阶段二：在冻结母鸡策略下训练老鹰。")
    parser.add_argument(
        "--hen-model",
        type=str,
        default="results/curriculum/best_hen/best_model.zip",
        help="已训练母鸡策略的模型路径（建议使用 best_model）。",
    )
    parser.add_argument("--total-steps", type=int, default=300_000, help="PPO 总训练步数。")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="评估频率（按环境步数计）。")
    parser.add_argument("--save-dir", type=str, default="results/curriculum", help="模型和日志保存目录。")
    parser.add_argument("--seed", type=int, default=123, help="随机种子。")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (auto, cpu, cuda)。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhysicsConfig()
    
    # 使用 16 个并行环境，充分利用 14700KF 的多核优势
    # 这里的 env_kwargs 会被传递给每个并行进程中的 EagleTrainingEnv 构造函数
    n_envs = 16
    env = make_vec_env(
        EagleTrainingEnv,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "hen_policy_path": args.hen_model,
            "config": cfg,
            "device": "cpu"  # 强制在 CPU 上运行冻结的母鸡推理，避免多进程争抢 GPU 显存
        },
    )

    # 评估环境保持单进程即可
    eval_env = EagleTrainingEnv(
        hen_policy_path=args.hen_model, config=cfg, seed=args.seed + 1, device=args.device
    )

    # 使用子目录来保存最佳模型，避免覆盖母鸡的模型
    best_model_dir = save_dir / "best_eagle"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(save_dir),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    progress_cb = ProgressBarCallback(
        total_timesteps=args.total_steps,
        description="老鹰训练进度",
    )

    callbacks = CallbackList([eval_cb, progress_cb])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # 保留 SB3 自身的英文日志输出
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
        batch_size=512,     # 配合并行环境增大 batch_size
        n_steps=256,        # 每个环境 256 步，总共 16*256=4096 步更新一次
        learning_rate=3e-4,
        gamma=0.995,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_steps, callback=callbacks, tb_log_name="Eagle_Stage2")
    model.save(save_dir / "eagle_stage_1")


if __name__ == "__main__":
    main()

