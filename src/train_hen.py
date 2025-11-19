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

from curriculum_env import HenTrainingEnv, PhysicsConfig

# --- 全局配置参数 ---
TOTAL_STEPS = 1_000_000        # PPO 总训练步数
EVAL_FREQ = 100_000            # 评估频率（按环境步数计）
SAVE_DIR = "results/curriculum"  # 模型和日志保存目录
SEED = 42                      # 随机种子
DEVICE = "auto"                # 训练设备 (auto, cpu, cuda)
N_ENVS = 16                    # 并行环境数量
BATCH_SIZE = 512               # 批处理大小
N_STEPS = 256                  # 每个环境每次更新的采样步数
LEARNING_RATE = 3e-4           # 学习率
GAMMA = 0.995                  # 折扣因子


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
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS, help="PPO 总训练步数。")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ, help="评估频率（按环境步数计）。")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="模型和日志保存目录。")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子。")
    parser.add_argument("--device", type=str, default=DEVICE, help="训练设备 (auto, cpu, cuda)。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhysicsConfig()
    
    # 使用 16 个并行环境，充分利用 14700KF 的多核优势
    # 总 Batch Size = n_envs * n_steps = 16 * 256 = 4096
    env = make_vec_env(
        HenTrainingEnv,
        n_envs=N_ENVS,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": cfg},
    )
    
    # 评估环境保持单个即可，避免多余开销
    eval_env = HenTrainingEnv(config=cfg, seed=args.seed + 1)

    # 使用子目录来保存最佳模型，防止被后续阶段覆盖
    best_model_dir = save_dir / "best_hen"
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
        description="母鸡训练进度",
    )

    callbacks = CallbackList([eval_cb, progress_cb])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # 保留 SB3 自身的英文日志输出
        tensorboard_log=str(save_dir / "tb"),
        seed=args.seed,
        batch_size=BATCH_SIZE,     # 稍微增大 batch_size 以适应更大的吞吐量
        n_steps=N_STEPS,           # 每个环境采 256 步，总共 16*256=4096 步进行一次更新
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_steps, callback=callbacks, tb_log_name="Hen_Stage1")
    model.save(save_dir / "hen_stage_1")


if __name__ == "__main__":
    main()
