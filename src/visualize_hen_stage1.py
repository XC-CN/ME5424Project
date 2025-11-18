import argparse
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from stable_baselines3 import PPO

from curriculum_env import HenTrainingEnv, PhysicsConfig


def _configure_chinese_font() -> None:
    """
    配置 Matplotlib 的中文字体，避免中文标题/标签显示为乱码或方块。

    会优先尝试常见中文字体，若系统不存在则回落到默认字体。
    """
    preferred_fonts = [
        "Microsoft YaHei",  # Windows 常见
        "SimHei",  # 黑体
        "PingFang SC",  # macOS
        "Noto Sans CJK SC",  # 通用 CJK
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    for name in preferred_fonts:
        try:
            fm.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        else:
            mpl.rcParams["font.sans-serif"] = [name] + mpl.rcParams.get(
                "font.sans-serif", []
            )
            break
    # 解决负号显示问题
    mpl.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    所有帮助信息均使用中文，便于在控制台查看。
    """
    parser = argparse.ArgumentParser(
        description="阶段一母鸡策略可视化（包含老鹰与小鸡链条）。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/curriculum/best_model.zip",
        help="母鸡策略模型路径（.zip），默认使用评估过程中表现最好的模型。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="可视化的 episode 数量。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="环境随机种子基数（不同 episode 会在此基础上递增）。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="可视化帧率（每秒刷新次数，默认 15）。",
    )
    return parser.parse_args()


def _init_figure(world_size: float) -> Tuple[plt.Figure, plt.Axes, dict]:
    """
    初始化 Matplotlib 图像窗口与散点句柄。

    返回值:
        fig, ax, handles 字典（包含 hen/eagle/chicks 三类散点）。
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")
    ax.set_title("阶段一：母鸡防守可视化（含小鸡链条）")
    ax.set_xlabel("X 位置")
    ax.set_ylabel("Y 位置")

    hen_scatter = ax.scatter([], [], s=80, c="tab:orange", label="母鸡")
    eagle_scatter = ax.scatter([], [], s=80, c="tab:blue", label="老鹰")
    chicks_scatter = ax.scatter([], [], s=40, c="tab:green", label="小鸡链条")

    ax.legend(loc="upper right")

    handles = {
        "hen": hen_scatter,
        "eagle": eagle_scatter,
        "chicks": chicks_scatter,
    }
    return fig, ax, handles


def _update_scatter(env: HenTrainingEnv, handles: dict, ax: plt.Axes, step_idx: int) -> None:
    """
    根据当前环境状态更新散点位置。

    显示母鸡、老鹰以及整条小鸡链条的位置。
    """
    hen_pos = env.hen.position
    eagle_pos = env.eagle.position
    chick_positions = [body.position for body in env.chicks]

    handles["hen"].set_offsets(np.array([[hen_pos.x, hen_pos.y]], dtype=float))
    handles["eagle"].set_offsets(np.array([[eagle_pos.x, eagle_pos.y]], dtype=float))

    if chick_positions:
        xs = [p.x for p in chick_positions]
        ys = [p.y for p in chick_positions]
        handles["chicks"].set_offsets(np.column_stack([xs, ys]))
    else:
        handles["chicks"].set_offsets(np.empty((0, 2), dtype=float))

    ax.set_xlabel(f"当前步数：{step_idx}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误：未找到模型文件：{model_path}")
        print("请先完成阶段一训练，或检查 --model 参数路径是否正确。")
        return

    print(f"加载母鸡策略模型：{model_path}")
    model = PPO.load(model_path.as_posix())

    cfg = PhysicsConfig()
    env = HenTrainingEnv(config=cfg, seed=args.seed)

    fig, ax, handles = _init_figure(world_size=cfg.world_size)
    pause = 1.0 / max(float(args.fps), 1.0)

    try:
        for ep in range(args.episodes):
            print(f"开始第 {ep + 1} 个 episode 的可视化。")
            obs, _ = env.reset(seed=args.seed + ep)
            episode_reward = 0.0

            for step in range(cfg.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += float(reward)
                _update_scatter(env, handles, ax, step_idx=step + 1)

                fig.canvas.draw()
                fig.canvas.flush_events()
                if pause > 0:
                    plt.pause(pause)

                if terminated or truncated:
                    print(
                        f"第 {ep + 1} 个 episode 在第 {step + 1} 步结束，累计奖励 = {episode_reward:.3f}。"
                    )
                    break

        print("可视化结束，关闭窗口即可退出程序。")
        plt.ioff()
        plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()


