import argparse
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import math
import numpy as np
from stable_baselines3 import PPO

from curriculum_env import EagleTrainingEnv, PhysicsConfig


def _configure_chinese_font() -> None:
    """
    配置 Matplotlib 的中文字体，避免中文标题/标签显示为乱码或方块。
    """
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
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
    mpl.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数（阶段二：老鹰策略可视化）。
    """
    parser = argparse.ArgumentParser(
        description="阶段二老鹰策略可视化（冻结母鸡 + 小鸡链条）。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/curriculum/eagle_stage_1.zip",
        help="老鹰策略模型路径（.zip），默认使用阶段二训练脚本保存的 eagle_stage_1.zip。",
    )
    parser.add_argument(
        "--hen-model",
        type=str,
        default="results/curriculum/hen_stage_1.zip",
        help="冻结母鸡策略模型路径（.zip），需与阶段二训练时使用的一致。",
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
        help="环境随机种子基数（不同 episode 在此基础上递增）。",
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
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")
    ax.set_title("阶段二：老鹰进攻可视化（冻结母鸡 + 小鸡链条）")
    ax.set_xlabel("X 位置")
    ax.set_ylabel("Y 位置")

    hen_scatter = ax.scatter([], [], s=80, c="tab:orange", label="母鸡")
    eagle_scatter = ax.scatter([], [], s=80, c="tab:blue", label="老鹰")
    chicks_scatter = ax.scatter([], [], s=40, c="tab:green", label="小鸡链条")
    hen_wing_line, = ax.plot(
        [], [], color="red", linewidth=2.0, alpha=0.8, label="母鸡翅膀"
    )

    ax.legend(loc="upper right")

    handles = {
        "hen": hen_scatter,
        "eagle": eagle_scatter,
        "chicks": chicks_scatter,
        "hen_wing": hen_wing_line,
    }
    return fig, ax, handles


def _update_scatter(env: EagleTrainingEnv, handles: dict, ax: plt.Axes, step_idx: int) -> None:
    """
    根据当前环境状态更新散点位置与翅膀线。
    """
    hen_pos = env.hen.position
    eagle_pos = env.eagle.position
    chick_positions = [body.position for body in env.chicks]

    handles["hen"].set_offsets(np.array([[hen_pos.x, hen_pos.y]], dtype=float))
    handles["eagle"].set_offsets(np.array([[eagle_pos.x, eagle_pos.y]], dtype=float))

    caught = False
    if chick_positions:
        tail_pos = chick_positions[-1]
        dist_eagle_tail = math.hypot(
            float(eagle_pos.x - tail_pos.x), float(eagle_pos.y - tail_pos.y)
        )
        catch_radius = getattr(env.cfg, "catch_radius", 0.0)
        if catch_radius > 0.0 and dist_eagle_tail < catch_radius:
            caught = True

        xs = [p.x for p in chick_positions]
        ys = [p.y for p in chick_positions]
        handles["chicks"].set_offsets(np.column_stack([xs, ys]))

        colors = ["tab:green"] * len(chick_positions)
        if caught and len(colors) > 0:
            colors[-1] = "red"
        handles["chicks"].set_color(colors)

        # 绘制母鸡翅膀：垂直于母鸡->尾端方向，长度根据 block_margin 决定
        arm_span = max(getattr(env.cfg, "block_margin", 0.0), 0.0)
        if arm_span > 0.0:
            vx = float(tail_pos.x - hen_pos.x)
            vy = float(tail_pos.y - hen_pos.y)
            norm = math.hypot(vx, vy)
            if norm < 1e-5:
                nx, ny = 0.0, 1.0
            else:
                # 垂直方向
                nx = -vy / norm
                ny = vx / norm
            x1 = hen_pos.x - nx * arm_span
            y1 = hen_pos.y - ny * arm_span
            x2 = hen_pos.x + nx * arm_span
            y2 = hen_pos.y + ny * arm_span
            handles["hen_wing"].set_data([x1, x2], [y1, y2])
        else:
            handles["hen_wing"].set_data([], [])
    else:
        handles["chicks"].set_offsets(np.empty((0, 2), dtype=float))
        handles["hen_wing"].set_data([], [])

    ax.set_xlabel(f"当前步数：{step_idx}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误：未找到老鹰模型文件：{model_path}")
        print("请先完成阶段二训练，或检查 --model 参数路径是否正确。")
        return

    hen_model_path = Path(args.hen_model)
    if not hen_model_path.exists():
        print(f"错误：未找到母鸡模型文件：{hen_model_path}")
        print("请确保已经完成阶段一训练，或检查 --hen-model 参数路径是否正确。")
        return

    print(f"加载老鹰策略模型：{model_path}")
    model = PPO.load(model_path.as_posix())

    cfg = PhysicsConfig()
    env = EagleTrainingEnv(
        hen_policy_path=hen_model_path.as_posix(), config=cfg, seed=args.seed
    )

    fig, ax, handles = _init_figure(world_size=cfg.world_size)
    pause = 1.0 / max(float(args.fps), 1.0)

    # 注册关闭事件回调
    is_running = True

    def on_close(event):
        nonlocal is_running
        is_running = False

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        for ep in range(args.episodes):
            if not is_running:
                break

            print(f"开始第 {ep + 1} 个 episode 的可视化。")
            obs, _ = env.reset(seed=args.seed + ep)
            episode_reward = 0.0

            for step in range(cfg.max_steps):
                if not is_running:
                    print("窗口已关闭，停止可视化。")
                    return

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

        if is_running:
            print("阶段二可视化结束，关闭窗口即可退出程序。")
            plt.ioff()
            plt.show()
        else:
            print("可视化已中断。")
    finally:
        env.close()


if __name__ == "__main__":
    main()


