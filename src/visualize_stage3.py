import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from stable_baselines3 import PPO

# 确保能导入 src 下的模块
sys.path.append(str(Path(__file__).parent))
from curriculum_env import EagleTrainingEnv, PhysicsConfig, resolve_model_path


def _configure_chinese_font() -> None:
    """
    配置 Matplotlib 的中文字体。
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
            mpl.rcParams["font.sans-serif"] = [name] + mpl.rcParams.get("font.sans-serif", [])
            break
    mpl.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段三：母鸡与老鹰联合对抗可视化")
    parser.add_argument(
        "--hen-model",
        type=str,
        default="results/curriculum/stage3/hen_stage_3_final.zip",
        help="训练好的母鸡模型路径",
    )
    parser.add_argument(
        "--eagle-model",
        type=str,
        default="results/curriculum/stage3/eagle_stage_3_final.zip",
        help="训练好的老鹰模型路径",
    )
    parser.add_argument("--episodes", type=int, default=3, help="可视化回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fps", type=float, default=20.0, help="可视化帧率")
    return parser.parse_args()


def _init_figure(world_size: float) -> Tuple[plt.Figure, plt.Axes, dict]:
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")
    ax.set_title("阶段三：母鸡(Orange) vs 老鹰(Blue) 联合对抗")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    hen_scatter = ax.scatter([], [], s=100, c="tab:orange", label="母鸡 (Hen)")
    eagle_scatter = ax.scatter([], [], s=100, c="tab:blue", label="老鹰 (Eagle)")
    chicks_scatter = ax.scatter([], [], s=50, c="tab:green", label="小鸡 (Chicks)")
    hen_wing_line, = ax.plot([], [], color="red", linewidth=2.0, alpha=0.7, label="阻挡带")

    ax.legend(loc="upper right")
    handles = {
        "hen": hen_scatter,
        "eagle": eagle_scatter,
        "chicks": chicks_scatter,
        "hen_wing": hen_wing_line,
    }
    return fig, ax, handles


def _update_scatter(env: EagleTrainingEnv, handles: dict, ax: plt.Axes, step_idx: int) -> None:
    hen_pos = env.hen.position
    eagle_pos = env.eagle.position
    chick_positions = [body.position for body in env.chicks]

    handles["hen"].set_offsets(np.array([[hen_pos.x, hen_pos.y]], dtype=float))
    handles["eagle"].set_offsets(np.array([[eagle_pos.x, eagle_pos.y]], dtype=float))

    # 小鸡链条
    if chick_positions:
        xs = [p.x for p in chick_positions]
        ys = [p.y for p in chick_positions]
        handles["chicks"].set_offsets(np.column_stack([xs, ys]))
        
        # 检查抓捕状态
        tail_pos = chick_positions[-1]
        dist_eagle_tail = math.hypot(eagle_pos.x - tail_pos.x, eagle_pos.y - tail_pos.y)
        caught = dist_eagle_tail < getattr(env.cfg, "catch_radius", 0.7)
        
        colors = ["tab:green"] * len(chick_positions)
        if caught:
            colors[-1] = "red"
        handles["chicks"].set_color(colors)

        # 母鸡翅膀 (Block Margin)
        arm_span = max(getattr(env.cfg, "block_margin", 0.0), 0.0)
        if arm_span > 0.0:
            vx = float(tail_pos.x - hen_pos.x)
            vy = float(tail_pos.y - hen_pos.y)
            norm = math.hypot(vx, vy)
            if norm < 1e-5:
                nx, ny = 0.0, 1.0
            else:
                nx, ny = -vy / norm, vx / norm
            
            x1, y1 = hen_pos.x - nx * arm_span, hen_pos.y - ny * arm_span
            x2, y2 = hen_pos.x + nx * arm_span, hen_pos.y + ny * arm_span
            handles["hen_wing"].set_data([x1, x2], [y1, y2])
    
    ax.set_xlabel(f"Step: {step_idx}")


def main():
    args = parse_args()
    
    # 1. Resolve paths
    hen_path = resolve_model_path(args.hen_model, "hen_stage_3_final")
    eagle_path = resolve_model_path(args.eagle_model, "eagle_stage_3_final")
    
    if not hen_path.exists():
        print(f"Error: Hen model not found at {hen_path}")
        return
    if not eagle_path.exists():
        print(f"Error: Eagle model not found at {eagle_path}")
        return

    print(f"Loading Hen: {hen_path}")
    print(f"Loading Eagle: {eagle_path}")
    
    # 2. Load Eagle Model (Hen model is loaded by Env)
    eagle_model = PPO.load(eagle_path.as_posix(), device="cpu")
    
    # 3. Create Environment
    # Using EagleTrainingEnv is convenient because it handles the "Hen vs Model" logic internally
    cfg = PhysicsConfig()
    env = EagleTrainingEnv(
        hen_policy_path=hen_path.as_posix(),
        config=cfg,
        seed=args.seed,
        device="cpu"
    )

    # 4. Visualization Loop
    fig, ax, handles = _init_figure(cfg.world_size)
    pause_time = 1.0 / max(args.fps, 1.0)
    
    is_running = True
    def on_close(event):
        nonlocal is_running
        is_running = False
    fig.canvas.mpl_connect("close_event", on_close)

    try:
        for ep in range(args.episodes):
            if not is_running: break
            
            print(f"--- Episode {ep+1} ---")
            obs, _ = env.reset(seed=args.seed + ep)
            total_reward = 0.0
            
            for step in range(cfg.max_steps):
                if not is_running: break
                
                # Eagle acts via model
                action, _ = eagle_model.predict(obs, deterministic=True)
                
                # Env applies Hen action internally (using loaded hen model)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                
                # Render
                _update_scatter(env, handles, ax, step+1)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(pause_time)
                
                if terminated or truncated:
                    res = "Caught!" if terminated else "Time out"
                    print(f"Episode finished at step {step+1}. Result: {res}, Reward: {total_reward:.2f}")
                    break
            
            if is_running:
                plt.pause(1.0)

    finally:
        env.close()
        print("Visualization finished.")
        if is_running:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    main()

