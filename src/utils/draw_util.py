import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import imageio
from PIL import Image
import numpy as np

# 初始化文本对象为None
text_obj = None


def resize_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # Resize the image to be divisible by 16
    new_width = (img.width // 16) * 16
    new_height = (img.height // 16) * 16
    if new_width != img.width or new_height != img.height:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(img)


def get_gradient_color(start_color, end_color, num_points, idx):
    start_rgba = np.array(mcolors.to_rgba(start_color))
    end_rgba = np.array(mcolors.to_rgba(end_color))
    ratio = idx / max(1, num_points - 1)
    gradient_rgba = start_rgba + (end_rgba - start_rgba) * ratio
    return mcolors.to_hex(gradient_rgba)


def update(ax, env, uav_plots, target_plots, protector_plots, uav_search_patches, frame, num_steps, interval=2, paint_all=True):
    global text_obj

    if frame == 0:
        return
    for i, uav in enumerate(env.uav_list):
        uav_x = env.position['all_uav_xs'][0: frame: interval]
        uav_y = env.position['all_uav_ys'][0: frame: interval]
        uav_x = [sublist[i] for sublist in uav_x]
        uav_y = [sublist[i] for sublist in uav_y]
        if uav_x and uav_y:
            colors = [get_gradient_color('#E1FFFF', "#1100FF", frame, idx) for idx in range(len(uav_x))]
            uav_plots[i].set_offsets(np.column_stack([uav_x, uav_y]))
            uav_plots[i].set_color(colors)
            uav_search_patches[i].center = (uav_x[-1], uav_y[-1])
        else:
            print(f"Warning: UAV {i} position list is empty at frame {frame}.")

    for i in range(env.m_targets):
        target_x = env.position['all_target_xs'][0: frame: interval]
        target_y = env.position['all_target_ys'][0: frame: interval]
        target_x = [sublist[i] for sublist in target_x]
        target_y = [sublist[i] for sublist in target_y]
        if target_x and target_y:
            colors = [get_gradient_color('#FFC0CB', '#DC143C', frame, idx) for idx in range(len(target_x))]
            target_plots[i].set_offsets(np.column_stack([target_x, target_y]))
            target_plots[i].set_color(colors)
        else:
            print(f"Warning: Target {i} position list is empty at frame {frame}.")

    for i, _ in enumerate(env.protector_list):
        prot_x = env.position['all_protector_xs'][0: frame: interval]
        prot_y = env.position['all_protector_ys'][0: frame: interval]
        prot_x = [sublist[i] for sublist in prot_x]
        prot_y = [sublist[i] for sublist in prot_y]
        if prot_x and prot_y:
            colors = [get_gradient_color('#E0FFE0', '#008000', frame, idx) for idx in range(len(prot_x))]
            protector_plots[i].set_offsets(np.column_stack([prot_x, prot_y]))
            protector_plots[i].set_color(colors)
        else:
            print(f"Warning: Protector {i} position list is empty at frame {frame}.")

    text_str = (
        f"detected target num = {env.covered_target_num[frame]}\n"
        f"detected target rate = {env.covered_target_num[frame] / env.m_targets * 100:.2f}%"
    )

    if text_obj is not None:
        text_obj.remove()

    text_obj = ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       color='black')


def draw_animation(config, env, num_steps, ep_num, frames=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
    ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
    uav_plots = [ax.scatter([], [], marker='o', color='b', linestyle='None', s=2, alpha=1) for _ in range(env.n_uav)]
    target_plots = [ax.scatter([], [], marker='o', color='r', linestyle='None', s=3, alpha=1) for _ in range(env.m_targets)]
    protector_plots = [ax.scatter([], [], marker='o', color='g', linestyle='None', s=4, alpha=1) for _ in range(env.n_protectors)]
    uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
    for patch in uav_search_patches:
        ax.add_patch(patch)

    save_dir = os.path.join(config["save_dir"], "frames")
    os.makedirs(save_dir, exist_ok=True)

    step_interval = 5
    for frame in range(0, num_steps, step_interval):
        update(ax, env, uav_plots, target_plots, protector_plots, uav_search_patches, frame, num_steps, step_interval)
        plt.draw()
        plt.savefig(os.path.join(save_dir, f'frame_{frame:04d}.png'))
        plt.pause(0.001)

    plt.close(fig)

    video_path = os.path.join(config["save_dir"], "animated", f'animated_plot_{ep_num + 1}.mp4')
    writer = imageio.get_writer(video_path, fps=5, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            img_array = resize_image(frame_path)
            writer.append_data(img_array)
    writer.close()

    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)


def plot_reward_curve(config, return_list, name):
    plt.figure(figsize=(6, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(config["save_dir"], name + ".png"))


# 新增：贴图动画渲染

def _safe_read_image(path, fallback_color=(128, 128, 128, 255), size=(256, 256)):
    try:
        img = Image.open(path).convert('RGBA')
        return np.array(img)
    except Exception:
        # 生成占位图
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([size[0]*0.25, size[1]*0.25, size[0]*0.75, size[1]*0.75], fill=fallback_color)
        return np.array(img)


def draw_textured_animation(config, env, num_steps, ep_num, assets_dir="assets", step_interval=5, uva_zoom=0.5, target_zoom=0.24, protector_zoom=0.34):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.x_max)
    ax.set_ylim(0, env.y_max)
    ax.set_aspect('equal')

    # 背景贴图
    bg_path = os.path.join(assets_dir, "background.png")
    bg_img = _safe_read_image(bg_path, fallback_color=(235, 255, 235, 255), size=(1024, 1024))
    ax.imshow(bg_img, extent=[0, env.x_max, 0, env.y_max], zorder=0)

    # 载入角色贴图
    uav_img = _safe_read_image(os.path.join(assets_dir, "eagle.png"), fallback_color=(30, 144, 255, 255))
    tgt_img = _safe_read_image(os.path.join(assets_dir, "chick.png"), fallback_color=(255, 215, 0, 255))
    prot_img = _safe_read_image(os.path.join(assets_dir, "protector.png"), fallback_color=(255, 0, 0, 255))

    # 初始化 artists（使用当前起始位置）
    def make_artist(img, x, y, size, z):
        s = size
        return ax.imshow(img, extent=(x - s, x + s, y - s, y + s), zorder=z)

    s_uav = env.uav_list[0].dp * uva_zoom
    s_tgt = env.uav_list[0].dp * target_zoom
    s_prot = env.uav_list[0].dp * protector_zoom
    uav_artists = [make_artist(uav_img, uav.x, uav.y, size=s_uav, z=10) for uav in env.uav_list]
    tgt_artists = [make_artist(tgt_img, t.x, t.y, size=s_tgt, z=9) for t in env.target_list]
    prot_artists = [make_artist(prot_img, p.x, p.y, size=s_prot, z=11) for p in env.protector_list]
    # 可探测范围（dp）覆盖层
    uav_dp_patches = [patches.Circle((uav.x, uav.y), uav.dp, facecolor='lightblue', edgecolor='dodgerblue', alpha=0.2, linewidth=1.0, zorder=8) for uav in env.uav_list]
    for patch in uav_dp_patches:
        ax.add_patch(patch)

    # 阻挡手臂线段（前臂/后臂）
    arm_front_lines = [ax.plot([], [], color='red', linewidth=2.0, zorder=12)[0] for _ in env.protector_list]
    arm_rear_lines = [ax.plot([], [], color='red', linewidth=2.0, zorder=12)[0] for _ in env.protector_list]

    save_dir = os.path.join(config["save_dir"], "frames")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "animated"), exist_ok=True)

    # 基于历史轨迹更新到当前帧的位置（不旋转）
    total_steps = min(num_steps, len(env.position['all_uav_xs']))
    for frame in range(1, total_steps, step_interval):
        # UAV 更新
        if frame < len(env.position['all_uav_xs']):
            uav_xs = env.position['all_uav_xs'][frame]
            uav_ys = env.position['all_uav_ys'][frame]
            for i in range(min(len(uav_artists), len(uav_xs))):
                uav_artists[i].set_extent((uav_xs[i] - s_uav, uav_xs[i] + s_uav, uav_ys[i] - s_uav, uav_ys[i] + s_uav))
                # 同步更新可探测范围圆心
                uav_dp_patches[i].center = (uav_xs[i], uav_ys[i])
        # Target 更新（被捕获后隐藏）
        if frame < len(env.position['all_target_xs']):
            tgt_xs = env.position['all_target_xs'][frame]
            tgt_ys = env.position['all_target_ys'][frame]
            for i in range(min(len(tgt_artists), len(tgt_xs))):
                # 只要被捕获就隐藏；若记录了捕获帧，则 frame >= captured_step 时隐藏
                captured_flag = hasattr(env.target_list[i], 'captured') and env.target_list[i].captured
                has_step = hasattr(env.target_list[i], 'captured_step') and env.target_list[i].captured_step is not None
                captured = captured_flag and (not has_step or frame >= env.target_list[i].captured_step)
                if captured:
                    tgt_artists[i].set_visible(False)
                else:
                    tgt_artists[i].set_visible(True)
                    tgt_artists[i].set_extent((tgt_xs[i] - s_tgt, tgt_xs[i] + s_tgt, tgt_ys[i] - s_tgt, tgt_ys[i] + s_tgt))
        # Protector 更新 + 阻挡手臂线段
        if frame < len(env.position['all_protector_xs']):
            prot_xs = env.position['all_protector_xs'][frame]
            prot_ys = env.position['all_protector_ys'][frame]
            for i in range(min(len(prot_artists), len(prot_xs))):
                prot_artists[i].set_extent((prot_xs[i] - s_prot, prot_xs[i] + s_prot, prot_ys[i] - s_prot, prot_ys[i] + s_prot))
        # 手臂绘制（垂直于运动方向，始终可视）
        if frame < len(env.position['all_protector_xs']):
            prot_xs = env.position['all_protector_xs'][frame]
            prot_ys = env.position['all_protector_ys'][frame]
            # 用上一帧计算速度方向，确保“垂直于运动”
            prev_idx = max(0, frame - 1)
            prev_xs = env.position['all_protector_xs'][prev_idx] if prev_idx < len(env.position['all_protector_xs']) else prot_xs
            prev_ys = env.position['all_protector_ys'][prev_idx] if prev_idx < len(env.position['all_protector_ys']) else prot_ys
        
            for i in range(len(arm_front_lines)):
                if i < len(env.protector_list) and i < len(prot_xs):
                    cx, cy = prot_xs[i], prot_ys[i]
                    vx = cx - prev_xs[i]
                    vy = cy - prev_ys[i]
                    speed = np.hypot(vx, vy)
        
                    if speed > 1e-8:
                        # 运动方向的法向（垂直）：(-vy, vx)
                        nx, ny = -vy / speed, vx / speed
                    else:
                        # 静止时回退用朝向h的法向（垂直于朝向）
                        h = getattr(env.protector_list[i], 'h', 0.0)
                        nx, ny = -np.sin(h), np.cos(h)
        
                    L = getattr(env.protector_list[i], 'safe_r', s_prot)  # 半长度：配置可调
                    x1, y1 = cx - nx * L, cy - ny * L
                    x2, y2 = cx + nx * L, cy + ny * L
        
                    arm_front_lines[i].set_data([cx, x2], [cy, y2])
                    arm_rear_lines[i].set_data([cx, x1], [cy, y1])
                else:
                    arm_front_lines[i].set_data([], [])
                    arm_rear_lines[i].set_data([], [])

        # 覆盖文本
        if frame < len(env.covered_target_num):
            text_str = (
                f"detected target num = {env.covered_target_num[frame]}\n"
                f"detected target rate = {env.covered_target_num[frame] / env.m_targets * 100:.2f}%"
            )
            global text_obj
            if text_obj is not None:
                text_obj.remove()
            text_obj = ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                               color='black')

        plt.draw()
        plt.savefig(os.path.join(save_dir, f'tex_frame_{frame:04d}.png'))
        plt.pause(0.0001)

    plt.close(fig)

    # 生成 MP4
    video_path = os.path.join(config["save_dir"], "animated", f'animated_textured_plot_{ep_num + 1}.mp4')
    writer = imageio.get_writer(video_path, fps=5, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

    for frame in range(1, total_steps, step_interval):
        frame_path = os.path.join(save_dir, f'tex_frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            img_array = resize_image(frame_path)
            writer.append_data(img_array)
    writer.close()

    # 清理帧图
    for frame in range(1, total_steps, step_interval):
        frame_path = os.path.join(save_dir, f'tex_frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)

