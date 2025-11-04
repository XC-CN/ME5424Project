import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import imageio


class LiveRenderer:
    """
    鍗曠獥鍙ｅ疄鏃舵覆鏌撳櫒锛屽彲閫夋樉绀猴紝涔熷彲浠ュ綍鍒舵垚瑙嗛銆?    """

    def __init__(
        self,
        env,
        pause=0.05,
        trail_steps=60,
        show=True,
        record=False,
        video_path=None,
        video_fps=10,
    ):
        self.env = env
        self.pause = max(float(pause), 0.0)
        self.trail_steps = max(int(trail_steps), 1)
        self.show = bool(show)
        self.record = bool(record and video_path)
        self.video_path = video_path
        self.video_fps = max(int(video_fps), 1)
        self.video_writer = None

        if self.show:
            plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, env.x_max)
        self.ax.set_ylim(0, env.y_max)
        self.ax.set_aspect('equal')
        self.ax.set_title("璇勪及浠跨湡瀹炴椂鎾斁")

        self.uav_scatter = self.ax.scatter([], [], s=70, c='#1f77b4', label='鑰侀拱')
        self.target_scatter = self.ax.scatter([], [], s=50, c='#ff7f0e', label='灏忛浮', alpha=0.9)
        self.protector_scatter = self.ax.scatter([], [], s=60, c='#2ca02c', label='姣嶉浮', alpha=0.9)

        self.uav_trails = [
            self.ax.plot([], [], color='#1f77b4', linewidth=1.2, alpha=0.4)[0]
            for _ in env.uav_list
        ]
        self.target_trails = [
            self.ax.plot([], [], color='#ff7f0e', linewidth=1.0, alpha=0.3)[0]
            for _ in env.target_list
        ]
        self.protector_trails = [
            self.ax.plot([], [], color='#2ca02c', linewidth=1.0, alpha=0.3)[0]
            for _ in env.protector_list
        ]

        self.protector_front_wings = [
            self.ax.plot([], [], color='#d62728', linewidth=2.0, alpha=0.85)[0]
            for _ in env.protector_list
        ]
        self.protector_rear_wings = [
            self.ax.plot([], [], color='#d62728', linewidth=1.4, alpha=0.6, linestyle='--')[0]
            for _ in env.protector_list
        ]

        self.uav_ranges = [
            patches.Circle(
                (uav.x, uav.y),
                getattr(uav, 'dp', 0.0),
                facecolor='none',
                edgecolor='dodgerblue',
                alpha=0.4,
                linewidth=1.0,
            )
            for uav in env.uav_list
        ]
        for patch in self.uav_ranges:
            self.ax.add_patch(patch)

        self.coverage_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, fontsize=10,
            verticalalignment='top', color='black'
        )
        self.ax.legend(loc='upper right')

        if self.record:
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            self.video_writer = imageio.get_writer(
                self.video_path,
                fps=self.video_fps,
                codec='libx264',
                format='FFMPEG',
                pixelformat='yuv420p',
            )

    @staticmethod
    def _set_offsets(scatter, xs, ys):
        if xs and ys:
            scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            scatter.set_offsets(np.empty((0, 2)))

    @staticmethod
    def _set_trail(line, xs, ys):
        if xs and ys:
            line.set_data(xs, ys)
        else:
            line.set_data([], [])

    def _update_trails(self):
        history_uav_x = self.env.position['all_uav_xs']
        history_uav_y = self.env.position['all_uav_ys']
        history_target_x = self.env.position['all_target_xs']
        history_target_y = self.env.position['all_target_ys']
        history_prot_x = self.env.position['all_protector_xs']
        history_prot_y = self.env.position['all_protector_ys']

        for idx, line in enumerate(self.uav_trails):
            xs = [step[idx] for step in history_uav_x[-self.trail_steps:] if idx < len(step)]
            ys = [step[idx] for step in history_uav_y[-self.trail_steps:] if idx < len(step)]
            self._set_trail(line, xs, ys)

        for idx, line in enumerate(self.target_trails):
            xs = [step[idx] for step in history_target_x[-self.trail_steps:] if idx < len(step)]
            ys = [step[idx] for step in history_target_y[-self.trail_steps:] if idx < len(step)]
            self._set_trail(line, xs, ys)

        for idx, line in enumerate(self.protector_trails):
            xs = [step[idx] for step in history_prot_x[-self.trail_steps:] if idx < len(step)]
            ys = [step[idx] for step in history_prot_y[-self.trail_steps:] if idx < len(step)]
            self._set_trail(line, xs, ys)

    def _update_protector_wings(self, env):
        for idx, protector in enumerate(env.protector_list):
            safe_radius = float(getattr(protector, 'safe_r', getattr(protector, 'obs_radius', 0.0)))
            heading = float(getattr(protector, 'h', 0.0))
            front_line = self.protector_front_wings[idx]
            rear_line = self.protector_rear_wings[idx]

            if safe_radius <= 0.0:
                front_line.set_data([], [])
                rear_line.set_data([], [])
                continue

            dir_x = np.cos(heading)
            dir_y = np.sin(heading)
            norm_x = -dir_y
            norm_y = dir_x

            span = safe_radius
            front_center_x = protector.x + dir_x * (0.6 * safe_radius)
            front_center_y = protector.y + dir_y * (0.6 * safe_radius)
            rear_center_x = protector.x - dir_x * (0.4 * safe_radius)
            rear_center_y = protector.y - dir_y * (0.4 * safe_radius)

            front_line.set_data(
                [front_center_x - norm_x * span, front_center_x + norm_x * span],
                [front_center_y - norm_y * span, front_center_y + norm_y * span],
            )
            rear_line.set_data(
                [rear_center_x - norm_x * span * 0.6, rear_center_x + norm_x * span * 0.6],
                [rear_center_y - norm_y * span * 0.6, rear_center_y + norm_y * span * 0.6],
            )

    def _capture_frame(self):
        if not self.record or self.video_writer is None:
            return
        canvas = self.fig.canvas
        renderer = getattr(canvas, "renderer", None)
        if renderer is None:
            return
        raw = renderer.buffer_rgba()
        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((int(round(renderer.height)), int(round(renderer.width)), 4))
        frame = frame[..., :3]
        self.video_writer.append_data(frame)

    def __call__(self, step, env=None):
        env = env or self.env
        xs = [agent.x for agent in env.uav_list]
        ys = [agent.y for agent in env.uav_list]
        self._set_offsets(self.uav_scatter, xs, ys)

        tgt_xs = [agent.x for agent in env.target_list if not getattr(agent, 'captured', False)]
        tgt_ys = [agent.y for agent in env.target_list if not getattr(agent, 'captured', False)]
        self._set_offsets(self.target_scatter, tgt_xs, tgt_ys)

        prot_xs = [agent.x for agent in env.protector_list]
        prot_ys = [agent.y for agent in env.protector_list]
        self._set_offsets(self.protector_scatter, prot_xs, prot_ys)

        for patch, uav in zip(self.uav_ranges, env.uav_list):
            patch.center = (uav.x, uav.y)
            patch.set_radius(getattr(uav, 'dp', patch.radius))

        self._update_trails()
        self._update_protector_wings(env)

        coverage = env.covered_target_num[-1] if env.covered_target_num else 0
        rate = coverage / max(env.m_targets, 1) * 100.0
        self.coverage_text.set_text(
            f"Step: {step + 1}\n覆盖目标：{coverage}\n覆盖率：{rate:.1f}%"
        )

        self.fig.canvas.draw()
        if self.show:
            self.fig.canvas.flush_events()
            if self.pause > 0.0:
                plt.pause(self.pause)
        self._capture_frame()

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        plt.ioff()
        plt.close(self.fig)


def plot_reward_curve(config, return_list, name):
    plt.figure(figsize=(6, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title(name)
    plt.grid(True)
    os.makedirs(config["save_dir"], exist_ok=True)
    plt.savefig(os.path.join(config["save_dir"], f"{name}.png"))
    plt.close()


__all__ = ["LiveRenderer", "plot_reward_curve"]



