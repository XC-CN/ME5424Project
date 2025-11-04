import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm
import numpy as np
import imageio


def _configure_matplotlib() -> None:
    """Configure font fallback so Chinese text renders properly."""
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
        mpl.rcParams["font.sans-serif"] = [name] + [
            f for f in mpl.rcParams.get("font.sans-serif", []) if f != name
        ]
        break
    else:
        mpl.rcParams.setdefault("font.sans-serif", ["DejaVu Sans"])
    mpl.rcParams["axes.unicode_minus"] = False


_configure_matplotlib()


class LiveRenderer:
    """Single-window renderer that can optionally record to MP4."""

    def __init__(
        self,
        env,
        pause: float = 0.05,
        trail_steps: int = 60,
        show: bool = True,
        record: bool = False,
        video_path: str | None = None,
        video_fps: int = 10,
    ) -> None:
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
        self.ax.set_aspect("equal")
        self.ax.set_title("\u8bc4\u4f30\u4eff\u771f\u5b9e\u65f6\u64ad\u653e")

        self.uav_scatter = self.ax.scatter([], [], s=70, c="#1f77b4", label="\u8001\u9e70")
        self.target_scatter = self.ax.scatter([], [], s=50, c="#ff7f0e", label="\u5c0f\u9e21", alpha=0.9)
        self.protector_scatter = self.ax.scatter([], [], s=60, c="#2ca02c", label="\u6bcd\u9e21", alpha=0.9)

        self.uav_trails = [
            self.ax.plot([], [], color="#1f77b4", linewidth=1.2, alpha=0.4)[0]
            for _ in env.uav_list
        ]
        self.target_trails = [
            self.ax.plot([], [], color="#ff7f0e", linewidth=1.0, alpha=0.3)[0]
            for _ in env.target_list
        ]
        self.protector_trails = [
            self.ax.plot([], [], color="#2ca02c", linewidth=1.0, alpha=0.3)[0]
            for _ in env.protector_list
        ]
        self.protector_wings = [
            self.ax.plot([], [], color="#d62728", linewidth=2.0, alpha=0.85)[0]
            for _ in env.protector_list
        ]

        self.uav_ranges = [
            patches.Circle(
                (uav.x, uav.y),
                getattr(uav, "dp", 0.0),
                facecolor="none",
                edgecolor="dodgerblue",
                alpha=0.4,
                linewidth=1.0,
            )
            for uav in env.uav_list
        ]
        for patch in self.uav_ranges:
            self.ax.add_patch(patch)

        self.coverage_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="black",
        )
        self.ax.legend(loc="upper right")

        if self.record:
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            self.video_writer = imageio.get_writer(
                self.video_path,
                fps=self.video_fps,
                codec="libx264",
                format="FFMPEG",
                pixelformat="yuv420p",
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
        history_uav_x = self.env.position["all_uav_xs"]
        history_uav_y = self.env.position["all_uav_ys"]
        history_target_x = self.env.position["all_target_xs"]
        history_target_y = self.env.position["all_target_ys"]
        history_prot_x = self.env.position["all_protector_xs"]
        history_prot_y = self.env.position["all_protector_ys"]

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
            base_radius = getattr(protector, "safe_r", 0.0)
            if base_radius <= 0.0:
                base_radius = getattr(protector, "obs_radius", getattr(protector, "dp", 0.0))
            wing_line = self.protector_wings[idx]

            if base_radius <= 0.0:
                wing_line.set_data([], [])
                continue

            heading = float(getattr(protector, "h", 0.0))
            dir_x = np.cos(heading)
            dir_y = np.sin(heading)
            norm_x = -dir_y
            norm_y = dir_x

            span = base_radius
            cx = protector.x
            cy = protector.y

            wing_line.set_data(
                [cx - norm_x * span, cx + norm_x * span],
                [cy - norm_y * span, cy + norm_y * span],
            )

    def _capture_frame(self):
        if not self.record or self.video_writer is None:
            return
        canvas = self.fig.canvas
        buf = canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8)
        width, height = canvas.get_width_height()
        pixel_count = frame.size // 4
        if width * height != pixel_count:
            height = int(round(np.sqrt(pixel_count)))
            width = pixel_count // height if height else 0
        if width * height != pixel_count or width == 0 or height == 0:
            return
        frame = frame[: width * height * 4].reshape((height, width, 4))
        frame = frame[..., :3]
        self.video_writer.append_data(frame)

    def __call__(self, step, env=None):
        env = env or self.env
        xs = [agent.x for agent in env.uav_list]
        ys = [agent.y for agent in env.uav_list]
        self._set_offsets(self.uav_scatter, xs, ys)

        tgt_xs = [agent.x for agent in env.target_list if not getattr(agent, "captured", False)]
        tgt_ys = [agent.y for agent in env.target_list if not getattr(agent, "captured", False)]
        self._set_offsets(self.target_scatter, tgt_xs, tgt_ys)

        prot_xs = [agent.x for agent in env.protector_list]
        prot_ys = [agent.y for agent in env.protector_list]
        self._set_offsets(self.protector_scatter, prot_xs, prot_ys)

        for patch, uav in zip(self.uav_ranges, env.uav_list):
            patch.center = (uav.x, uav.y)
            patch.set_radius(getattr(uav, "dp", patch.radius))

        self._update_trails()
        self._update_protector_wings(env)

        coverage = env.covered_target_num[-1] if env.covered_target_num else 0
        rate = coverage / max(env.m_targets, 1) * 100.0
        self.coverage_text.set_text(
            f"Step: {step + 1}\n\u8986\u76d6\u76ee\u6807\uff1a{coverage}\n\u8986\u76d6\u7387\uff1a{rate:.1f}%"
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
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title(name)
    plt.grid(True)
    os.makedirs(config["save_dir"], exist_ok=True)
    plt.savefig(os.path.join(config["save_dir"], f"{name}.png"))
    plt.close()


__all__ = ["LiveRenderer", "plot_reward_curve"]

