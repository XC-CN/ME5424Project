import numpy as np
from math import cos, sin, pi, atan2, hypot
from typing import List, Optional, Tuple


class Target:
    """
    Trainable target (chick) agent that can observe the world, take discrete
    actions and keep track of shaped reward components.
    """

    def __init__(
        self,
        agent_id: int,
        x0: float,
        y0: float,
        h0: float,
        v_max: float,
        h_max: float,
        dt: float,
        capture_radius: float,
        action_dim: int = 12,
        obs_radius: Optional[float] = None,
        max_uav: int = 1,
        max_protector: int = 1,
        max_target: int = 1,
    ):
        self.id = agent_id
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max
        self.h_max = h_max
        self.dt = dt

        self.Na = max(1, action_dim)
        self.action = 0
        self.capture_radius = capture_radius
        self.obs_radius = obs_radius if obs_radius and obs_radius > 0 else capture_radius

        self.max_uav = max(0, max_uav)
        self.max_protector = max(0, max_protector)
        # include other targets (excluding itself)
        self.max_other_targets = max(0, max_target - 1)

        self.world_bounds: Tuple[float, float] = (1.0, 1.0)

        self.captured = False
        self.captured_step: Optional[int] = None

        self.obs = np.zeros(self.state_dim(), dtype=np.float32)
        self.reward = 0.0
        self.raw_reward = {
            "safety_reward": 0.0,
            "danger_penalty": 0.0,
            "capture_penalty": 0.0,
            "approach_bonus": 0.0,
            "escape_bonus": 0.0,
            "movement_penalty": 0.0,
        }
        self.last_min_protector_dist: Optional[float] = None
        self.last_min_uav_dist: Optional[float] = None
        self.last_step_speed: float = 0.0
        self.stagnant_steps: int = 0
        self.last_positions: List[Tuple[float, float]] = []  # 用于检测转圈
        self.circular_motion_steps: int = 0  # 转圈步数计数

    # ------------------------------------------------------------------
    def reset_capture(self) -> None:
        self.captured = False
        self.captured_step = None

    def set_world_bounds(self, x_max: float, y_max: float) -> None:
        self.world_bounds = (x_max, y_max)

    def clamp_inside(self, x_max: float, y_max: float) -> None:
        self.set_world_bounds(x_max, y_max)
        self._clamp_inside()

    # ------------------------------------------------------------------
    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    def discrete_action(self, action_idx: int) -> float:
        action_idx = int(np.clip(action_idx, 0, self.Na - 1))
        na = action_idx + 1
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: Optional[int]) -> Tuple[float, float, float]:
        if self.captured:
            return self.x, self.y, self.h
        prev_x, prev_y = self.x, self.y
        if action is not None:
            self.action = int(action)
            heading_delta = self.discrete_action(self.action)
        else:
            heading_delta = float(np.random.uniform(-self.h_max, self.h_max))
        dx = self.dt * self.v_max * cos(self.h)
        dy = self.dt * self.v_max * sin(self.h)
        self.x += dx
        self.y += dy
        self.h += self.dt * heading_delta
        self.h = (self.h + pi) % (2 * pi) - pi
        self._clamp_inside()
        self.last_step_speed = float(hypot(self.x - prev_x, self.y - prev_y))
        # 记录位置历史用于转圈检测（保留最近10步）
        self.last_positions.append((self.x, self.y))
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)
        return self.x, self.y, self.h

    def _clamp_inside(self) -> None:
        x_max, y_max = self.world_bounds
        if self.x < 0:
            self.x = 0
            self.h = pi - self.h
        if self.x > x_max:
            self.x = x_max
            self.h = pi - self.h
        if self.y < 0:
            self.y = 0
            self.h = -self.h
        if self.y > y_max:
            self.y = y_max
            self.h = -self.h

    # ------------------------------------------------------------------
    def reset_reward(self) -> None:
        self.reward = 0.0
        for key in self.raw_reward:
            self.raw_reward[key] = 0.0

    def state_dim(self) -> int:
        uav_feat = self.max_uav * 3
        prot_feat = self.max_protector * 3
        other_targets = self.max_other_targets * 3
        # self features: normalised position + heading (cos/sin)
        return 4 + uav_feat + prot_feat + other_targets

    def get_local_state(self) -> np.ndarray:
        return self.obs.copy()

    def _encode_entities(
        self,
        entities: List,
        max_count: int,
        skip_self: bool = False,
    ) -> List[float]:
        features: List[float] = []
        if max_count <= 0:
            return features
        radius = max(self.obs_radius, 1e-6)
        for entity in sorted(
            entities, key=lambda e: self.distance(self.x, self.y, e.x, e.y)
        ):
            if skip_self and getattr(entity, "id", None) == self.id:
                continue
            dx = (entity.x - self.x) / radius
            dy = (entity.y - self.y) / radius
            dist = float(np.hypot(dx, dy))
            features.extend([dx, dy, dist])
            if len(features) // 3 >= max_count:
                break
        while len(features) // 3 < max_count:
            features.extend([0.0, 0.0, 0.0])
        return features[: max_count * 3]

    def build_observation(
        self,
        uav_list: List,
        protector_list: List,
        target_list: List,
    ) -> None:
        x_max, y_max = self.world_bounds
        self_features = np.array(
            [
                self.x / max(x_max, 1e-6),
                self.y / max(y_max, 1e-6),
                cos(self.h),
                sin(self.h),
            ],
            dtype=np.float32,
        )
        uav_feat = np.array(
            self._encode_entities(uav_list, self.max_uav), dtype=np.float32
        )
        protector_feat = np.array(
            self._encode_entities(protector_list, self.max_protector), dtype=np.float32
        )
        other_target_feat = np.array(
            self._encode_entities(target_list, self.max_other_targets, skip_self=True),
            dtype=np.float32,
        )
        self.obs = np.concatenate(
            (self_features, uav_feat, protector_feat, other_target_feat)
        )

    # ------------------------------------------------------------------
    def heading_to(self, x: float, y: float) -> float:
        return atan2(y - self.y, x - self.x)


__all__ = ["Target"]
