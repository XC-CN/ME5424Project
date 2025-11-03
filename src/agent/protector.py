import numpy as np
from math import pi, cos, sin, atan2
from typing import List, Optional, Tuple


class Protector:
    """
    Trainable protector (hen) agent. Mirrors the structure of UAV while exposing
    observations, rewards and discrete actions that can be consumed by MAAC style policies.
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
        safe_radius: float,
        action_dim: int = 12,
        obs_radius: Optional[float] = None,
        max_uav: int = 1,
        max_target: int = 1,
        max_protector: int = 1,
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
        self.safe_r = safe_radius
        self.obs_radius = obs_radius if obs_radius and obs_radius > 0 else safe_radius

        self.max_uav = max(0, max_uav)
        self.max_target = max(0, max_target)
        # exclude the current protector itself when building the observation
        self.max_other_protectors = max(0, max_protector - 1)

        self.world_bounds: Tuple[float, float] = (1.0, 1.0)

        self.obs = np.zeros(self.state_dim(), dtype=np.float32)
        self.reward = 0.0
        self.raw_reward = {
            "protect_reward": 0.0,
            "block_reward": 0.0,
            "failure_penalty": 0.0,
        }

    # ------------------------------------------------------------------
    # geometry helpers
    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    def discrete_action(self, action_idx: int) -> float:
        """
        Map discrete action index into heading delta.
        """
        action_idx = int(np.clip(action_idx, 0, self.Na - 1))
        na = action_idx + 1
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: Optional[int]) -> Tuple[float, float, float]:
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

    def set_world_bounds(self, x_max: float, y_max: float) -> None:
        self.world_bounds = (x_max, y_max)

    def clamp_inside(self, x_max: float, y_max: float) -> None:
        self.set_world_bounds(x_max, y_max)
        self._clamp_inside()

    # ------------------------------------------------------------------
    # observation & reward bookkeeping
    def reset_reward(self) -> None:
        self.reward = 0.0
        for key in self.raw_reward:
            self.raw_reward[key] = 0.0

    def state_dim(self) -> int:
        # self features + neighbours (dx, dy, distance) for each entity
        other_p = self.max_other_protectors * 3
        uavs = self.max_uav * 3
        targets = self.max_target * 3
        return 4 + other_p + uavs + targets

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
        for entity in sorted(
            entities, key=lambda e: self.distance(self.x, self.y, e.x, e.y)
        ):
            if skip_self and getattr(entity, "id", None) == self.id:
                continue
            dx = (entity.x - self.x) / max(self.obs_radius, 1e-6)
            dy = (entity.y - self.y) / max(self.obs_radius, 1e-6)
            dist = float(np.hypot(dx, dy))
            features.extend([dx, dy, dist])
            if len(features) // 3 >= max_count:
                break
        # pad to fixed size
        while len(features) // 3 < max_count:
            features.extend([0.0, 0.0, 0.0])
        return features[: max_count * 3]

    def build_observation(
        self,
        uav_list: List,
        target_list: List,
        protector_list: List,
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
        target_feat = np.array(
            self._encode_entities(target_list, self.max_target), dtype=np.float32
        )
        protector_feat = np.array(
            self._encode_entities(protector_list, self.max_other_protectors, skip_self=True),
            dtype=np.float32,
        )
        self.obs = np.concatenate((self_features, protector_feat, uav_feat, target_feat))

    # ------------------------------------------------------------------
    # blocking helpers
    def heading_to(self, x: float, y: float) -> float:
        return atan2(y - self.y, x - self.x)


__all__ = ["Protector"]
