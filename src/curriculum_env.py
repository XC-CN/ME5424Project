"""
Curriculum-style Gymnasium environments for the eagle-vs-hen training pipeline.

The design follows a shared Box2D physics core (`BasePhysicsEnv`) and two
specializations:
    - HenTrainingEnv: trains the hen against a heuristic eagle.
    - EagleTrainingEnv: trains the eagle against a frozen hen policy.

Both environments expose the same action space (continuous acceleration in X/Y)
but return perspective-specific observations so that each agent only sees the
world from its own frame. When the hen policy is frozen (stage 2), its actions
are inferred with `model.predict` inside the environment and applied to the
physics step, while the returned observation/reward are for the active learner
(the eagle).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from Box2D import (
    b2CircleShape,
    b2DistanceJointDef,
    b2Vec2,
    b2World,
)
from gymnasium import spaces
from stable_baselines3 import PPO


@dataclass
class PhysicsConfig:
    world_size: float = 20.0  # half side length; world is [-world_size, world_size]^2
    dt: float = 1.0 / 30.0
    max_steps: int = 600
    hen_radius: float = 0.4
    eagle_radius: float = 0.45
    chick_radius: float = 0.25
    chain_links: int = 3
    chain_spacing: float = 1.0
    joint_frequency_hz: float = 4.0
    joint_damping: float = 0.7
    hen_max_speed: float = 6.0
    eagle_max_speed: float = 7.0
    max_force: float = 25.0
    catch_radius: float = 0.7
    block_margin: float = 1.5


class BasePhysicsEnv(gym.Env):
    """
    Shared Box2D simulation for hen/eagle curriculum training.

    Agents control planar accelerations. Box2D handles link dynamics for the
    chicks chained behind the hen via distance joints, adding some inertia the
    hen must account for while moving. Observations are perspective-specific and
    normalized to roughly [-1, 1].
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: PhysicsConfig | None = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or PhysicsConfig()
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Continuous acceleration in x/y, clipped to [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: pos/vel of self, relative vectors to opponent and tail,
        # tail velocity, chain stretch, time fraction.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.world: Optional[b2World] = None
        self.hen = None
        self.eagle = None
        self.chicks = []
        self.step_count = 0

    # --- Gymnasium helpers -------------------------------------------------

    def seed(self, seed: Optional[int] = None) -> None:
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self._build_world()
        self.step_count = 0
        obs = self._get_obs(role=self._active_role())
        return obs, {}

    # --- World construction -------------------------------------------------

    def _build_world(self) -> None:
        self.world = b2World(gravity=(0, 0), doSleep=True)
        spawn_radius = 0.4 * self.cfg.world_size

        hen_pos = self._random_point(radius=spawn_radius)
        eagle_pos = self._random_point(radius=spawn_radius)
        # Avoid pathological overlap at reset.
        while (hen_pos - eagle_pos).length < 2.0:
            eagle_pos = self._random_point(radius=spawn_radius)

        self.hen = self._create_agent(hen_pos, self.cfg.hen_radius)
        self.eagle = self._create_agent(eagle_pos, self.cfg.eagle_radius)
        self.chicks = self._create_chicks(self.hen.position)

    def _create_agent(self, pos: b2Vec2, radius: float):
        body = self.world.CreateDynamicBody(
            position=pos,
            linearDamping=1.2,
            angularDamping=1.2,
        )
        body.CreateFixture(shape=b2CircleShape(radius=radius), density=1.0, friction=0.3)
        return body

    def _create_chicks(self, hen_pos: b2Vec2):
        chicks = []
        prev_body = self.hen
        for link_idx in range(self.cfg.chain_links):
            offset = b2Vec2(-(link_idx + 1) * self.cfg.chain_spacing, 0.0)
            body = self.world.CreateDynamicBody(
                position=hen_pos + offset,
                linearDamping=2.0,
                angularDamping=2.0,
            )
            body.CreateFixture(
                shape=b2CircleShape(radius=self.cfg.chick_radius),
                density=0.3,
                friction=0.5,
            )
            joint_def = b2DistanceJointDef(
                bodyA=prev_body,
                bodyB=body,
                length=self.cfg.chain_spacing,
                frequencyHz=self.cfg.joint_frequency_hz,
                dampingRatio=self.cfg.joint_damping,
            )
            self.world.CreateJoint(joint_def)
            chicks.append(body)
            prev_body = body
        return chicks

    def _random_point(self, radius: float) -> b2Vec2:
        angle = self.rng.uniform(0, 2 * math.pi)
        r = self.rng.uniform(0.1 * radius, radius)
        return b2Vec2(r * math.cos(angle), r * math.sin(angle))

    # --- Simulation helpers -------------------------------------------------

    def _apply_action(self, body, action: np.ndarray, max_speed: float) -> None:
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, -1.0, 1.0)
        force = b2Vec2(float(act[0]) * self.cfg.max_force, float(act[1]) * self.cfg.max_force)
        body.ApplyForceToCenter(force, True)

        # Velocity capping keeps the simulation stable.
        vel = body.linearVelocity
        speed = vel.length
        if speed > max_speed:
            body.linearVelocity = vel * (max_speed / speed)

    def _enforce_bounds(self) -> None:
        bound = self.cfg.world_size
        for body in [self.hen, self.eagle, *self.chicks]:
            pos = body.position
            clamped_x = float(np.clip(pos.x, -bound, bound))
            clamped_y = float(np.clip(pos.y, -bound, bound))
            if clamped_x != pos.x or clamped_y != pos.y:
                body.position = b2Vec2(clamped_x, clamped_y)
                body.linearVelocity = b2Vec2(0.0, 0.0)

    # --- Observations -------------------------------------------------------

    def _normalize_pos(self, vec: b2Vec2) -> np.ndarray:
        return np.array([vec.x, vec.y], dtype=np.float32) / self.cfg.world_size

    def _normalize_vel(self, vec: b2Vec2, max_speed: float) -> np.ndarray:
        return np.array([vec.x, vec.y], dtype=np.float32) / max_speed

    def _chain_stretch(self) -> float:
        """Return average stretch factor of the chain."""
        stretch = 0.0
        prev = self.hen
        for chick in self.chicks:
            dist = (chick.position - prev.position).length
            stretch += dist / max(self.cfg.chain_spacing, 1e-6)
            prev = chick
        return stretch / max(len(self.chicks), 1)

    def _get_obs(self, role: str) -> np.ndarray:
        assert role in ("hen", "eagle"), "role must be 'hen' or 'eagle'"
        agent = self.hen if role == "hen" else self.eagle
        opponent = self.eagle if role == "hen" else self.hen
        agent_max_speed = self.cfg.hen_max_speed if role == "hen" else self.cfg.eagle_max_speed
        opponent_max_speed = self.cfg.eagle_max_speed if role == "hen" else self.cfg.hen_max_speed

        tail = self.chicks[-1]
        rel_opponent = opponent.position - agent.position
        rel_tail = tail.position - agent.position

        obs = np.concatenate(
            [
                self._normalize_pos(agent.position),
                self._normalize_vel(agent.linearVelocity, agent_max_speed),
                self._normalize_pos(rel_opponent),
                self._normalize_pos(rel_tail),
                self._normalize_vel(tail.linearVelocity, agent_max_speed),
                np.array(
                    [
                        self._chain_stretch(),
                        float(self.step_count) / float(self.cfg.max_steps),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    # --- Abstract-ish hooks -------------------------------------------------

    def _active_role(self) -> str:
        raise NotImplementedError

    def _compute_reward_done(self) -> Tuple[float, bool]:
        raise NotImplementedError

    # --- Gymnasium step -----------------------------------------------------

    def step(self, action: np.ndarray):
        raise NotImplementedError


class HenTrainingEnv(BasePhysicsEnv):
    """Stage 1: train the hen against a heuristic eagle."""

    def __init__(self, config: PhysicsConfig | None = None, seed: Optional[int] = None):
        super().__init__(config=config, seed=seed)

    def _active_role(self) -> str:
        return "hen"

    # Heuristic eagle pursues the tail with a bit of lateral drift to simulate flanking.
    def _heuristic_eagle_action(self) -> np.ndarray:
        tail = self.chicks[-1]
        to_tail = tail.position - self.eagle.position
        lateral = b2Vec2(-to_tail.y, to_tail.x)
        if lateral.length > 1e-6:
            lateral.Normalize()
        drift = 0.25 * math.sin(0.1 * self.step_count)
        desired = to_tail + drift * lateral
        if desired.length < 1e-6:
            return np.zeros(2, dtype=np.float32)
        desired.Normalize()
        return np.array([desired.x, desired.y], dtype=np.float32)

    def _compute_reward_done(self) -> Tuple[float, bool]:
        tail = self.chicks[-1]
        dist_eagle_tail = (self.eagle.position - tail.position).length
        dist_hen_tail = (self.hen.position - tail.position).length
        dist_hen_eagle = (self.hen.position - self.eagle.position).length

        reward = 0.6 * math.tanh(dist_eagle_tail / self.cfg.world_size)
        reward += 0.2 * math.tanh(dist_hen_eagle / self.cfg.block_margin)
        reward -= 0.05 * self._chain_stretch()
        reward -= 0.1 * abs(dist_hen_tail - self.cfg.chain_spacing * self.cfg.chain_links)

        caught = dist_eagle_tail < self.cfg.catch_radius
        terminated = bool(caught)
        if caught:
            reward -= 5.0

        truncated = self.step_count >= self.cfg.max_steps
        done_flag = terminated or truncated
        return reward, done_flag

    def step(self, action: np.ndarray):
        self._apply_action(self.hen, action, self.cfg.hen_max_speed)
        heuristic_act = self._heuristic_eagle_action()
        self._apply_action(self.eagle, heuristic_act, self.cfg.eagle_max_speed)

        self.world.Step(self.cfg.dt, 6, 2)
        self._enforce_bounds()
        self.step_count += 1

        reward, done = self._compute_reward_done()
        obs = self._get_obs(role="hen")
        terminated = done and (self.step_count < self.cfg.max_steps)
        truncated = self.step_count >= self.cfg.max_steps

        info = {"dist_to_tail": float((self.eagle.position - self.chicks[-1].position).length)}
        return obs, reward, terminated, truncated, info


class EagleTrainingEnv(BasePhysicsEnv):
    """Stage 2: train the eagle against a frozen hen policy."""

    def __init__(
        self,
        hen_policy_path: str,
        config: PhysicsConfig | None = None,
        seed: Optional[int] = None,
        device: str | None = None,
    ):
        super().__init__(config=config, seed=seed)
        resolved_path = Path(hen_policy_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Hen policy path not found: {resolved_path}")
        self.hen_model = PPO.load(resolved_path.as_posix(), device=device)

    def _active_role(self) -> str:
        return "eagle"

    def _compute_reward_done(self) -> Tuple[float, bool]:
        tail = self.chicks[-1]
        dist_eagle_tail = (self.eagle.position - tail.position).length
        dist_eagle_hen = (self.eagle.position - self.hen.position).length

        reward = 1.0 - math.tanh(dist_eagle_tail / self.cfg.world_size)
        reward -= 0.05 * math.tanh(dist_eagle_hen / self.cfg.block_margin)
        reward -= 0.02 * self._chain_stretch()

        caught = dist_eagle_tail < self.cfg.catch_radius
        terminated = bool(caught)
        if caught:
            reward += 6.0

        truncated = self.step_count >= self.cfg.max_steps
        done_flag = terminated or truncated
        return reward, done_flag

    def step(self, action: np.ndarray):
        # Hen acts using its frozen policy, consuming a hen-perspective observation.
        hen_obs = self._get_obs(role="hen")
        hen_action, _ = self.hen_model.predict(hen_obs, deterministic=True)

        self._apply_action(self.hen, hen_action, self.cfg.hen_max_speed)
        self._apply_action(self.eagle, action, self.cfg.eagle_max_speed)

        self.world.Step(self.cfg.dt, 6, 2)
        self._enforce_bounds()
        self.step_count += 1

        reward, done = self._compute_reward_done()
        obs = self._get_obs(role="eagle")
        terminated = done and (self.step_count < self.cfg.max_steps)
        truncated = self.step_count >= self.cfg.max_steps

        info = {
            "dist_to_tail": float((self.eagle.position - self.chicks[-1].position).length),
            "dist_to_hen": float((self.eagle.position - self.hen.position).length),
        }
        return obs, reward, terminated, truncated, info


def resolve_model_path(path_str: str, default_name: str) -> Path:
    """Utility to resolve a model path with a reasonable default."""
    path = Path(path_str) if path_str else Path("results") / default_name
    if path.is_dir():
        return path / f"{default_name}.zip"
    return path

