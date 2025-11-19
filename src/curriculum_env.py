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
        # 记录老鹰在最近一次撞到翅膀后的“状态机”，用于实现：冲击 -> 直线后退到远点 -> 绕着母鸡旋转。
        self.eagle_state: str = "idle"  # "idle" / "retreat" / "orbit"
        self.eagle_retreat_target_dist: float = 0.0  # 目标退到的母鸡距离
        self.eagle_retreat_steps: int = 0           # 已经退了多少步
        self.eagle_orbit_steps: int = 0             # 已经绕圈多少步
        self.eagle_orbit_sign: int = 1              # 绕圈方向，+1 或 -1

    def _active_role(self) -> str:
        return "hen"

    @staticmethod
    def _blocking_score_along_segment(
        hen_pos: b2Vec2, tail_pos: b2Vec2, eagle_pos: b2Vec2, arm_span: float
    ) -> float:
        """
        计算母鸡是否有效“挡在”老鹰与小鸡尾端之间的几何得分。

        思路：
        - 以老鹰为原点，指向小鸡尾端的向量为主轴；
        - 将母鸡相对老鹰的位置投影到这条主轴上，要求投影在线段 E→T 之间；
        - 再计算母鸡到这条线段的垂直距离，距离越小、越接近直线，得分越高；
        - arm_span 为“臂展”宽度，垂距超过 arm_span 时视为没有挡住。
        """
        if arm_span <= 0.0:
            return 0.0

        vec_et = np.array(
            [tail_pos.x - eagle_pos.x, tail_pos.y - eagle_pos.y], dtype=float
        )
        dist_et = np.linalg.norm(vec_et)
        if dist_et < 1e-6:
            return 0.0

        vec_eh = np.array(
            [hen_pos.x - eagle_pos.x, hen_pos.y - eagle_pos.y], dtype=float
        )
        # 母鸡在老鹰->尾端这条线段上的投影长度
        proj = float(np.dot(vec_eh, vec_et) / dist_et)
        if proj <= 0.0 or proj >= dist_et:
            # 投影不在 E->T 段上，说明不在两者之间
            return 0.0

        # 母鸡到该主轴的最短向量
        closest_vec = vec_eh - (proj / dist_et) * vec_et
        perp_dist = np.linalg.norm(closest_vec)
        if perp_dist > arm_span:
            return 0.0

        return max(0.0, 1.0 - perp_dist / max(arm_span, 1e-6))

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

        # 1) 挡线得分：母鸡是否挡在老鹰与小鸡尾端之间，带“臂展”宽度
        arm_span = max(self.cfg.block_margin, 1e-3)
        block_score = self._blocking_score_along_segment(
            hen_pos=self.hen.position,
            tail_pos=tail.position,
            eagle_pos=self.eagle.position,
            arm_span=arm_span,
        )

        # 2) 让老鹰远离尾端（避免被抓）
        avoid_catch_score = math.tanh(dist_eagle_tail / self.cfg.world_size)

        # 3) 链条拉伸与尾距惩罚：母鸡既要挡线，又不能把链条拉得过长或过短
        stretch_penalty = self._chain_stretch()
        tail_dist_penalty = abs(
            dist_hen_tail - self.cfg.chain_spacing * self.cfg.chain_links
        )

        # 4) 边界惩罚：母鸡贴近世界边界会被适度惩罚，鼓励在场内中部完成防守
        bound = float(self.cfg.world_size)
        hx, hy = float(self.hen.position.x), float(self.hen.position.y)
        max_abs_coord = max(abs(hx), abs(hy))
        # 当 |x| 或 |y| 超过 0.7 * world_size 时开始线性增加惩罚，最大惩罚为 1.0
        border_margin = 0.7 * bound
        if max_abs_coord <= border_margin or bound <= 0.0:
            border_penalty = 0.0
        else:
            border_penalty = min((max_abs_coord - border_margin) / (bound - border_margin), 1.0)

        reward = (
            0.7 * block_score
            + 0.3 * avoid_catch_score
            - 0.05 * stretch_penalty
            - 0.1 * tail_dist_penalty
            - 0.2 * border_penalty
        )

        caught = dist_eagle_tail < self.cfg.catch_radius
        # 5) 生存奖励：只要小鸡链条尾端尚未被捕获，随着时间推移给予一个很小的正奖励
        if not caught and self.cfg.max_steps > 0:
            survival_bonus = 0.02 * (float(self.step_count) / float(self.cfg.max_steps))
            reward += survival_bonus

        terminated = bool(caught)
        if caught:
            reward -= 5.0

        truncated = self.step_count >= self.cfg.max_steps
        done_flag = terminated or truncated
        return reward, done_flag

    def step(self, action: np.ndarray):
        self._apply_action(self.hen, action, self.cfg.hen_max_speed)
        heuristic_act = self._heuristic_eagle_action()

        # 根据老鹰当前状态机调整其加速度方向：
        # - idle   ：正常追击尾端（带轻微扰动）；
        # - retreat：从母鸡前方直线后退到远点；
        # - orbit  ：在远点附近绕着母鸡旋转一小圈。
        if self.eagle_state == "retreat":
            # 以“远离母鸡”的方向为主，追击方向为辅
            vec_h2e = np.array(
                [self.eagle.position.x - self.hen.position.x,
                 self.eagle.position.y - self.hen.position.y],
                dtype=float,
            )
            norm = np.linalg.norm(vec_h2e)
            if norm > 1e-6:
                away_dir = vec_h2e / norm
                w = 0.9  # 退避阶段更偏向纯后退
                blended = (1.0 - w) * heuristic_act + w * away_dir
                heuristic_act = np.clip(blended, -1.0, 1.0)
            self.eagle_retreat_steps += 1

            # 判断是否退到了目标距离或超过最大退避步数
            dist_h2e = float(norm) if norm > 1e-6 else 0.0
            max_retreat_steps = 20
            if dist_h2e >= self.eagle_retreat_target_dist or self.eagle_retreat_steps >= max_retreat_steps:
                # 进入绕圈阶段：以当前相对位置为切向方向基础
                self.eagle_state = "orbit"
                self.eagle_orbit_steps = 0
                # 根据当前老鹰在母鸡周围的位置，选定一个稳定的绕圈方向（这里固定为逆时针）
                self.eagle_orbit_sign = 1

        elif self.eagle_state == "orbit":
            # 以母鸡为圆心，沿切向方向绕圈：t 为法向的垂线方向
            vec_h2e = np.array(
                [self.eagle.position.x - self.hen.position.x,
                 self.eagle.position.y - self.hen.position.y],
                dtype=float,
            )
            norm = np.linalg.norm(vec_h2e)
            if norm > 1e-6:
                # 垂直方向 (顺/逆时针)
                base_t = np.array([-vec_h2e[1], vec_h2e[0]], dtype=float) / norm
                t = self.eagle_orbit_sign * base_t
                w = 0.7  # 绕圈阶段切向为主
                blended = w * t + (1.0 - w) * heuristic_act
                heuristic_act = np.clip(blended, -1.0, 1.0)
            self.eagle_orbit_steps += 1

            max_orbit_steps = 20
            if self.eagle_orbit_steps >= max_orbit_steps:
                # 绕圈结束，回到正常追击状态
                self.eagle_state = "idle"

        # 根据状态调整好方向后，再真正施加力
        self._apply_action(self.eagle, heuristic_act, self.cfg.eagle_max_speed)

        self.world.Step(self.cfg.dt, 6, 2)
        self._enforce_bounds()
        # 在物理步进之后，根据母鸡的“翅膀”对老鹰进行物理弹回处理：
        # 翅膀定义为一条垂直于“母鸡 -> 小鸡尾端”方向、穿过母鸡位置的线段，
        # 当老鹰从母鸡正面撞上这条翅膀时，会被沿法线方向弹回一段距离，并对速度做镜面反射，
        # 从而形成“冲上去被弹开、再绕着母鸡旋转”的效果。
        tail = self.chicks[-1]
        arm_span = max(self.cfg.block_margin, 0.0)
        if arm_span > 0.0:
            # 轴向向量：母鸡 -> 小鸡尾端（作为翅膀法线方向）
            vec_ht = np.array(
                [tail.position.x - self.hen.position.x, tail.position.y - self.hen.position.y],
                dtype=float,
            )
            dist_ht = np.linalg.norm(vec_ht)
            if dist_ht > 1e-6:
                n = vec_ht / dist_ht  # 法线方向（指向尾端）
                vec_he = np.array(
                    [self.eagle.position.x - self.hen.position.x, self.eagle.position.y - self.hen.position.y],
                    dtype=float,
                )
                # 在法线方向上的投影（>0 表示老鹰位于母鸡朝向尾端的一侧，0<dist<dist_ht 表示在母鸡和尾端之间）
                dist_normal = float(np.dot(vec_he, n))
                lateral_vec = vec_he - dist_normal * n
                radial_dist = float(np.linalg.norm(lateral_vec))
                # 当老鹰位于母鸡与尾端之间，且横向距离落在翅膀长度范围内时，视为撞上翅膀
                if 0.0 < dist_normal < dist_ht and radial_dist <= arm_span:
                    # 当前速度分解到法线与切向方向，进行“镜面反射”+衰减
                    v = np.array(
                        [self.eagle.linearVelocity.x, self.eagle.linearVelocity.y],
                        dtype=float,
                    )
                    vn_mag = float(np.dot(v, n))
                    v_n = vn_mag * n          # 法向分量
                    v_t = v - v_n             # 切向分量（沿翅膀方向）
                    restitution = 0.5         # 反弹系数：法向速度反转并衰减
                    friction = 0.9            # 切向保留系数：保留大部分切向速度，便于绕行
                    v_reflect = -restitution * v_n + friction * v_t
                    self.eagle.linearVelocity = b2Vec2(float(v_reflect[0]), float(v_reflect[1]))

                    # 碰撞到翅膀时：
                    # 1) 进入“退避”状态，目标是从母鸡前方退到一个更远的位置；
                    # 2) 之后自动切换到“绕圈”状态，在远点附近围绕母鸡旋转一小段时间。
                    vec_h2e = np.array(
                        [self.eagle.position.x - self.hen.position.x,
                         self.eagle.position.y - self.hen.position.y],
                        dtype=float,
                    )
                    dist_h2e = float(np.linalg.norm(vec_h2e))
                    extra_retreat = 2.0   # 希望比当前距离再多退 2 个单位长度
                    min_retreat_dist = 3.0
                    target_dist = max(dist_h2e + extra_retreat, min_retreat_dist)
                    self.eagle_retreat_target_dist = target_dist
                    self.eagle_retreat_steps = 0
                    self.eagle_orbit_steps = 0
                    self.eagle_state = "retreat"

                    # 沿 -n 方向仅做极小的位移修正，避免数值上卡在翅膀内部，
                    # 不再将老鹰“瞬移”到数米之外，视觉上主要由速度反弹产生“弹开”效果。
                    knock = min(0.1, 0.1 * arm_span)
                    self.eagle.position = b2Vec2(
                        float(self.eagle.position.x - n[0] * knock),
                        float(self.eagle.position.y - n[1] * knock),
                    )
                    self._enforce_bounds()

        # 除了翅膀带之外，母鸡自身的圆形刚体也应对老鹰产生碰撞反弹：
        # 当老鹰中心与母鸡中心的距离小于半径和时，直接以“母鸡 -> 老鹰”为法线方向进行反弹，
        # 并触发同样的退避 + 绕圈状态机。
        hen_r = float(getattr(self.cfg, "hen_radius", 0.4))
        eagle_r = float(getattr(self.cfg, "eagle_radius", 0.45))
        vec_h2e_body = np.array(
            [self.eagle.position.x - self.hen.position.x,
             self.eagle.position.y - self.hen.position.y],
            dtype=float,
        )
        dist_h2e_body = float(np.linalg.norm(vec_h2e_body))
        if dist_h2e_body > 1e-6 and dist_h2e_body < (hen_r + eagle_r):
            n_body = vec_h2e_body / dist_h2e_body  # 法线方向：母鸡 -> 老鹰
            v = np.array(
                [self.eagle.linearVelocity.x, self.eagle.linearVelocity.y],
                dtype=float,
            )
            vn_mag = float(np.dot(v, n_body))
            v_n = vn_mag * n_body
            v_t = v - v_n
            restitution = 0.5
            friction = 0.9
            v_reflect = -restitution * v_n + friction * v_t
            self.eagle.linearVelocity = b2Vec2(float(v_reflect[0]), float(v_reflect[1]))

            # 同样触发退避 + 绕圈状态机
            extra_retreat = 2.0
            min_retreat_dist = 3.0
            target_dist = max(dist_h2e_body + extra_retreat, min_retreat_dist)
            self.eagle_retreat_target_dist = target_dist
            self.eagle_retreat_steps = 0
            self.eagle_orbit_steps = 0
            self.eagle_state = "retreat"

            knock_body = min(0.1, 0.1 * (hen_r + eagle_r))
            self.eagle.position = b2Vec2(
                float(self.eagle.position.x + n_body[0] * knock_body),
                float(self.eagle.position.y + n_body[1] * knock_body),
            )
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
        device: str | None = "cpu",
    ):
        """
        :param hen_policy_path: 冻结母鸡策略的模型路径（.zip）
        :param config: 物理配置
        :param seed: 随机种子
        :param device: 加载母鸡策略的设备，默认使用 CPU（推荐）。如需显式使用 GPU，可传入 'cuda'。
        """
        super().__init__(config=config, seed=seed)
        resolved_path = Path(hen_policy_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Hen policy path not found: {resolved_path}")
        # Stable-Baselines3 要求 device 为字符串或 torch.device，不能是 None，这里默认使用 CPU。
        load_device = device or "cpu"
        self.hen_model = PPO.load(resolved_path.as_posix(), device=load_device)

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

