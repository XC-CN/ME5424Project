import os.path
from math import e
from utils.data_util import clip_and_normalize
from agent.uav import UAV
from agent.target import Target
from agent.protector import Protector
import numpy as np
from math import pi
import random
from typing import List


class Environment:
    def __init__(self, n_uav: int, m_targets: int, n_protectors: int, x_max: float, y_max: float, na: int):
        """
        :param n_uav: scalar
        :param m_targets: scalar
        :param x_max: scalar
        :param y_max: scalar
        :param na: scalar
        """
        # size of the environment
        self.x_max = x_max
        self.y_max = y_max

        # dim of action space and state space
        # communication(4 scalar, a), observation(4 scalar), boundary and state information(2 scalar, a)
        # self.state_dim = (4 + na) + 4 + (2 + na)
        self.state_dim = (4 + 1) + 4 + (2 + 1)
        self.action_dim = na

        # agents parameters in the environments
        self.n_uav = n_uav
        self.m_targets = m_targets
        self.n_protectors = n_protectors
        # self.uav_safe_radius = None
        self.protector_safe_radius = None

        # agents
        self.uav_list = []
        self.target_list = []
        self.protector_list = []

        # position of uav, target and protector
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': [], 'all_prot_xs': [], 'all_prot_ys': []}

        # coverage rate of target
        self.covered_target_num = []

    def __reset(self, t_v_max, t_h_max, p_v_max, p_h_max, p_safe,
                u_v_max, u_h_max, na, dc, dp, dt, init_x, init_y,
                t_capture=None, p_action_dim=None, t_action_dim=None,
                p_obs_radius=None, t_obs_radius=None):
        """
        reset the location for all uav_s at (init_x, init_y)
        reset the store position to empty
        :return: should be the initial states !!!!
        """
        if isinstance(init_x, List) and isinstance(init_y, List):
            self.uav_list = [UAV(init_x[i],
                                 init_y[i],
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt) for i in range(self.n_uav)]
        elif not isinstance(init_x, List) and not isinstance(init_y, List):
            self.uav_list = [UAV(init_x,
                                 init_y,
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt) for _ in range(self.n_uav)]
        elif isinstance(init_x, List):
            self.uav_list = [UAV(init_x[i],
                                 init_y,
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt) for i in range(self.n_uav)]
        elif isinstance(init_y, List):
            self.uav_list = [UAV(init_x,
                                 init_y[i],
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt) for i in range(self.n_uav)]
        else:
            print("wrong init position")
        for uav in self.uav_list:
            if hasattr(uav, "set_world_bounds"):
                uav.set_world_bounds(self.x_max, self.y_max)

        target_action_dim = t_action_dim if t_action_dim is not None else na
        protector_action_dim = p_action_dim if p_action_dim is not None else na
        target_obs_radius = t_obs_radius if t_obs_radius is not None else dp
        protector_obs_radius = p_obs_radius if p_obs_radius is not None else p_safe
        capture_radius = t_capture if t_capture is not None else dp

        self.target_list = []
        for idx in range(self.m_targets):
            tx = random.uniform(0, self.x_max)
            ty = random.uniform(0, self.y_max)
            th = random.uniform(-pi, pi)
            target = Target(agent_id=idx,
                            x0=tx,
                            y0=ty,
                            h0=th,
                            v_max=t_v_max,
                            h_max=t_h_max,
                            dt=dt,
                            capture_radius=capture_radius,
                            action_dim=target_action_dim,
                            obs_radius=target_obs_radius,
                            max_uav=self.n_uav,
                            max_protector=self.n_protectors,
                            max_target=self.m_targets)
            target.set_world_bounds(self.x_max, self.y_max)
            self.target_list.append(target)

        self.protector_list = []
        for idx in range(self.n_protectors):
            px = random.uniform(0, self.x_max)
            py = random.uniform(0, self.y_max)
            ph = random.uniform(-pi, pi)
            protector = Protector(agent_id=idx,
                                  x0=px,
                                  y0=py,
                                  h0=ph,
                                  v_max=p_v_max,
                                  h_max=p_h_max,
                                  dt=dt,
                                  safe_radius=p_safe,
                                  action_dim=protector_action_dim,
                                  obs_radius=protector_obs_radius,
                                  max_uav=self.n_uav,
                                  max_target=self.m_targets,
                                  max_protector=self.n_protectors)
            protector.set_world_bounds(self.x_max, self.y_max)
            self.protector_list.append(protector)

        for target in self.target_list:
            target.last_min_protector_dist = None
            target.last_min_uav_dist = None
            target.last_step_speed = 0.0
            target.stagnant_steps = 0
            target.last_positions = []
            target.circular_motion_steps = 0
        for protector in self.protector_list:
            protector.last_min_target_dist = None
            protector.last_min_uav_dist = None
            protector.last_step_speed = 0.0
            protector.stagnant_steps = 0
            protector.last_positions = []
            protector.circular_motion_steps = 0

        for protector in self.protector_list:
            protector.build_observation(self.uav_list, self.target_list, self.protector_list)
        for target in self.target_list:
            target.build_observation(self.uav_list, self.protector_list, self.target_list)

        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': [], 'all_protector_xs': [], 'all_protector_ys': []}
        self.covered_target_num = []
        self.step_i = 0  # 姝ヨ鏁板櫒

    def reset(self, config):
        # self.__reset(t_v_max=config["target"]["v_max"],
        #              t_h_max=pi / float(config["target"]["h_max"]),
        #              u_v_max=config["uav"]["v_max"],
        #              u_h_max=pi / float(config["uav"]["h_max"]),
        #              na=config["environment"]["na"],
        #              dc=config["uav"]["dc"],
        #              dp=config["uav"]["dp"],
        #              dt=config["uav"]["dt"],
        #              init_x=config['environment']['x_max']/2, init_y=config['environment']['y_max']/2)
        self.__reset(t_v_max=config["target"]["v_max"],
                     t_h_max=pi / float(config["target"]["h_max"]),
                     p_v_max=config["protector"]["v_max"],
                     p_h_max=pi / float(config["protector"]["h_max"]),
                     p_safe=config["protector"]["safe_radius"],
                     u_v_max=config["uav"]["v_max"],
                     u_h_max=pi / float(config["uav"]["h_max"]),
                     na=config["environment"]["na"],
                     dc=config["uav"]["dc"],
                     dp=config["uav"]["dp"],
                     dt=config["uav"]["dt"],
                     init_x=config['environment']['x_max']/2,
                     init_y=config['environment']['y_max']/2,
                     t_capture=config["target"].get("capture_radius"),
                     p_action_dim=config["protector"].get("na"),
                     t_action_dim=config["target"].get("na"),
                     p_obs_radius=config["protector"].get("obs_radius"),
                     t_obs_radius=config["target"].get("obs_radius"))

    def get_states(self):
        """
        Get the observation of all agent groups.
        :return: dict(role -> List[np.ndarray])
        """
        return {
            'uav': [uav.get_local_state() for uav in self.uav_list],
            'protector': [protector.get_local_state() for protector in self.protector_list],
            'target': [target.get_local_state() for target in self.target_list]
        }

    def step(self, config, pmi, uav_actions, protector_actions=None, target_actions=None):
        """
        state transfer functions
        :param config:
        :param pmi: PMI network
        :param uav_actions: {0,1,...,Na - 1}
        :return: states, rewards, covered_target_count, done
        """
        # update the position of targets
        # 宸叉崟鑾风洰鏍囦笉鍐嶆洿鏂颁綅缃?
        for i, target in enumerate(self.target_list):
            if not getattr(target, 'captured', False):
                t_action = None
                if target_actions is not None and i < len(target_actions):
                    t_action = target_actions[i]
                target.update_position(t_action)
        # 鏇存�?UAV
        for i, uav in enumerate(self.uav_list):
            action = uav_actions[i] if uav_actions is not None and i < len(uav_actions) else None
            uav.update_position(action)
        # 鏇存柊淇濇姢�?
        for i, prot in enumerate(self.protector_list):
            p_action = None
            if protector_actions is not None and i < len(protector_actions):
                p_action = protector_actions[i]
            prot.update_position(p_action)
            prot.clamp_inside(self.x_max, self.y_max)

        for protector in self.protector_list:
            protector.build_observation(self.uav_list, self.target_list, self.protector_list)
        for target in self.target_list:
            target.build_observation(self.uav_list, self.protector_list, self.target_list)

        # 纰版挒寮瑰紑鏁堟灉锛歎AV 纰板埌淇濇姢鑰呮墜鑷傚悗琚墿鐞嗘帹�?
        protector_cfg = config.get('protector', {})
        kb = protector_cfg.get('knockback', 0.0)
        arm_th = protector_cfg.get('arm_thickness', 0.0)
        lock_base = int(protector_cfg.get('heading_lock_duration', 0))
        if kb > 0 and arm_th > 0:
            # 浣跨敤涓婁竴甯у潗鏍囦及璁′繚鎶よ€呰繍鍔ㄦ柟�?
            prev_idx = max(0, len(self.position['all_protector_xs']) - 1)
            prev_xs = self.position['all_protector_xs'][prev_idx] if prev_idx < len(self.position['all_protector_xs']) else []
            prev_ys = self.position['all_protector_ys'][prev_idx] if prev_idx < len(self.position['all_protector_ys']) else []

            def closest_point_on_segment(px, py, x1, y1, x2, y2):
                vx, vy = x2 - x1, y2 - y1
                seg_len2 = vx * vx + vy * vy
                if seg_len2 <= 1e-12:
                    return x1, y1, np.hypot(px - x1, py - y1)
                t = ((px - x1) * vx + (py - y1) * vy) / seg_len2
                t = max(0.0, min(1.0, t))
                cx, cy = x1 + t * vx, y1 + t * vy
                return cx, cy, np.hypot(px - cx, py - cy)

            for i, prot in enumerate(self.protector_list):
                cx, cy = prot.x, prot.y
                if i < len(prev_xs):
                    vx = cx - prev_xs[i]
                    vy = cy - prev_ys[i]
                    speed = np.hypot(vx, vy)
                else:
                    vx = vy = 0.0
                    speed = 0.0

                # 娉曞悜锛堝瀭鐩翠簬杩愬姩鏂瑰悜锛夛紱闈欐鏃剁敤鏈濆悜鐨勬硶鍚?
                if speed > 1e-8:
                    nx, ny = -vy / speed, vx / speed
                else:
                    h = getattr(prot, 'h', 0.0)
                    nx, ny = -np.sin(h), np.cos(h)

                L = getattr(prot, 'safe_r', 0.0)  # 鎵嬭噦鍗婇暱�?
                # 涓ゆ潯鑷傜殑绔�?
                x1, y1 = cx - nx * L, cy - ny * L  # 鍚庤噦绔偣
                x2, y2 = cx + nx * L, cy + ny * L  # 鍓嶈噦绔偣

                for uav in self.uav_list:
                    # 鍒颁袱鏉¤噦鐨勬渶杩戠偣涓庤窛�?
                    cfx, cfy, d_front = closest_point_on_segment(uav.x, uav.y, cx, cy, x2, y2)
                    crx, cry, d_rear  = closest_point_on_segment(uav.x, uav.y, cx, cy, x1, y1)
                    if d_front < d_rear:
                        cxn, cyn, dmin = cfx, cfy, d_front
                    else:
                        cxn, cyn, dmin = crx, cry, d_rear
                    if dmin < arm_th and dmin > 1e-6:
                        # 浠庢渶杩戠偣鎸囧�?UAV 鐨勬硶鍚?
                        ux = (uav.x - cxn) / dmin
                        uy = (uav.y - cyn) / dmin
                        knockback_angle = np.arctan2(uy, ux)
                        collision_intensity = 1.0 - (dmin / arm_th)
                        kb_factor = 1.0 + (kb / 1000.0)
                        lock_duration = 0
                        if lock_base > 0:
                            lock_duration = max(1, int(lock_base * (0.5 + collision_intensity) * kb_factor))
                        push = min(kb, arm_th - dmin)  # 涓嶈秴杩?knockback
                        uav.x += ux * push
                        uav.y += uy * push
                        uav.x = np.clip(uav.x, 0, self.x_max)
                        uav.y = np.clip(uav.y, 0, self.y_max)
                        if hasattr(uav, "apply_knockback"):
                            uav.apply_knockback(knockback_angle, lock_duration)

        # 鐩爣鎹曡幏妫€�?
        for t_idx, target in enumerate(self.target_list):
            if getattr(target, 'captured', False):
                continue
            for uav in self.uav_list:
                cap_r = config.get("target", {}).get("capture_radius", uav.dp)
                dx = uav.x - target.x
                dy = uav.y - target.y
                if np.hypot(dx, dy) <= cap_r:
                    target.captured = True
                    target.captured_step = self.step_i
                    if hasattr(uav, "captured_targets_count"):
                        uav.captured_targets_count += 1
                    break

        all_targets_captured = bool(self.target_list) and all(getattr(target, 'captured', False) for target in self.target_list)

        # UAV 瑙傛祴涓庨€氫俊
        for uav in self.uav_list:
            uav.observe_target(self.target_list)
            uav.observe_protector(self.protector_list)
            uav.observe_uav(self.uav_list)

        reward_summary = self.calculate_rewards(config=config, pmi=pmi)
        uav_summary = reward_summary["uav"]
        protector_summary = reward_summary["protector"]
        target_summary = reward_summary["target"]
        next_states = self.get_states()

        covered_targets = self.calculate_covered_target()
        self.covered_target_num.append(covered_targets)

        # trace the position matrix
        target_xs, target_ys = self.__get_all_target_position()
        self.position['all_target_xs'].append(target_xs)
        self.position['all_target_ys'].append(target_ys)
        uav_xs, uav_ys = self.__get_all_uav_position()
        self.position['all_uav_xs'].append(uav_xs)
        self.position['all_uav_ys'].append(uav_ys)
        prot_xs, prot_ys = self.__get_all_protector_position()
        self.position['all_protector_xs'].append(prot_xs)
        self.position['all_protector_ys'].append(prot_ys)
        # 姝ラ€掑锛堢粺涓€鏃舵満�?
        self.step_i += 1

        reward = {
            'uav': uav_summary,
            'protector': protector_summary,
            'target': target_summary
        }

        return next_states, reward, covered_targets, all_targets_captured

    def __get_all_uav_position(self) -> (List[float], List[float]):
        """
        :return: all the position of the uav through this epoch
        """
        uav_xs = []
        uav_ys = []
        for uav in self.uav_list:
            uav_xs.append(uav.x)
            uav_ys.append(uav.y)
        return uav_xs, uav_ys

    def __get_all_target_position(self) -> (List[float], List[float]):
        """
        :return: all the position of the targets through this epoch
        """
        target_xs = []
        target_ys = []
        for target in self.target_list:
            target_xs.append(target.x)
            target_ys.append(target.y)
        return target_xs, target_ys
    
    def __get_all_protector_position(self) -> (List[float], List[float]):
        """
        :return: all the position of the targets through this epoch
        """
        prot_xs = []
        prot_ys = []
        for protector in self.protector_list:
            prot_xs.append(protector.x)
            prot_ys.append(protector.y)
        return prot_xs, prot_ys

    def get_uav_and_target_position(self) -> (List[float], List[float], List[float], List[float]):
        """
        :return: both the uav and the target position matrix
        """
        return (self.position['all_uav_xs'], self.position['all_uav_ys'],
                self.position['all_target_xs'], self.position['all_target_ys'])

    def calculate_rewards(self, config, pmi):
        # ---------------- UAV reward components ----------------
        target_tracking_rewards = []
        boundary_punishments = []
        duplicate_tracking_punishments = []
        protector_punishments = []
        for uav in self.uav_list:
            (target_tracking_reward,
             boundary_punishment,
             duplicate_tracking_punishment,
             protector_punishment) = uav.calculate_raw_reward(self.uav_list,
                                                              self.target_list,
                                                              self.protector_list,
                                                              self.x_max,
                                                              self.y_max)

            target_tracking_reward = clip_and_normalize(
                target_tracking_reward,
                0,
                2 * config['environment']['m_targets'],
                0)
            duplicate_tracking_punishment = clip_and_normalize(
                duplicate_tracking_punishment,
                -e / 2 * config['environment']['n_uav'],
                0,
                -1)
            boundary_punishment = clip_and_normalize(boundary_punishment, -1 / 2, 0, -1)
            protector_punishment = clip_and_normalize(
                protector_punishment,
                -e / 2 * max(config['environment']['n_protectors'], 1),
                0,
                -1)

            target_tracking_rewards.append(target_tracking_reward)
            boundary_punishments.append(boundary_punishment)
            duplicate_tracking_punishments.append(duplicate_tracking_punishment)
            protector_punishments.append(protector_punishment)

            uav.raw_reward = (config["uav"]["alpha"] * target_tracking_reward +
                              config["uav"]["beta"] * boundary_punishment +
                              config["uav"]["gamma"] * duplicate_tracking_punishment +
                              config["uav"]["omega"] * protector_punishment)

        uav_rewards = []
        for uav in self.uav_list:
            reward = uav.calculate_cooperative_reward(self.uav_list, pmi, config['cooperative'])
            uav.reward = clip_and_normalize(reward, -1, 1)
            uav_rewards.append(uav.reward)

        # ---------------- Protector reward components ----------------
        protector_cfg = config.get("protector", {})
        p_alpha = protector_cfg.get("alpha", 0.5)
        p_beta = protector_cfg.get("beta", 1.0)
        p_gamma = protector_cfg.get("gamma", 10.0)
        default_safe_radius = protector_cfg.get("safe_radius", 0.0)
        p_approach = protector_cfg.get("approach_bonus_weight", 0.0)
        p_retreat = protector_cfg.get("retreat_bonus_weight", 0.0)
        p_move_penalty = protector_cfg.get("movement_penalty_weight", 0.0)
        p_min_speed = protector_cfg.get("movement_min_speed", 0.0)
        p_stagnate_steps = int(protector_cfg.get("stagnate_steps", 5))
        p_circular_penalty = protector_cfg.get("circular_motion_penalty", 0.5)  # 转圈惩罚权重
        p_position_change_threshold = protector_cfg.get("position_change_threshold", 50.0)  # 位置变化阈值

        failure_flag = any(target.captured for target in self.target_list)

        protector_protect = []
        protector_block = []
        protector_failure = []
        protector_rewards = []
        protector_approach_bonus = []
        protector_retreat_bonus = []
        protector_movement_penalty = []

        for protector in self.protector_list:
            protector.reset_reward()
            safe_radius = getattr(protector, "safe_r", default_safe_radius)
            radius_norm = max(safe_radius, 1e-6)

            protect_score = 0.0
            block_score = 0.0

            min_target_dist = None
            for target in self.target_list:
                if target.captured:
                    continue
                dist_pt = Protector.distance(protector.x, protector.y, target.x, target.y)
                if safe_radius > 0 and dist_pt <= safe_radius:
                    protect_score += max(0.0, 1.0 - dist_pt / radius_norm)
                if min_target_dist is None or dist_pt < min_target_dist:
                    min_target_dist = dist_pt

                for uav in self.uav_list:
                    block_score += self._blocking_score(protector, target, uav, safe_radius)

            min_uav_dist = None
            if self.uav_list:
                min_uav_dist = min(
                    Protector.distance(protector.x, protector.y, uav.x, uav.y)
                    for uav in self.uav_list
                )

            approach_bonus = 0.0
            if min_target_dist is not None:
                if protector.last_min_target_dist is not None:
                    delta = protector.last_min_target_dist - min_target_dist
                    if delta > 0:
                        approach_bonus = p_approach * delta
                protector.last_min_target_dist = min_target_dist

            retreat_bonus = 0.0
            if min_uav_dist is not None:
                if protector.last_min_uav_dist is not None:
                    delta = min_uav_dist - protector.last_min_uav_dist
                    if delta > 0:
                        retreat_bonus = p_retreat * delta
                protector.last_min_uav_dist = min_uav_dist

            # 改进的停滞检测：基于位置变化而非速度
            position_change = 0.0
            if len(protector.last_positions) >= 5:
                # 计算最近5步的总位移
                start_pos = protector.last_positions[0]
                end_pos = protector.last_positions[-1]
                position_change = Protector.distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
            
            # 转圈检测：位置变化小但速度不为零
            circular_penalty = 0.0
            if len(protector.last_positions) >= 8:
                # 计算最近8步的总位移
                start_pos = protector.last_positions[0]
                end_pos = protector.last_positions[-1]
                total_displacement = Protector.distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
                # 计算路径长度（各步位移之和）
                path_length = 0.0
                for i in range(len(protector.last_positions) - 1):
                    path_length += Protector.distance(
                        protector.last_positions[i][0], protector.last_positions[i][1],
                        protector.last_positions[i+1][0], protector.last_positions[i+1][1]
                    )
                # 如果路径长度远大于总位移，说明在转圈
                if path_length > 0 and total_displacement < p_position_change_threshold and path_length > 3 * p_position_change_threshold:
                    protector.circular_motion_steps += 1
                    if protector.circular_motion_steps >= 3:  # 连续3次检测到转圈
                        circular_penalty = -p_circular_penalty * (protector.circular_motion_steps / 3.0)
                else:
                    protector.circular_motion_steps = max(0, protector.circular_motion_steps - 1)
            
            movement_penalty = 0.0
            if p_min_speed > 0:
                # 基于位置变化而非速度
                if position_change < p_position_change_threshold:
                    protector.stagnant_steps += 1
                else:
                    protector.stagnant_steps = 0
                if protector.stagnant_steps >= p_stagnate_steps and p_move_penalty > 0:
                    movement_penalty = -p_move_penalty
            else:
                protector.stagnant_steps = 0

            failure_penalty = -1.0 if failure_flag else 0.0

            protector.raw_reward["protect_reward"] = protect_score
            protector.raw_reward["block_reward"] = block_score
            protector.raw_reward["failure_penalty"] = failure_penalty
            protector.raw_reward["approach_bonus"] = approach_bonus
            protector.raw_reward["retreat_bonus"] = retreat_bonus
            protector.raw_reward["movement_penalty"] = movement_penalty
            protector.raw_reward["circular_penalty"] = circular_penalty

            total_protector_reward = (
                p_alpha * protect_score
                + p_beta * block_score
                + p_gamma * failure_penalty
                + approach_bonus
                + retreat_bonus
                + movement_penalty
                + circular_penalty
            )
            protector.reward = total_protector_reward

            protector_protect.append(protect_score)
            protector_block.append(block_score)
            protector_failure.append(failure_penalty)
            protector_rewards.append(total_protector_reward)
            protector_approach_bonus.append(approach_bonus)
            protector_retreat_bonus.append(retreat_bonus)
            protector_movement_penalty.append(movement_penalty)

        # ---------------- Target reward components ----------------
        target_cfg = config.get("target", {})
        t_alpha = target_cfg.get("alpha", 0.3)
        t_beta = target_cfg.get("beta", 0.5)
        t_gamma = target_cfg.get("gamma", 100.0)
        t_approach = target_cfg.get("approach_bonus_weight", 0.0)
        t_escape = target_cfg.get("escape_bonus_weight", 0.0)
        t_move_penalty = target_cfg.get("movement_penalty_weight", 0.0)
        t_min_speed = target_cfg.get("movement_min_speed", 0.0)
        t_stagnate_steps = int(target_cfg.get("stagnate_steps", 5))
        t_circular_penalty = target_cfg.get("circular_motion_penalty", 0.5)  # 转圈惩罚权重
        t_position_change_threshold = target_cfg.get("position_change_threshold", 50.0)  # 位置变化阈值

        target_safety = []
        target_danger = []
        target_capture = []
        target_rewards = []
        target_approach_bonus = []
        target_escape_bonus = []
        target_movement_penalty = []

        max_safe_radius = max(
            (getattr(p, "safe_r", default_safe_radius) for p in self.protector_list),
            default=default_safe_radius
        )
        safe_norm = max(max_safe_radius, 1e-6)

        for target in self.target_list:
            target.reset_reward()

            # safety reward relative to nearest protector
            safety_score = 0.0
            min_dist_protector = None
            if self.protector_list:
                min_dist_protector = min(
                    Protector.distance(protector.x, protector.y, target.x, target.y)
                    for protector in self.protector_list
                )
                if max_safe_radius > 0 and min_dist_protector <= max_safe_radius:
                    safety_score = max(0.0, 1.0 - min_dist_protector / safe_norm)

            danger_score = 0.0
            min_dist_uav = None
            if self.uav_list:
                min_dist_uav = min(
                    Target.distance(target.x, target.y, uav.x, uav.y)
                    for uav in self.uav_list
                )
                danger_score = 1.0 / (min_dist_uav + 1.0)

            approach_bonus = 0.0
            if min_dist_protector is not None:
                if target.last_min_protector_dist is not None:
                    delta = target.last_min_protector_dist - min_dist_protector
                    if delta > 0:
                        approach_bonus = t_approach * delta
                target.last_min_protector_dist = min_dist_protector

            escape_bonus = 0.0
            if min_dist_uav is not None:
                if target.last_min_uav_dist is not None:
                    delta = min_dist_uav - target.last_min_uav_dist
                    if delta > 0:
                        escape_bonus = t_escape * delta
                target.last_min_uav_dist = min_dist_uav

            # 改进的停滞检测：基于位置变化而非速度
            position_change = 0.0
            if len(target.last_positions) >= 5:
                # 计算最近5步的总位移
                start_pos = target.last_positions[0]
                end_pos = target.last_positions[-1]
                position_change = Target.distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
            
            # 转圈检测：位置变化小但速度不为零
            circular_penalty = 0.0
            if len(target.last_positions) >= 8:
                # 计算最近8步的总位移
                start_pos = target.last_positions[0]
                end_pos = target.last_positions[-1]
                total_displacement = Target.distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
                # 计算路径长度（各步位移之和）
                path_length = 0.0
                for i in range(len(target.last_positions) - 1):
                    path_length += Target.distance(
                        target.last_positions[i][0], target.last_positions[i][1],
                        target.last_positions[i+1][0], target.last_positions[i+1][1]
                    )
                # 如果路径长度远大于总位移，说明在转圈
                if path_length > 0 and total_displacement < t_position_change_threshold and path_length > 3 * t_position_change_threshold:
                    target.circular_motion_steps += 1
                    if target.circular_motion_steps >= 3:  # 连续3次检测到转圈
                        circular_penalty = -t_circular_penalty * (target.circular_motion_steps / 3.0)
                else:
                    target.circular_motion_steps = max(0, target.circular_motion_steps - 1)
            
            movement_penalty = 0.0
            if t_min_speed > 0:
                # 基于位置变化而非速度
                if position_change < t_position_change_threshold:
                    target.stagnant_steps += 1
                else:
                    target.stagnant_steps = 0
                if target.stagnant_steps >= t_stagnate_steps and t_move_penalty > 0:
                    movement_penalty = -t_move_penalty
            else:
                target.stagnant_steps = 0

            capture_penalty = -1.0 if target.captured else 0.0

            target.raw_reward["safety_reward"] = safety_score
            target.raw_reward["danger_penalty"] = -danger_score
            target.raw_reward["capture_penalty"] = capture_penalty
            target.raw_reward["approach_bonus"] = approach_bonus
            target.raw_reward["escape_bonus"] = escape_bonus
            target.raw_reward["movement_penalty"] = movement_penalty
            target.raw_reward["circular_penalty"] = circular_penalty

            total_target_reward = (
                t_alpha * safety_score
                + t_beta * target.raw_reward["danger_penalty"]
                + t_gamma * capture_penalty
                + approach_bonus
                + escape_bonus
                + movement_penalty
                + circular_penalty
            )
            target.reward = total_target_reward

            target_safety.append(safety_score)
            target_danger.append(target.raw_reward["danger_penalty"])
            target_capture.append(capture_penalty)
            target_rewards.append(total_target_reward)
            target_approach_bonus.append(approach_bonus)
            target_escape_bonus.append(escape_bonus)
            target_movement_penalty.append(movement_penalty)

        reward_summary = {
            "uav": {
                "rewards": uav_rewards,
                "target_tracking": target_tracking_rewards,
                "boundary": boundary_punishments,
                "duplicate": duplicate_tracking_punishments,
                "protector_collision": protector_punishments,
            },
            "protector": {
                "rewards": protector_rewards,
                "protect_reward": protector_protect,
                "block_reward": protector_block,
                "failure_penalty": protector_failure,
                "approach_bonus": protector_approach_bonus,
                "retreat_bonus": protector_retreat_bonus,
                "movement_penalty": protector_movement_penalty,
            },
            "target": {
                "rewards": target_rewards,
                "safety_reward": target_safety,
                "danger_penalty": target_danger,
                "capture_penalty": target_capture,
                "approach_bonus": target_approach_bonus,
                "escape_bonus": target_escape_bonus,
                "movement_penalty": target_movement_penalty,
            }
        }

        return reward_summary

    @staticmethod
    def _blocking_score(protector, target, uav, tolerance):
        if tolerance <= 0:
            tolerance = 0.0
        vec_ut = np.array([target.x - uav.x, target.y - uav.y], dtype=float)
        dist_ut = np.linalg.norm(vec_ut)
        if dist_ut < 1e-6:
            return 0.0
        vec_up = np.array([protector.x - uav.x, protector.y - uav.y], dtype=float)
        proj = float(np.dot(vec_up, vec_ut) / dist_ut)
        if proj <= 0 or proj >= dist_ut:
            return 0.0
        closest_vec = vec_up - (proj / dist_ut) * vec_ut
        perp_dist = np.linalg.norm(closest_vec)
        if tolerance <= 0:
            return 0.0
        if perp_dist > tolerance:
            return 0.0
        return max(0.0, 1.0 - perp_dist / tolerance)

    def save_position(self, save_dir, epoch_i):
        u_xy = np.array([self.position["all_uav_xs"],
                         self.position["all_uav_ys"]]).transpose()  # n_uav * num_steps * 2
        t_xy = np.array([self.position["all_target_xs"],
                         self.position["all_target_ys"]]).transpose()  # m_target * num_steps * 2
        p_xy = np.array([self.position["all_protector_xs"],
                         self.position["all_protector_ys"]]).transpose()  # n_protector * num_steps * 2

        np.savetxt(os.path.join(save_dir, "u_xy", 'u_xy' + str(epoch_i) + '.csv'),
                   u_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
        np.savetxt(os.path.join(save_dir, "t_xy", 't_xy' + str(epoch_i) + '.csv'),
                   t_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
        np.savetxt(os.path.join(save_dir, "p_xy", 'p_xy' + str(epoch_i) + '.csv'),
                   p_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')

    def save_covered_num(self, save_dir, epoch_i):
        covered_target_num_array = np.array(self.covered_target_num).reshape(-1, 1)

        np.savetxt(os.path.join(save_dir, "covered_target_num", 'covered_target_num' + str(epoch_i) + '.csv'),
                   covered_target_num_array, delimiter=',', header='covered_target_num', comments='')

    def calculate_covered_target(self):
        covered_target_num = 0
        for target in self.target_list:
            for uav in self.uav_list:
                if uav.distance(uav.x, uav.y, target.x, target.y) < uav.dp:
                    covered_target_num += 1
                    break
        return covered_target_num



