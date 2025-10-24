import os.path
from math import e
from utils.data_util import clip_and_normalize
from agent.uav import UAV
from agent.target import TARGET
from agent.protector import PROTECTOR
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

    def __reset(self, t_v_max, t_h_max, p_v_max, p_h_max, p_safe, u_v_max, u_h_max, na, dc, dp, dt, init_x, init_y):
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
        # the initial position of the target is random, having randon headings
        self.target_list = [TARGET(random.uniform(0, self.x_max),
                                   random.uniform(0, self.y_max),
                                   random.uniform(-pi, pi),
                                   random.uniform(-pi / 6, pi / 6),
                                   t_v_max, t_h_max, dt)
                            for _ in range(self.m_targets)]
        self.protector_list = [PROTECTOR(random.uniform(0, self.x_max),
                                         random.uniform(0, self.y_max),
                                         random.uniform(-pi, pi),
                                         0, p_v_max, p_h_max, dt, p_safe) for _ in range(self.n_protectors)]
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': [], 'all_protector_xs': [], 'all_protector_ys': []}
        self.covered_target_num = []
        self.step_i = 0  # 新增：统一的步计数器

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
                     u_v_max=config["uav"]["v_max"],
                     u_h_max=pi / float(config["uav"]["h_max"]),
                     p_v_max=config["protector"]["v_max"],
                     p_h_max=pi / float(config["protector"]["h_max"]),
                     p_safe=config["protector"]["safe_radius"],
                     na=config["environment"]["na"],
                     dc=config["uav"]["dc"],
                     dp=config["uav"]["dp"],
                     dt=config["uav"]["dt"],
                     init_x=config['environment']['x_max']/2, init_y=config['environment']['y_max']/2)

    def get_states(self) -> (List['np.ndarray']):
        """
        get the state of the uav_s
        :return: list of np array, each element is a 1-dim array with size of 12
        """
        uav_states = []
        # collect the overall communication and target observation by each uav
        for uav in self.uav_list:
            uav_states.append(uav.get_local_state())
        return uav_states

    def step(self, config, pmi, actions):
        """
        state transfer functions
        :param config:
        :param pmi: PMI network
        :param actions: {0,1,...,Na - 1}
        :return: states, rewards
        """
        # update the position of targets
        # 已捕获目标不再更新位置
        for i, target in enumerate(self.target_list):
            if not getattr(target, 'captured', False):
                target.update_position(self.x_max, self.y_max)
        # 更新 UAV
        for i, uav in enumerate(self.uav_list):
            uav.update_position(actions[i])
        # 更新保护者
        for i, prot in enumerate(self.protector_list):
            prot.update_position(actions[i])
            prot.clamp_inside(self.x_max, self.y_max)

        # === 新增：UAV 碰到保护者手臂后被弹开 ===
        kb = config.get('protector', {}).get('knockback', 0.0)
        arm_th = config.get('protector', {}).get('arm_thickness', 0.0)
        if kb > 0 and arm_th > 0:
            # 使用上一帧坐标估计保护者运动方向
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

                # 法向（垂直于运动方向）；静止时用朝向的法向
                if speed > 1e-8:
                    nx, ny = -vy / speed, vx / speed
                else:
                    h = getattr(prot, 'h', 0.0)
                    nx, ny = -np.sin(h), np.cos(h)

                L = getattr(prot, 'safe_r', 0.0)  # 手臂半长度
                # 两条臂的端点
                x1, y1 = cx - nx * L, cy - ny * L  # 后臂端点
                x2, y2 = cx + nx * L, cy + ny * L  # 前臂端点

                for uav in self.uav_list:
                    # 到两条臂的最近点与距离
                    cfx, cfy, d_front = closest_point_on_segment(uav.x, uav.y, cx, cy, x2, y2)
                    crx, cry, d_rear  = closest_point_on_segment(uav.x, uav.y, cx, cy, x1, y1)
                    if d_front < d_rear:
                        cxn, cyn, dmin = cfx, cfy, d_front
                    else:
                        cxn, cyn, dmin = crx, cry, d_rear

                    if dmin < arm_th and dmin > 1e-6:
                        # 从最近点指向 UAV 的法向
                        ux = (uav.x - cxn) / dmin
                        uy = (uav.y - cyn) / dmin
                        push = min(kb, arm_th - dmin)  # 不超过 knockback
                        uav.x += ux * push
                        uav.y += uy * push
                        uav.x = np.clip(uav.x, 0, self.x_max)
                        uav.y = np.clip(uav.y, 0, self.y_max)
        # === 新增逻辑结束 ===

        # === 新增：捕捉检测（在观测与记录轨迹之前执行）===
        for t_idx, target in enumerate(self.target_list):
            if getattr(target, 'captured', False):
                continue
            for uav in self.uav_list:
                cap_r = config.get("target", {}).get("capture_radius", uav.dp)
                dx = uav.x - target.x
                dy = uav.y - target.y
                if np.hypot(dx, dy) <= cap_r:
                    target.captured = True
                    target.captured_step = self.step_i  # 使用统一步计数器
                    # 可选：日志验证
                    # print(f"[Capture] Target {t_idx} captured at step {self.step_i}")
                    break
        # === 捕捉检测结束 ===

        # observation and communication
        uav.observe_target(self.target_list)
        uav.observe_protector(self.protector_list)
        uav.observe_uav(self.uav_list)

        (rewards,
         target_tracking_reward,
         boundary_punishment,
         duplicate_tracking_punishment) = self.calculate_rewards(config=config, pmi=pmi)
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
        # 步递增（统一时机）
        self.step_i += 1

        reward = {
            'rewards': rewards,
            'target_tracking_reward': target_tracking_reward,
            'boundary_punishment': boundary_punishment,
            'duplicate_tracking_punishment': duplicate_tracking_punishment
        }

        return next_states, reward, covered_targets

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

    def calculate_rewards(self, config, pmi) -> ([float], float, float, float):
        # raw reward first
        target_tracking_rewards = []
        boundary_punishments = []
        duplicate_tracking_punishments = []
        protector_punishments = []
        for uav in self.uav_list:
            # raw reward for each uav (not clipped)
            (target_tracking_reward,
             boundary_punishment,
             duplicate_tracking_punishment,
             protector_punishment) = uav.calculate_raw_reward(self.uav_list, self.target_list, self.protector_list, self.x_max, self.y_max)

            # clip op
            target_tracking_reward = clip_and_normalize(target_tracking_reward,
                                                        0, 2 * config['environment']['m_targets'], 0)
            duplicate_tracking_punishment = clip_and_normalize(duplicate_tracking_punishment,
                                                               -e / 2 * config['environment']['n_uav'], 0, -1)
            boundary_punishment = clip_and_normalize(boundary_punishment, -1/2, 0, -1)
            protector_punishment = clip_and_normalize(protector_punishment,
                                                      -e / 2 * config['environment']['n_protectors'], 0, -1)

            # append
            target_tracking_rewards.append(target_tracking_reward)
            boundary_punishments.append(boundary_punishment)
            duplicate_tracking_punishments.append(duplicate_tracking_punishment)
            protector_punishments.append(protector_punishment)

            # weights
            uav.raw_reward = (config["uav"]["alpha"] * target_tracking_reward + config["uav"]["beta"] *
                              boundary_punishment + config["uav"]["gamma"] * duplicate_tracking_punishment + config["uav"]["omega"] * duplicate_tracking_punishment)

        rewards = []
        for uav in self.uav_list:
            reward = uav.calculate_cooperative_reward(self.uav_list, pmi, config['cooperative'])
            uav.reward = clip_and_normalize(reward, -1, 1)
            rewards.append(uav.reward)
        return rewards, target_tracking_rewards, boundary_punishments, duplicate_tracking_punishments

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


