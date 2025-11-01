import random
from math import pi, sin, cos


class PROTECTOR:
    def __init__(self, x0: float, y0: float, h0: float, act_id, v_max: float, h_max: float, dt, safe_radius: float):
        self.x, self.y = x0, y0
        self.h = h0
        self.act_id = act_id
        self.v_max = v_max
        self.h_max = h_max
        self.dt = dt
        self.safe_r = safe_radius

    def update_position(self, action_id=None):
        """随机小角速 + 匀速前进（可用于简单仿真）"""
        self.act_id = 0 if action_id is None else action_id
        a = random.uniform(-self.h_max, self.h_max)
        self.x += self.dt * self.v_max * cos(self.h)
        self.y += self.dt * self.v_max * sin(self.h)
        self.h += a * self.dt

    def clamp_inside(self, x_max, y_max):
        """边界夹紧 + 调头"""
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


    # 如需训练保护者，可以加 get_local_state(...)
