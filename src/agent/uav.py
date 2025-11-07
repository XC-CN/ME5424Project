import random
from math import cos, sin, sqrt, exp, pi, e, atan2, hypot
import numpy as np
from typing import List, Tuple, Optional
from models.PMINet import PMINetwork
from agent.target import Target
from agent.protector import Protector
# 使用numpy实现softmax，避免依赖scipy
def softmax(x):
    """使用numpy实现softmax函数"""
    exp_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
    return exp_x / np.sum(exp_x)
from utils.data_util import clip_and_normalize


class UAV:
    def __init__(self, x0, y0, h0, a_idx, v_max, h_max, na, dc, dp, dt):
        """
        :param dt: float, 闂傚倷鐒﹁ぐ鍐洪敃鍌涘亱闂侇剙绉甸崕宥夋煕閺囥劌澧い蟻鍥ㄢ拻闁稿本绋掔亸锔戒繆椤愩垺鍋ユ�?
        :param x0: float, 闂備胶顫嬮崟顐㈩潔闂?
        :param y0: float, 闂備胶顫嬮崟顐㈩潔闂?
        :param h0: float, 闂備礁鎼悧濠勬崲閸岀偛绠?
        :param v_max: float, 闂備礁鎼悧鍐磻閹炬剚鐔嗛柛顐㈡閹冲骸鈻撻悩缁樷拺妞ゆ劑鍊曢弸搴ㄦ�?
        :param h_max: float, 闂備礁鎼悧鍐磻閹炬剚鐔嗛柛顐㈡濡厼顭囬敓鐘斥拺妞ゆ劑鍊曢弸搴ㄦ�?
        :param na: int, 闂備礁鎲￠弻锝夊礉鐎ｎ剛绀婇柡鍐ㄥ€婚惌姘箾瀹割喕绨奸柨娑氬枛閺岋綁濡搁妷銉紑濠电儑缍嗛崰妤呭�?
        :param dc: float, 濠电偞鍨堕幐鍝モ偓娑掓櫅铻為柕鍫濇椤╂煡鏌ｉ幋鐐嗘垹鑺卞顑炵懓鈹冮崹顔瑰亾閺嶎偆鐭堥柨鏇炲€归崕宥夋煕閺囥劌澧ù鐘趁…鍧楀醇閸℃顥犻柣娑掓櫇�?
        :param dp: float, 闂佽崵鍠愰悷銉╁磹閸︻厾鐭堥柟缁㈠枟閸庡孩銇勯弮鍌涙珪闁搞劌銈搁弻锝夊Ω閵夈儺浠惧┑鐐存崄閸╂牕顕ラ崟顕呮▉缂備焦鍔栫粙鎾跺垝?
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na

        # action
        self.a = a_idx

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp

        # time interval
        self.dt = dt

        # world bounds (闂備焦鐪归崹鍏肩椤掑嫮宓侀柛鈩兩戝▍鐘绘煕閹扳晛濡介棅顒夊墴閺屾盯骞掗幘鍓佺暤濡炪値鍋呴崝娆忕暦閻樿鍐€闁靛瀵屽Σ顖炴煟閻樿京顦﹀褌绮欓�?
        self.world_bounds: Optional[Tuple[float, float]] = None

        # set of local information
        # self.communication = []
        self.target_observation = []
        self.protector_observation = []
        self.uav_communication = []
        self._target_distances = []
        self._protector_distances = []

        # reward
        self.raw_reward = 0
        self.reward = 0
        # heading knockback lock (number of steps to keep heading unchanged)
        self.lock = 0
        # number of targets captured within the last environment step
        self.captured_targets_count = 0
        # latest applied heading delta in radians
        self.heading_delta = 0.0

    def __distance(self, target) -> float:
        """
        calculate the distance from uav to target
        :param target: class UAV or class Target
        :return: scalar
        """
        return sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    @staticmethod
    def distance(x1, y1, x2, y2) -> float:
        """
        calculate the distance from uav to target
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_world_bounds(self, x_max: float, y_max: float):
        self.world_bounds = (max(float(x_max), 0.0), max(float(y_max), 0.0))

    def _clamp_inside(self):
        if self.world_bounds is None:
            return
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

    def update_position(self, action: Optional[int]) -> Tuple[float, float, float]:
        """
        Receive an optional discrete action index and update the current position.
        When the UAV is in a locked state (e.g. after knockback), the heading will
        remain unchanged until the lock counter reaches zero.
        """
        if action is not None:
            self.a = int(action)
            self.heading_delta = self.discrete_action(self.a)
        else:
            self.heading_delta = float(np.random.uniform(-self.h_max, self.h_max))

        dx = self.dt * self.v_max * cos(self.h)
        dy = self.dt * self.v_max * sin(self.h)
        self.x += dx
        self.y += dy

        if self.lock > 0:
            self.lock -= 1
        else:
            self.h += self.dt * self.heading_delta

        self.h = (self.h + pi) % (2 * pi) - pi
        if self.world_bounds is not None:
            self._clamp_inside()
        return self.x, self.y, self.h

    def discrete_action(self, a_idx: int) -> float:
        """
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 闂備礁鎲￠〃鍛存偪閸ヮ剨缍栨俊銈勭劍閸庣喖鏌ㄩ弮鍥撻柡鍡╁弮閺屾稑鈻庨幇鎯扳偓鍧楁煕?
        """
        # from action space to the real world action
        na = a_idx + 1  # �?1 闁诲孩顔栭崰鎺楀磻閹炬枼鏀芥い鏃傗拡閸庢垿鏌涚仦璇插�?
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def apply_knockback(self, knockback_angle: float, lock_duration: int = 0) -> None:
        """
        Immediately set the heading to the provided angle and lock it for a number of
        subsequent integration steps so the UAV cannot steer until the lock expires.
        """
        self.h = (knockback_angle + pi) % (2 * pi) - pi
        self.lock = max(self.lock, int(max(0, lock_duration)))
        # Reset heading delta so observers do not read stale angular moves
        self.heading_delta = 0.0

    def _get_axis_scale(self) -> Tuple[float, float]:
        if self.world_bounds is not None:
            x_max, y_max = self.world_bounds
            return max(x_max, 1.0), max(y_max, 1.0)
        base = max(self.dp, 1.0)
        return base, base

    def observe_target(self, targets_list: List['Target'], relative=True):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return: None
        """
        self.target_observation = []  # Reset observed targets
        self._target_distances = []
        x_scale, y_scale = self._get_axis_scale()
        speed_scale = max(self.v_max, 1e-6)

        for target in targets_list:
            if getattr(target, 'captured', False):
                continue

            dist = max(self.__distance(target), 1e-6)
            self._target_distances.append(dist)

            if relative:
                dx = (target.x - self.x) / x_scale
                dy = (target.y - self.y) / y_scale
                vx = cos(target.h) * target.v_max / speed_scale - cos(self.h)
                vy = sin(target.h) * target.v_max / speed_scale - sin(self.h)
            else:
                dx = target.x / x_scale
                dy = target.y / y_scale
                vx = cos(target.h) * target.v_max / speed_scale
                vy = sin(target.h) * target.v_max / speed_scale

            self.target_observation.append((dx, dy, vx, vy))
                    
    def observe_protector(self, protectors_list: List['Protector'], relative=True):
        """
        Observing target with a radius within dp
        :param relative: relative to uav itself
        :param targets_list: [class UAV]
        :return: None
        """
        self.protector_observation = []  # Reset observed protectors
        self._protector_distances = []
        x_scale, y_scale = self._get_axis_scale()
        speed_scale = max(self.v_max, 1e-6)

        for protector in protectors_list:
            dist = max(self.__distance(protector), 1e-6)
            self._protector_distances.append(dist)

            if relative:
                dx = (protector.x - self.x) / x_scale
                dy = (protector.y - self.y) / y_scale
                vx = cos(protector.h) * protector.v_max / speed_scale - cos(self.h)
                vy = sin(protector.h) * protector.v_max / speed_scale - sin(self.h)
            else:
                dx = protector.x / x_scale
                dy = protector.y / y_scale
                vx = cos(protector.h) * protector.v_max / speed_scale
                vy = sin(protector.h) * protector.v_max / speed_scale

            self.protector_observation.append((dx, dy, vx, vy))

    def observe_uav(self, uav_list: List['UAV'], relative=True):  # communication
        """
        communicate with other uav_s with a radius within dp
        :param relative: relative to uav itself
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_communication = []  # Reset observed targets
        for uav in uav_list:
            dist = self.__distance(uav)
            if dist <= self.dc and uav != self:
                # add (x, y, vx, vy, a) information
                if relative:
                    self.uav_communication.append(((uav.x - self.x) / self.dc,
                                                   (uav.y - self.y) / self.dc,
                                                   cos(uav.h) - cos(self.h),
                                                   sin(uav.h) - sin(self.h),
                                                   (uav.a - self.a) / self.Na))
                else:
                    self.uav_communication.append((uav.x / self.dc,
                                                   uav.y / self.dc,
                                                   cos(uav.h),
                                                   sin(uav.h),
                                                   uav.a / self.Na))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, float]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, float]):
        """
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        """
        observation = self.target_observation + self.protector_observation
        return self.uav_communication, observation, (self.x / self.dc, self.y / self.dc, self.a / self.Na)

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, sb = self.__get_all_local_state()

        if communication:
            d_communication = []  # store the distance from each uav to itself
            for x, y, vx, vy, na in communication:
                d_communication.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            # communication = self.__transform_to_array2d(communication)
            communication = np.array(communication)
            communication_weighted = communication / np.array(d_communication)[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=0)
        else:
            # average_communication = np.zeros(4 + self.Na)  # empty communication
            average_communication = -np.ones(4 + 1)  # empty communication  # TODO -1闂備礁鎲￠懝楣冩偋婵犲嫭鍏滈柛顐ｆ礀�?

        if observation:
            distances = np.array(self._target_distances + self._protector_distances, dtype=np.float32)
            if distances.size == 0:
                average_observation = -np.ones(4)
            else:
                distances = np.maximum(distances, 1e-3)
                observation = np.array(observation, dtype=np.float32)
                observation_weighted = observation / distances[:, np.newaxis]
                average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation  # TODO -1闂備礁鎲￠懝楣冩偋婵犲嫭鍏滈柛顐ｆ礀�?

        sb = np.array(sb)
        result = np.hstack((average_communication, average_observation, sb))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        :return: np.ndarray
        """
        # using weighted mean method:
        return self.__get_local_state_by_weighted_mean()

    def __calculate_multi_target_tracking_reward(self, uav_list) -> float:
        """
        calculate multi target tracking reward
        :return: scalar [1, 2)
        """
        track_reward = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= self.dp:
                    reward = 1 + (self.dp - distance) / self.dp
                    # track_reward += clip_and_normalize(reward, 1, 2, 0)
                    track_reward += reward  # 婵犵數鍋涙径鍥礈濠靛棴鑰垮☉鏃戞苟ip, 闂備線娼荤拹鐔煎礉韫囨稒鍎嶉柣鏂垮悑閸嬨劑鏌曟繝蹇涙妞は佸喚鐔嗛柤鍝ユ暩婢х敻鏌涙惔锝傛寘lip
        return track_reward

    def __calculate_target_capture_reward(self) -> float:
        """
        Reward accumulated targets captured in the last environment step.
        """
        capture_reward = float(self.captured_targets_count)
        self.captured_targets_count = 0
        return capture_reward

    def __calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=2) -> float:
        """
        calculate duplicate tracking punishment
        :param uav_list: [class UAV]
        :param radio: radio闂備焦妞垮鍧楀礉瀹ュ拋鐒介柣銏㈩焾缁犲磭鎲稿澶婃槬婵°倕鎳庣粻姘舵煃閸濆嫬鏆熺紒瀣墦閺岋綁濡搁妷銉紓闁诲骸鐏氶悧鐘茬�? 闂佺儵鍓濈敮鎺楀箠鎼淬劌鍚规い鎾跺枎缁剁偞鎱ㄥΟ铏癸紞缂佺姴銈搁弻鐔兼濞戞銈夋煟閿濆拋鐓肩€规洘顨婃俊鐑芥晝閳ь剟宕濋敂鍓х＝?
        :return: scalar (-e/2, -1/2]
        """
        total_punishment = 0
        for other_uav in uav_list:
            if other_uav != self:
                distance = self.__distance(other_uav)
                if distance <= radio * self.dp:
                    punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
                    # total_punishment += clip_and_normalize(punishment, -e/2, -1/2, -1)
                    total_punishment += punishment  # 婵犵數鍋涙径鍥礈濠靛棴鑰垮☉鏃戞苟ip, 闂備線娼荤拹鐔煎礉韫囨稒鍎嶉柣鏂垮悑閸嬨劑鏌曟繝蹇涙妞は佸喚鐔嗛柤鍝ユ暩婢х敻鏌涙惔锝傛寘lip
        return total_punishment

    def __calculate_boundary_punishment(self, x_max: float, y_max: float) -> float:
        """
        :param x_max: border of the map at x-axis, scalar
        :param y_max: border of the map at y-axis, scalar
        :return:
        """
        x_to_0 = self.x - 0
        x_to_max = x_max - self.x
        y_to_0 = self.y - 0
        y_to_max = y_max - self.y
        d_bdr = min(x_to_0, x_to_max, y_to_0, y_to_max)
        if 0 <= self.x <= x_max and 0 <= self.y <= y_max:
            if d_bdr < self.dp:
                boundary_punishment = -0.5 * (self.dp - d_bdr) / self.dp
            else:
                boundary_punishment = 0
        else:
            boundary_punishment = -1/2
        return boundary_punishment  # 婵犵數鍋涙径鍥礈濠靛棴鑰垮☉鏃戞苟ip, 闂備線娼荤拹鐔煎礉韫囨稒鍎嶉柣鏂垮悑閸嬨劑鏌曟繝蹇涙妞は佸喚鐔嗛柤鍝ユ暩婢х敻鏌涙惔锝傛寘lip
        # return clip_and_normalize(boundary_punishment, -1/2, 0, -1)

    def calculate_raw_reward(self, uav_list: List['UAV'], target__list: List['Target'], protector_list: List['Protector'], x_max, y_max):
        """
        calculate three parts of the reward/punishment for this uav
        :return: float, float, float
        """
        reward = (self.__calculate_multi_target_tracking_reward(target__list) +
                  self.__calculate_target_capture_reward())
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        punishment = self.__calculate_duplicate_tracking_punishment(uav_list)
        protector_punishment = self.__calculate_protector_collision_punishment(protector_list)
        return reward, boundary_punishment, punishment, protector_punishment

    def __calculate_protector_collision_punishment(self, protector_list: List['Protector']) -> float:
        """
        缂傚倷鑳舵慨顓㈠磻閹剧粯鐓曟俊銈勭劍缁€鍫熺箾閸喎鐏╅柟鐟板閳规垿宕奸銈傚亾閺屻儲鈷掗柛鎰╁妿婢ф洟鏌嶈閸撴瑩宕ョ€ｎ喖纾块柣銏㈡暩绾剧偓鎱ㄥΟ鍝勨挃缂?
        - �?UAV 濠电偞鍨堕幐鍝ヨ姳婵傜绠查柕蹇嬪€曠粻?Protector 闂佽崵濮烽崕銈囨崲閸曨垪�?< threshold 闂備礁鎲＄敮妤呮偡閵堝棭娈介柛銉墯閸嬨劑鏌ｉ弮鍌ょ劸闁告棑濡囩槐鎾诲礃閳哄啫鏆＄紓浣介哺閻熝囧箯瑜版帗鍋勯悘蹇庣娴滃墽鐥銏╂缂佲偓?
        - 闂佸搫顦弲婊堝蓟閵娿儍娲冀椤撶偟锛欓梺缁樻尭妤犲摜寰婇崸妤佺厱闁哄秲鍔嶉妵婵囦�?[-K, 0]闂備焦瀵х粙鎴︽偋婵犲偊鑰挎い鎾€棰佹睏闂佺懓鎼粔鍫曨敂闁秵鐓曢柡宥冨妿婢瑰嫮绱掓潏銊ュ摵闁轰礁绉舵禒锕傛嚍閵夈儮鍋撻悽鐢电＝濞撴艾娲ら獮姗€�?clip_and_normalize 闁荤喐绮庢晶妤冩暜婵犲嫮鍗氶柟缁㈠枛缁€宀勬煛瀹擃喖鍟伴崢?[-1,0]
        """
        # UAV 闂備胶鍘ч〃搴㈢閿濆顥婇柍鍝勬噹绾偓婵犵數濮撮崐鑽ょ玻閻愮儤鐓ユ繛鎴烆焾鐎氭壆鎲搁悧鍫㈠弨闁哄苯锕ら濂稿炊閳哄倻鈧參姊洪悷鎵憼闁绘濮烽崚鎺楊敍濠婂嫬�?
        uav_radius = getattr(self, "radius", 0.5)
        worst_pen = 0.0  # 闂備浇顕栭崹顏堝疾濠靛牏绀婇柟鐑樻尵閳绘梹绻涘顔荤凹濠殿喗绮嶉幈銊モ攽閹惧墎蓱闂佸搫妫涢崰鏍极? 闂佽崵鍋炵粙蹇涘礉鎼淬劌桅婵娉涚猾宥夋煟濡搫鏆遍柛鏃撳缁辨捇宕橀埡鍐ㄦ殹�?
        for prot in protector_list:
            # protector 闂傚倷鐒﹁ぐ鍐儔閼测斁鍋撻悷鎵紞闁逞屽墲椤鎳濇ィ鍐╁創婵せ鍋撴鐐╁�?safe_r闂備焦瀵х粙鎴︽偋婵犲洤鍨傞柕濠忛檮婵挳鎮归幁鎺戝闁?prot.safe_r�?
            prot_safe = getattr(prot, "safe_r", None)
            if prot_safe is None:
                # 闂備焦妞垮鍧楀礉閹寸姴鍨濇い鏍ㄧ矋婵ジ鏌曢崼婵囶棞闁?prot.radius
                prot_safe = getattr(prot, "radius", 0.5)
            threshold = prot_safe + uav_radius  # 闂佽崵鍠愰悷銉ノ涢弮鍌涘闁绘柨鎼弸鍫ユ煟濡も偓閻楀﹦鈧?闂佽崵濮崑鎾绘煥閺冨洦顥夐柣蹇擄躬閺屾稑螖閳ь剟鏁冮妶鍥偨?
            dx = prot.x - self.x
            dy = prot.y - self.y
            dist = hypot(dx, dy)
            if dist < threshold:
                # 闂備浇顕栭崹顏堝疾濠靛牏绀婇柟鐑橆殔缁犳澘顭跨捄鐑樻崳闁绘稈鏅濈槐鎺楀磼濠婂懎绗￠梺鎼炲妽閼规崘顣鹃梺鍝勫€告晶鑺ュ閹扮増鐓ユ繛鎴炵懃閸濈儤绻涚喊鍗炲妞ゆ洏鍎辫灃闁告洦鍘滈妸鈺佺骇闁冲搫鍊搁悘锕傛煕鎼达絾鏆╃紒杈ㄦ煥椤繂鐣烽崶鍡愬妼椤潡宕奸崱妤冃ょ紒鈧崟顖涘仯闁归偊鍠栧▍鎰�?
                # 闂備礁鎼悧婊勭濠婂懐绀婂┑鐘插暟閳绘棃鏌嶈閸撶喎鐣烽悩璇插唨闁靛鍎辨慨锔剧磽閸屾艾鏆為柣鏃戝墰濡叉劕鈹戦崶鈺冩嚌闁诲函缍嗘禍璺好洪妶澶嬬叆?- (threshold - dist)  -> 濠电偛顕慨鎾敄閸曨叀�?(-threshold, 0]
                pen = -(threshold - dist)
                if pen < worst_pen:
                    worst_pen = pen
        return worst_pen

    def __calculate_cooperative_reward_by_pmi(self, uav_list: List['UAV'], pmi_net: "PMINetwork", a) -> float:
        """
        calculate cooperative reward by pmi network
        :param pmi_net: class PMINetwork
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 闂備礁婀辩划顖炲礉閺囩喐娅犻柣妯款嚙缁€鍡涙煏閸繃顥為柣鎺嶇矙閺屻劌鈽夊Ο鑲╁姰婵犵妲呴崣鍐箚閸ャ劍濯寸紒娑橆儏閹胶绱撴担鍝勑ｆい銊ョ墕閳绘捇骞嬮悙鑼獮閻庡箍鍎卞Λ娆戝緤娴犲绠?
            return self.raw_reward

        neighbor_rewards = []
        neighbor_dependencies = []
        la = self.get_local_state()

        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
                other_uav_la = other_uav.get_local_state()
                _input = la * other_uav_la
                neighbor_dependencies.append(pmi_net.inference(_input.squeeze()))

        if len(neighbor_rewards):
            neighbor_rewards = np.array(neighbor_rewards)
            neighbor_dependencies = np.array(neighbor_dependencies).astype(np.float32)
            softmax_values = softmax(neighbor_dependencies)
            reward = (1 - a) * self.raw_reward + a * np.sum(neighbor_rewards * softmax_values).item()
        else:
            reward = (1 - a) * self.raw_reward
        return reward

    def __calculate_cooperative_reward_by_mean(self, uav_list: List['UAV'], a) -> float:
        """
        calculate cooperative reward by mean
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 闂備礁婀辩划顖炲礉閺囩喐娅犻柣妯款嚙缁€鍡涙煏閸繃顥為柣鎺嶇矙閺屻劌鈽夊Ο鑲╁姰婵犵妲呴崣鍐箚閸ャ劍濯寸紒娑橆儏閹胶绱撴担鍝勑ｆい銊ョ墕閳绘捇骞嬮悙鑼獮閻庡箍鍎卞Λ娆戝緤娴犲绠?
            return self.raw_reward

        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
        # 婵犵數鍋涙径鍥礈濠靛棴鑰垮〒姘ｅ亾鐎规洘绻堥幃鈺呭箵閹烘垶杈圥MI缂傚倸鍊搁崯顖炲垂閸︻厼�?
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else 0
        return reward

    def calculate_cooperative_reward(self, uav_list: List['UAV'], pmi_net=None, a=0.5) -> float:
        """
        :param uav_list:
        :param pmi_net:
        :param a: 0: selfish, 1: completely shared
        :return:
        """
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(uav_list, pmi_net, a)
        else:
            return self.__calculate_cooperative_reward_by_mean(uav_list, a)

    def get_rule_based_action(
        self,
        targets: List['Target'],
        protectors: List['Protector'],
        uav_list: List['UAV'],
        rule_params: dict,
    ) -> int:
        """
        根据一套启发式规则挑选离散动作：
        1. 选取最近且防守压力较小的目标；
        2. 结合边界与母鸡规避修正期望航向；
        3. 将期望航向转换为离散动作索引。
        """
        best_target = None
        best_score = -float("inf")

        for target in targets:
            if getattr(target, "captured", False):
                continue

            dist_to_target = self.distance(self.x, self.y, target.x, target.y)
            if dist_to_target < 1e-6:
                continue

            # 若母鸡距离目标更近，则认为防守严密，跳过
            defended = any(
                self.distance(protector.x, protector.y, target.x, target.y) < dist_to_target
                for protector in protectors
            )
            if defended:
                continue

            # 避免多个 UAV 聚集追同一个目标
            repetition_penalty = 0.0
            for other in uav_list:
                if other is self:
                    continue
                if self.distance(other.x, other.y, target.x, target.y) < self.dp:
                    repetition_penalty += 0.8

            score = 1.0 / dist_to_target - repetition_penalty
            if score > best_score:
                best_score = score
                best_target = target

        if best_target is None:
            return self.Na // 2

        desired_angle = atan2(best_target.y - self.y, best_target.x - self.x)

        boundary_avoid_threshold = rule_params.get("boundary_avoid_threshold", 15.0)
        boundary_avoid_strength = rule_params.get("boundary_avoid_strength", 0.8)
        protector_avoid_threshold = rule_params.get("protector_avoid_threshold", 20.0)
        protector_avoid_strength = rule_params.get("protector_avoid_strength", 1.0)

        turn_bias = 0.0

        if self.world_bounds:
            x_max, y_max = self.world_bounds
            if self.x < boundary_avoid_threshold:
                turn_bias += boundary_avoid_strength
            elif self.x > x_max - boundary_avoid_threshold:
                turn_bias -= boundary_avoid_strength
            if self.y < boundary_avoid_threshold:
                turn_bias += boundary_avoid_strength
            elif self.y > y_max - boundary_avoid_threshold:
                turn_bias -= boundary_avoid_strength

        for protector in protectors:
            dist_to_protector = self.distance(self.x, self.y, protector.x, protector.y)
            if dist_to_protector < protector_avoid_threshold:
                angle_from_protector = atan2(self.y - protector.y, self.x - protector.x)
                angle_diff = (angle_from_protector - desired_angle + pi) % (2 * pi) - pi
                turn_bias -= protector_avoid_strength * (pi / 2) * np.sign(angle_diff)

        final_angle_diff = (desired_angle - self.h + pi) % (2 * pi) - pi
        final_angle_diff += turn_bias

        angular_cmd = np.clip(final_angle_diff / self.dt, -self.h_max, self.h_max)
        k = (self.Na - 1) * angular_cmd / self.h_max
        a_idx = int(np.round((k + self.Na - 1) / 2))
        a_idx = max(0, min(self.Na - 1, a_idx))

        return a_idx

    def get_action_by_direction(self, target_list, uav_list):
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 濠电娀娼ч崐褰掓偋閺囩喐鍙忛柟鎯版濡炰粙鎮橀悙鏉戝姢闁告棑濡囩槐鎾诲礃閳哄喚鏆″┑鐐靛帶閻楁捇�?
        target_reward_weight = 1.0
        repetition_penalty_weight = 0.8
        self.epsilon = 0.25
        self.continue_tracing = 0.3
        
        best_score = float('-inf')
        best_angle = 0.0

        # 闂傚倸鍊搁幊搴ｆ崲閹存績鏀﹂柍褜鍓熼弻鐔煎垂椤愶絿鍑℃繝娈垮枓閺呯娀寮鍜佹僵妞ゆ劧绱曢棄宄瀙silon闂備焦鐪归崝宀€鈧凹鍙冮妴鍛邦樄闁诡喗濞婂畷鍫曨敄閽樺澹曟繝銏ｆ硾椤戝懘顢旈妷鈺傗拻闁告侗鍘剧粻鍙夌節瑜夐崑鎾绘⒑閻戔晜娅撻柛鐘愁殜閹?
        if random.random() < self.epsilon:
            return np.random.randint(0, self.Na)
        else:
            for target in target_list:
                target_x, target_y = target.x, target.y

                # 闁荤喐绮庢晶妤呭箰閸涘﹥娅犻柣妯款嚙缁秹鏌ｅΟ鍧楀摵闁活亙绮欓弻锟犲醇濠靛洦鍎撻梺绯曟櫅閻倿骞嗛崟顖ｆ晩闁伙絽鑻悘閬嶆⒑濮瑰洤濡奸悗姘煎灣缁絽鈽夊杈╃効?
                dist_to_target = distance(self.x, self.y, target_x, target_y)

                # 闂傚倷鐒﹁ぐ鍐矓閹绢啟鍥矗婢跺姘﹂梺绋匡功閸犳牠宕电€ｎ喗鐓熼柕濞垮劚椤忣亪鏌涙惔锝嗘毄缂佽鲸鏌ㄩ～婵嬪礈瑜忛ˇ顕€姊洪崗鍏肩凡闁哥喎鍟块‖濠冪節濮橆剛顦ч梺鍝勭墢閺佹悂鏌ч崒鐐寸厸闁告劦浜滃皬濠殿喚鎳撻ˇ闈涱嚕閻㈢浼犻柛鏇ㄥ墲閸氼偊姊绘担鐟扮祷缂佸鎸宠棟闁告瑥顦版禍銈夋煕婵犲嫬鏋涢柛銈咁儔閺屾稑螖閳ь剟鏁冮妶鍥偨闁绘劕鎼粈鍐煕濞戝崬鏋熺粭鎴︽⒑閸涘﹦鍟茬紒鐘冲灥閿曘垽顢旈崨顔绘唉闂佺锕﹂崰鏍吹鐎ｎ喗鐓曟慨姗嗗墮椤ㄦ瑧绱掗弮鈧幐鎶藉箚閸曨垼鏁婇柣锝呰嫰閻?
                repetition_penalty = 0.0
                for uav in uav_list:
                    uav_x, uav_y = uav.x, uav.y
                    if (uav_x, uav_y) != (self.x, self.y):
                        dist_to_target_from_other_uav = distance(uav_x, uav_y, target_x, target_y)
                        if dist_to_target_from_other_uav < self.dc:
                            repetition_penalty += repetition_penalty_weight

                # 闂佽崵濮崇欢銈囨閺囥垺鍋╅柡鍕箞娴滃綊鏌熼幆褍鏆辨い銈呮嚇閺岋綁鍩℃繝鍌涚亶闂佺粯鐗紞渚€骞嗛崘顔肩妞ゆ劑鍊楃粻鎺楁�?
                score = target_reward_weight / dist_to_target - repetition_penalty

                # 闂備礁鎼粔鐑斤綖婢跺﹦鏆ゅù锝呭閸ゆ淇婇妶鍌氫壕闂佹悶鍔岄柊锝夊蓟閸涱喖绶為悗锝庝憾娴犳挳姊洪崫鍕闁稿鎸搁湁闁稿繒鈷堥悞鑺ョ箾闊厼宓嗙€?
                if score > best_score:
                    best_score = score
                    best_angle = np.arctan2(target_y - self.y, target_x - self.x) - self.h

        # 濠电偛顕刊顓炩枍閹煎tinue_tracing闂備焦鐪归崝宀€鈧凹鍙冮妴鍛邦樄闁诡喗濞婂畷鍫曟晜閸撗屽晬闂備礁缍婇弲鎻掝渻閹烘梻绠旈柛灞剧〒閳绘棃鏌嶈閸撴氨绮欐径鎰垫晜闁告洦鍊ｅΔ鈧湁?
        if random.random() < self.continue_tracing:
            best_angle = 0
            
        # 闂佽绻愮换鎰涘▎蹇ヨ€块柨鐔哄Т鐎氬绻涢幋鐐殿暡闁伙綁浜堕幃宄扳枎閹邦剛鐟ㄩ柣搴ｆ嚀閸熷潡顢欒箛娑辨晩闁圭娴烽妶顕€姊虹涵鍜佸殐婵犫懇鍋撻柣鐘冲姉閸犳挾鍒掔€ｎ剛纾兼俊顖滅帛閻ゅ洭姊洪崨濠冪厽闁告柨顑囩槐鐐哄籍閸屾侗娼熼梺闈涚箳婵绮?
        a = np.clip(best_angle / self.dt, -self.h_max, self.h_max)
        k = (self.Na - 1) * a / self.h_max
        a_idx = int(np.round((k + self.Na - 1) / 2))
        a_idx = max(0, min(self.Na - 1, a_idx))
        return a_idx
