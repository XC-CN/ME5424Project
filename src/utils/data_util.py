import csv
import os.path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_csv(config, return_list):
    """
    :param config:
    :param return_list:
    :return: None
    """
    metric_map = {
        'return_list.csv': ('Reward', return_list.get('return_list', [])),
        'target_tracking_return_list.csv': ('target_tracking', return_list.get('target_tracking_return_list', [])),
        'boundary_punishment_return_list.csv': ('boundary_punishment', return_list.get('boundary_punishment_return_list', [])),
        'duplicate_tracking_punishment_return_list.csv': ('duplicate_tracking_punishment', return_list.get('duplicate_tracking_punishment_return_list', [])),
        'protector_collision_return_list.csv': ('protector_collision', return_list.get('protector_collision_return_list', [])),
        'protector_return_list.csv': ('protector_return', return_list.get('protector_return_list', [])),
        'protector_protect_reward_list.csv': ('protector_protect', return_list.get('protector_protect_reward_list', [])),
        'protector_block_reward_list.csv': ('protector_block', return_list.get('protector_block_reward_list', [])),
        'protector_failure_penalty_list.csv': ('protector_failure', return_list.get('protector_failure_penalty_list', [])),
        'target_return_list.csv': ('target_return', return_list.get('target_return_list', [])),
        'target_safety_reward_list.csv': ('target_safety', return_list.get('target_safety_reward_list', [])),
        'target_danger_penalty_list.csv': ('target_danger', return_list.get('target_danger_penalty_list', [])),
        'target_capture_penalty_list.csv': ('target_capture', return_list.get('target_capture_penalty_list', [])),
        'average_covered_targets_list.csv': ('average_covered_targets', return_list.get('average_covered_targets_list', [])),
        'max_covered_targets_list.csv': ('max_covered_targets', return_list.get('max_covered_targets_list', [])),
    }

    # 将CSV文件保存到metrics/csv目录
    csv_dir = os.path.join(config["save_dir"], "metrics", "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    for filename, (header, values) in metric_map.items():
        file_path = os.path.join(csv_dir, filename)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([header])
            for value in values:
                writer.writerow([value])


def clip_and_normalize(val, floor, ceil, choice=1):
    if val < floor or val > ceil:
        val = max(val, floor)
        val = min(val, ceil)
        logger.debug("overstep in clip: clipped to [%s, %s]", floor, ceil)
    val = np.clip(val, floor, ceil)
    mid = (floor + ceil) / 2
    if choice == -1:
        val = (val - floor) / (ceil - floor) - 1
    elif choice == 0:
        val = (val - floor) / (ceil - floor)
    else:
        val = (val - mid) / (mid - floor)
    return val
