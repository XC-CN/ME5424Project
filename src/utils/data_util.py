import csv
import os.path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_csv(config, return_list):
    """
    :param config:
    :param return_list:
        return_list = {
        'return_list': self.return_list,
        'target_tracking_return_list' :target_tracking_return_list,
        'boundary_punishment_return_list':boundary_punishment_return_list,
        'duplicate_tracking_punishment_return_list':duplicate_tracking_punishment_return_list,
        'protection_return_list': protection_return_list,  # 新增
        'interception_return_list': interception_return_list,  # 新增
        'overlapping_punishment_return_list': overlapping_punishment_return_list  # 新增
    }
    :return:
    """
    # 精简：只保存主要指标到单独CSV，详细数据已在training_metrics.csv中
    with open(os.path.join(config["save_dir"], 'return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Reward'])  # 写入表头
        for reward in return_list['return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'target_tracking_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target_tracking'])  # 写入表头
        for reward in return_list['target_tracking_return_list']:
            writer.writerow([reward])

    # 精简：不再单独保存boundary_punishment和duplicate_tracking_punishment，已在training_metrics.csv中
    # 如果存在protector奖励项，保存它们
    if 'protection_return_list' in return_list:
        with open(os.path.join(config["save_dir"], 'protection_return_list.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['protection_return'])
            for reward in return_list['protection_return_list']:
                writer.writerow([reward])
    
    if 'interception_return_list' in return_list:
        with open(os.path.join(config["save_dir"], 'interception_return_list.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['interception_return'])
            for reward in return_list['interception_return_list']:
                writer.writerow([reward])


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
