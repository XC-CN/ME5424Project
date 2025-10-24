# MARL UAVs-Targets-Tracking

一个基于多智能体强化学习（Multi-Agent Actor-Critic, MAAC）的无人机（UAV）与目标追踪仿真项目。环境中包含 UAV、目标（Target）与保护者（Protector），通过奖励与惩罚设计推动 UAV 学习高效追踪目标并避免进入保护者的“禁区”。

## 功能概览
- 多 UAV 协作与通信，基于离散动作控制移动与转向
- 目标捕获判定（`capture_radius`）与覆盖率统计
- 保护者安全半径惩罚与边界惩罚（靠近边界或保护者会受罚）
- 可选的协同奖励（PMI 网络或平均奖励分配）
- 训练日志、指标保存与简易动画绘制（贴图与轨迹）

## 目录结构
- `src/main.py`：入口脚本，解析参数与运行训练
- `src/train.py`：训练循环与采样更新
- `src/environment.py`：环境更新、奖励计算与位置记录
- `src/agent/uav.py`：UAV 行为、观测与奖励分量计算
- `src/agent/protector.py`：保护者移动与边界约束
- `src/agent/target.py`：目标移动与边界约束
- `src/models/actor_critic.py`：Actor-Critic 模型与策略采样
- `src/models/PMINet.py`：PMI（互信息）网络（协作奖励）
- `src/utils/*.py`：绘图、数据工具、参数解析等
- `src/configs/*.yaml`：配置文件（默认使用 `MAAC-R.yaml`）
- `results/MAAC-R/...`：训练输出目录（自动创建）

## 环境依赖
- Python 3.9+（建议）
- PyTorch、NumPy、SciPy、tqdm、Matplotlib、Pillow、imageio



## 如何运行

- 环境准备
  - 安装 Python（建议 3.10/3.11）
  - 安装依赖（一次安装）：`pip install numpy matplotlib pillow imageio pyyaml torch torchvision torchaudio`

- 训练（保存轨迹与曲线）
  - 命令：`python .\src\main.py --phase train --method MAAC-R -e 50 -s 300 -f 20`
  - 说明：`-e` 训练轮数；`-s` 每轮步数；`-f` 保存/日志频率
  - 输出：`results\MAAC-R\{实验ID}\` 下的
    - `animated\animated_textured_plot_1.mp4`（动画）
    - `frames\tex_frame_*.png`（帧图）
    - `actor\ / critic\ / pmi\`（权重保存目录，按需使用）
    - `t_xy\ / u_xy\ / p_xy\ / covered_target_num\`（轨迹与统计）
    - `logs\`（TensorBoard 日志）

- 评估（用已训练好的权重）  
  - 命令：`python .\src\main.py --phase evaluate --method MAAC-R -s 300 -a <actor权重路径> -c <critic权重路径> -p <pmi权重路径>`
  - 说明：可只填已有的权重；未提供则使用随机初始化

- 演示（内置启发式策略，无需权重）
  - 命令：`python .\src\main.py --phase run --method MAAC-R -s 300`
  - 说明：保存的动画和帧与训练/评估一致

- GPU/CPU 切换
  - 在 `src\configs\MAAC-R.yaml` 中设置：
    - `gpus: -1` 使用 CPU（默认回退）
    - `first_device: 0` 和 `gpus: 1` 使用首个 GPU；多卡设置 `gpus: N`

- 配置说明（常用）
  - `environment.n_uav / m_targets / n_protectors`：数量
  - `uav.dp`：无人机探测半径（影响观测与可视化）
  - `target.capture_radius`：目标被捕获半径（决定何时“消失”）
  - `protector.safe_radius`：保护者手臂长度的可视半径
  - `protector.knockback` / `protector.arm_thickness`：手臂物理弹开强度与厚度（可选）
  - 修改配置后直接重新运行命令即可生效


## 配置说明（MAAC-R.yaml）
- 顶部字段
  - `exp_name`：实验名称，参与输出目录命名
  - `result_dir`：结果根目录（如 `../results/MAAC-R`）
  - `seed`：随机种子
  - `cooperative`：协同奖励系数，控制自利 vs. 共享比例

- `environment`
  - `n_uav`：UAV 数量
  - `m_targets`：目标数量
  - `n_protectors`：保护者数量
  - `x_max`/`y_max`：地图尺寸（坐标单位）
  - `na`：离散动作空间维数（策略输出的类别数）

- `uav`
  - `dt`：时间步长
  - `v_max`：最大速度
  - `h_max`：角速度档位（每步最大转角约为 `π / h_max`）
  - `dc`：通信/重复追踪判定半径（用于“重复追踪惩罚”）
  - `dp`：观测/追踪半径（影响目标跟踪奖励触发）
  - `alpha`/`beta`/`gamma`/`omega`：奖励分量权重（线性组合为总奖励）

- `protector`
  - `safe_radius`：保护者的安全半径（禁区半径），UAV 进入会受罚
    - 值越大越容易触发惩罚；与动画中保护者“阻挡手臂”的半长度一致

- `target`
  - `capture_radius`：目标捕获半径（UAV 到目标距离 ≤ 此值即捕获）
    - 判定逻辑在 `environment.step`，被捕获目标会被隐藏

- `actor_critic`
  - `buffer_size`：经验回放容量
  - `sample_size`：每次采样数量；为 `0` 时通常按每轮 `num_steps * n_uav` 取样
  - `actor_lr`/`critic_lr`：学习率
  - `hidden_dim`：隐藏层维度
  - `gamma`：折扣因子（0.95）

- `pmi`
  - `hidden_dim`：PMI 网络隐藏层维度
  - `b2_size`：每轮选择的训练样本池大小
  - `batch_size`：训练批大小

## 奖励分量与越界裁剪
- 主要分量在 `UAV.calculate_raw_reward(...)` 与 `Environment.calculate_rewards(...)`：
  - `target_tracking_reward`：目标跟踪奖励
  - `duplicate_tracking_punishment`：重复追踪惩罚（多 UAV 过近）
  - `boundary_punishment`：靠近边界惩罚
  - `protector_punishment`：靠近保护者惩罚（距离 < `safe_radius`）
- 裁剪与归一化：`src/utils/data_util.py` 的 `clip_and_normalize(...)`。
  - 当原始值超出 `[floor, ceil]`，会被裁剪到边界并归一化用于训练。


## 结果输出
- 输出目录：`results/MAAC-R/<exp_name_时间戳_随机ID>/`
  - 包含：配置快照 `args.yaml`、指标 `*_return_list.csv`、可能的动画帧与汇总图

## 可选扩展：碰撞“弹开”效果
- 若希望“UAV 碰到保护者的手臂后被弹开”，可在 `config` 新增：
  - `protector.knockback`（弹开距离）、`protector.arm_thickness`（手臂碰撞厚度）
- 在 `environment.step(...)` 中实现点到线段最近距离的碰撞检测与推离（沿法线方向推离，推离距离不超过 `knockback`）。。


## 致谢
- 本项目用于课程与研究实验，欢迎在此基础上扩展更复杂的协作策略与物理交互。
