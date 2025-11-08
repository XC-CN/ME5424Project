# MAAC-R 多智能体鹰-鸡协同系统 🦅🐔🐣

> 课程项目：将原始“单鹰”方案升级为“老鹰、母鸡、小鸡”三方智能体的对抗与协作训练平台，基于多智能体 Actor-Critic (MAAC)。

## 项目概述

- **老鹰 (UAV/Eagle)** 🦅：负责主动捕获目标，策略由强化学习驱动。
- **母鸡 (Protector)** 🐔：学习挡在老鹰和小鸡之间，并保持护卫半径。
- **小鸡 (Target)** 🐣：学习规避老鹰并靠近最近的母鸡。
- 训练过程中三方共享同一个物理环境，但拥有独立的观测、网络和经验池；所有智能体都会被地图边界强制约束。

## 环境准备

- 推荐 Python 3.9 及以上版本（3.10/3.11 表现最佳）。
- 依赖：PyTorch、NumPy/SciPy、Matplotlib、Pillow、imageio、tqdm、PyYAML、TensorBoard。

```bash
pip install -r requirements.txt
# 若 requirements.txt 不完整，可手动安装常用依赖：
pip install numpy scipy matplotlib pillow imageio torch torchvision torchaudio tqdm pyyaml tensorboard
```

## 核心特性

- **三角色独立策略** 🧠：老鹰、母鸡、小鸡各自拥有 Actor-Critic 与经验回放池，避免策略冲突。
- **奖励重构** 📈：加入守护、阻挡、安全半径等细粒度奖励，精细引导行为。
- **强制边界** 🗺️：所有智能体位置会被实时裁剪在 `x_max × y_max` 地图内，防止出界。
- **实时评估** 📺：评估阶段自动弹出在线可视化窗口，可边仿真边观看结果。
- **自动加载最新模型** 🚀：在评估阶段未显式传入权重路径时，会自动使用 `results/MAAC-R/` 中最近一次训练产物。
- **物理碰撞与击退** 💥：新增了保护者对UAV的物理交互机制。当UAV靠近保护者时，会被其“手臂”弹开，并暂时锁定朝向，模拟真实的物理阻挡效果。

## 智能体策略概览

- **老鹰（UAV）**：每个仿真步都会读取场内所有未被捕获小鸡的实时位置，并结合护卫和队友的观测组成局部状态；奖励强调迅速捕获目标、避免越界和重复围捕，同时考虑被母鸡击退的惩罚。
- **母鸡（Protector）**：围绕安全半径持续构建观测，优先护住半径内的小鸡，并通过挡在老鹰与小鸡连线之间获取更高奖励；一旦有小鸡被捕会整体受罚，驱动其贴身防御。
- **小鸡（Target）**：持续感知最近的母鸡和老鹰，策略倾向于靠近保护者、远离威胁；奖励鼓励保持在护卫范围内并惩罚被捕，形成协同躲避行为。

## 快速上手

### 激活conda环境

```
conda activate ME5424Project
```

### 演示模式（无需训练、无需权重）

```bash
python src/main.py --phase run --method MAAC-R -s 300
```

- `--phase run`：使用内置启发式策略快速演示。
- `-s 300`：执行 300 步仿真。
- 仿真过程默认离线渲染，结束后会在 `results/MAAC-R/<experiment>/animated/` 输出合成动画。

### 训练模式（自动保存最新模型）

```bash
python src/main.py --phase train --method MAAC -e 100 -s 1000
```

- `-e 100`：训练 100 局（示例默认值，可按需调整）。
- `-s 1000`：每局最多 1000 步，配合奖励塑形观察长期行为。
- 训练期间默认不弹出可视化窗口。如需观测，可在命令后追加 `--render_when_train` 立即开启训练渲染。
- 训练完成后，`results/MAAC-R/<experiment>/` 会包含：
  - `actor/`、`critic/`、`pmi/`：按保存频率命名的模型快照（例如 `uav_actor_weights_20.pth`）。
  - `logs/`：TensorBoard 训练曲线。
  - `frames/`、`animated/`：关键帧与离线合成视频。
  - `u_xy/`、`t_xy/`、`p_xy/` 与 `covered_target_num/`：轨迹和覆盖统计 CSV。

### 评估模式（默认加载最新训练结果并实时播放）

```bash
python src/main.py --phase evaluate --method MAAC -s 500
```

- 未显式传入 `--actor_path` / `--protector_actor_path` / `--target_actor_path` 时，程序会自动在 `results/MAAC-R/` 中查找最近一次训练并加载对应权重。
- 评估阶段会弹出实时可视化窗口，支持拖动观察；默认不会在 `results/MAAC-R/<experiment>/` 下生成新文件，如需导出动画或统计数据，可在 `configs/MAAC-R.yaml` 的 `evaluate` 区段将 `save_outputs` 设为 `true`。
- 如需指定旧模型，可手动传入各角色的 `--*_actor_path` / `--*_critic_path`。
- 若在服务器或无图形界面运行，可在 `configs/MAAC-R.yaml` 将 `evaluate.enable_live` 改为 `false` 关闭在线渲染。

## 常用参数说明

| 参数                                                    | 说明                                  | 适用命令                |
| ------------------------------------------------------- | ------------------------------------- | ----------------------- |
| `--phase {train,evaluate,run}`                        | 控制运行阶段：训练 / 评估 / 演示      | 全部                    |
| `--method`                                            | 选择算法配置，示例使用 `MAAC-R`     | 全部                    |
| `-e, --num_episodes`                                  | 训练总局数                            | `train`               |
| `-s, --num_steps`                                     | 每局步数上限（评估/演示则为仿真步数） | 全部                    |
| `-f, --frequency`                                     | 训练保存与日志间隔                    | `train`               |
| `--render_when_train`                                 | 训练时强制开启实时渲染窗口            | `train`               |
| `--actor_path`, `--critic_path`                     | 指定老鹰 Actor/Critic 权重            | `evaluate`            |
| `--protector_actor_path`, `--protector_critic_path` | 指定母鸡模型权重                      | `evaluate`            |
| `--target_actor_path`, `--target_critic_path`       | 指定小鸡模型权重                      | `evaluate`            |
| `--pmi_path`                                          | 指定 PMI 网络权重                     | `train`, `evaluate` |

> 提示：评估阶段若仅想替换部分角色的权重，可以只传入对应参数；未填写的角色仍会自动加载最新快照。

## 训练产出与可视化

- **TensorBoard** 📊：执行 `tensorboard --logdir results/MAAC-R` 可查看奖励、损失等曲线。
- **离线动画** 🎥：`animated/` 目录下的 MP4 为高分辨率渲染，可用于汇报展示。
- **实时评估** 🖥️：`evaluate.enable_live=true` 时，每一步都会更新窗口中的鹰/鸡轨迹、覆盖率文本和探测半径。
- **数据导出** 💾：`*_xy/` 和 `covered_target_num/` CSV 便于后续自定义分析。

## 配置文件提示

- 核心配置位于 `src/configs/MAAC-R.yaml`，关键字段：
  - `environment.x_max / y_max`：地图边界；所有智能体都会在 `step` 内被裁剪到该范围。
  - `protector.safe_radius`、`target.capture_radius` 等参数可调整奖励强度。
  - `actor_critic`、`protector_actor_critic`、`target_actor_critic` 可分别设置学习率、隐藏层等超参数。
  - `evaluate` 区段可控制在线评估：`enable_live`、`render_pause`（刷新间隔秒）、`render_trail`（轨迹保存长度）、`save_animation`（是否导出离线视频）、`save_outputs`（是否写入 CSV/MP4）。
- **新增物理交互参数** ✨：
  - `protector.knockback`：击退强度。
  - `protector.arm_thickness`：碰撞判定的手臂厚度。
  - `protector.heading_lock_duration`：击退后UAV的朝向锁定时间。

## 开发者说明

- **`captured_targets_count`** 📝：`uav.py` 中新增了 `captured_targets_count` 属性，用于记录单步内捕获的目标数量。目前该值仅在环境中更新，尚未用于奖励计算，可作为未来奖励设计的扩展点。

## 常见问题

- **命令路径**：执行脚本时需使用斜杠路径，如 `python src/main.py ...`，不要写成 `src.main.py`。
- **无图形界面**：在远程服务器或 CI 环境评估时，请将 `evaluate.enable_live` 设为 `false`，避免窗口初始化失败。
- **模型未加载**：若 `results/MAAC-R/` 为空或缺少最新权重，评估会保持原始随机策略；此时请先完成至少一次训练或手动指定权重路径。
- **出界行为**：新版默认启用边界裁剪，若发现仍有异常，可检查配置中的 `x_max/y_max` 是否过小，或在调试时打印智能体坐标。
- **OMP 库冲突**：若出现 `OMP: Error #15` 提示，可在命令前导出环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`，例如 `set KMP_DUPLICATE_LIB_OK=TRUE`（Windows PowerShell 用 `$env:KMP_DUPLICATE_LIB_OK='TRUE'`）。

祝顺利完成实验与汇报！🎉
