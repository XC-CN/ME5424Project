# 🚁 MARL UAVs-Targets-Tracking

> 🎯 一个基于多智能体强化学习（Multi-Agent Actor-Critic, MAAC）的无人机与目标追踪仿真项目

环境中包含 **🚁 UAV**、**🎯 目标（Target）** 与 **🛡️ 保护者（Protector）**，通过奖励与惩罚设计推动 UAV 学习高效追踪目标并避免进入保护者的"禁区"。

## 📖 快速导航

- [✨ 功能概览](#-功能概览)
- [📦 环境依赖](#-环境依赖)
- [🚀 如何运行](#-如何运行)
  - [📖 命令参数详解](#-命令参数详解)（**必读！理解每个参数的含义**）
  - [1️⃣ 训练模式](#1️⃣-训练模式保存轨迹与曲线)
  - [2️⃣ 评估模式](#2️⃣-评估模式使用已训练权重)
  - [3️⃣ 演示模式](#3️⃣-演示模式启发式策略演示)（**推荐新手从这里开始！**）
- [💥 碰撞弹开效果](#-碰撞弹开效果)
- [⚙️ 详细配置说明](#️-详细配置说明maac-ryaml)
- [🎮 智能体介绍](#-智能体介绍)
- [🧠 算法框架对比](#-算法框架对比)

---

## ✨ 功能概览

- 🤝 **多 UAV 协作与通信** - 基于离散动作控制移动与转向
- 🎯 **目标捕获判定** - `capture_radius` 判定与覆盖率统计
- 🛡️ **保护者安全机制** - 安全半径惩罚与边界惩罚
- 🧠 **协同奖励系统** - PMI 网络或平均奖励分配
- 📊 **完整工具链** - 训练日志、指标保存与动画绘制

---

## 📂 目录结构

```
ME5424/
├── 📄 README.md                    # 项目文档
├── 📋 requirements.txt             # 依赖清单
├── 📂 src/                         # 源代码
│   ├── 🎯 main.py                  # 入口脚本，解析参数与运行训练
│   ├── 🔄 train.py                 # 训练循环与采样更新
│   ├── 🌍 environment.py           # 环境更新、奖励计算与位置记录
│   ├── 📂 agent/                   # 智能体模块
│   │   ├── 🚁 uav.py               # UAV 行为、观测与奖励分量计算
│   │   ├── 🛡️ protector.py         # 保护者移动与边界约束
│   │   └── 🎯 target.py            # 目标移动与边界约束
│   ├── 📂 models/                  # 神经网络模型
│   │   ├── actor_critic.py         # Actor-Critic 模型与策略采样
│   │   └── PMINet.py               # PMI（互信息）网络（协作奖励）
│   ├── 📂 utils/                   # 工具函数
│   │   ├── args_util.py            # 参数解析
│   │   ├── data_util.py            # 数据工具
│   │   ├── draw_util.py            # 绘图工具
│   │   └── gen_assets.py           # 资源生成
│   └── 📂 configs/                 # 配置文件
│       ├── ⭐ MAAC-R.yaml          # MAAC-R配置（默认推荐）
│       ├── MAAC.yaml               # MAAC配置
│       ├── MAAC-G.yaml             # MAAC-G配置
│       └── C-METHOD.yaml           # 对比方法
└── 📂 results/                     # 训练输出（自动创建）
    └── MAAC-R/
        └── {实验ID}/
            ├── 🎬 animated/        # 动画视频
            ├── 🖼️ frames/          # 训练帧图
            ├── 💾 actor/           # Actor权重
            ├── 💾 critic/          # Critic权重
            ├── 💾 pmi/             # PMI权重
            └── 📊 *.csv            # 训练指标
```

---

## � 快速开始（3分钟体验）

想立即看到效果？跳过训练，直接运行演示模式！

```bash
# 1. 进入项目目录
cd ME5424

# 2. 设置环境变量（避免警告）
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 3. 运行演示（无需训练，无需权重）
python .\src\main.py --phase run --method MAAC-R -s 3000
```

**命令含义：**
- `--phase run`：演示模式（启发式算法）
- `--method MAAC-R`：使用 MAAC-R 配置
- `-s 300`：运行 300 步（约15秒动画）

🎉 完成后在 `../results/MAAC-R/` 目录下查看生成的动画！

详细说明请查看：
- [命令参数详解](#-命令参数详解)
- [演示模式章节](#3️⃣-演示模式启发式策略演示)

---

## �📦 环境依赖

- 🐍 **Python**: 3.9+ （建议 3.10/3.11）
- 🔥 **PyTorch**: 深度学习框架
- 📊 **NumPy, SciPy**: 数值计算
- 📈 **Matplotlib**: 可视化
- 🖼️ **Pillow, imageio**: 图像处理与动画生成
- 📝 **tqdm, PyYAML, TensorBoard**: 进度条、配置解析与日志

### 💻 安装命令

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install numpy matplotlib pillow imageio pyyaml torch torchvision torchaudio tqdm scipy tensorboard
```

---

## 🚀 如何运行

### � 命令参数详解

在运行命令前，先了解各个参数的含义：

```bash
python .\src\main.py --phase <模式> --method <算法> [可选参数]
```

#### 必需参数

| 参数 | 完整形式 | 说明 | 可选值 | 默认值 |
|------|----------|------|--------|--------|
| `--phase` | `--phase` | **运行模式** | `train` / `evaluate` / `run` | `train` |
| `-m` | `--method` | **算法选择** | `MAAC-R` / `MAAC` / `MAAC-G` / `C-METHOD` | `MAAC-R` |

**`--phase` 参数详解：**
- 🏋️ `train`：训练模式 - 使用神经网络训练智能体，并保存模型权重
- 📊 `evaluate`：评估模式 - 加载已训练的模型权重进行测试
- 🎮 `run`：演示模式 - 使用启发式算法（无需权重）快速演示

**`--method` 参数详解：**
- ⭐ `MAAC-R`：**推荐** - MAAC算法 + 基于互信息的奖励分配（性能最佳）
- 🔵 `MAAC`：标准MAAC算法 + 平均奖励分配
- 🟢 `MAAC-G`：MAAC算法 + 全局奖励分配
- 🟡 `C-METHOD`：对比方法（不使用神经网络）

#### 可选参数

| 参数 | 完整形式 | 说明 | 类型 | 默认值 | 示例 |
|------|----------|------|------|--------|------|
| `-s` | `--num_steps` | **每轮步数** - 每个episode执行的仿真步数 | 整数 | 200 | `-s 300` |
| `-e` | `--num_episodes` | **训练轮数** - 总共训练多少个episodes（仅训练模式） | 整数 | 10000 | `-e 50` |
| `-f` | `--frequency` | **保存频率** - 每多少轮保存一次模型和日志 | 整数 | 100 | `-f 20` |
| `-a` | `--actor_path` | **Actor权重路径** - 预训练Actor网络权重文件路径 | 路径 | None | `-a ./results/.../actor_100.pth` |
| `-c` | `--critic_path` | **Critic权重路径** - 预训练Critic网络权重文件路径 | 路径 | None | `-c ./results/.../critic_100.pth` |
| `-p` | `--pmi_path` | **PMI权重路径** - 预训练PMI网络权重文件路径 | 路径 | None | `-p ./results/.../pmi_100.pth` |

#### 💡 命令示例解析

**示例命令：**
```bash
python .\src\main.py --phase run --method MAAC-R -s 300
```

**参数解释：**
- `python .\src\main.py`：执行主程序
- `--phase run`：使用**演示模式**（启发式算法，无需训练）
- `--method MAAC-R`：使用 **MAAC-R 配置文件**（`configs/MAAC-R.yaml`）
- `-s 300`：运行 **300 步**仿真（约15秒动画）

**其他常用命令：**

```bash
# 训练50轮，每轮300步，每20轮保存一次
python .\src\main.py --phase train --method MAAC-R -e 50 -s 300 -f 20

# 使用训练好的权重评估性能
python .\src\main.py --phase evaluate --method MAAC-R -s 500 -a ./results/MAAC-R/.../actor_100.pth

# 快速演示（100步，约5秒）
python .\src\main.py --phase run --method MAAC-R -s 100
```

---

### �📍 重要提示

**请确保在 `ME5424` 目录下运行命令**，或使用完整路径：

```bash
# 方法1: 进入ME5424目录
cd ME5424
python .\src\main.py --phase run --method MAAC-R -s 300

# 方法2: 从上级目录运行（使用完整路径）
python .\ME5424\src\main.py --phase run --method MAAC-R -s 300
```

### ⚙️ 环境变量设置（可选）

如果遇到 OpenMP 库冲突警告，请设置：
```bash
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Windows CMD
set KMP_DUPLICATE_LIB_OK=TRUE

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 1️⃣ 训练模式（保存轨迹与曲线）

```bash
python .\src\main.py --phase train --method MAAC-R -e 50 -s 300 -f 20
```

**参数说明：**
- `--phase train`：训练模式，使用神经网络训练智能体并保存权重
- `--method MAAC-R`：使用 MAAC-R 算法（推荐，基于互信息的奖励分配）
- `-e 50`：训练 50 轮（episodes）
- `-s 300`：每轮运行 300 步（steps）
- `-f 20`：每 20 轮保存一次模型和日志

**📁 输出内容：** `results\MAAC-R\{实验ID}\` 下的
- 🎬 `animated\animated_textured_plot_1.mp4` - 训练动画
- 🖼️ `frames\tex_frame_*.png` - 帧图
- 💾 `actor\ / critic\ / pmi\` - 模型权重
- 📍 `t_xy\ / u_xy\ / p_xy\` - 轨迹数据
- 📊 `covered_target_num\` - 覆盖率统计
- 📈 `logs\` - TensorBoard 日志

---

### 2️⃣ 评估模式（使用已训练权重）

```bash
python .\src\main.py --phase evaluate --method MAAC-R -s 300 -a <actor权重路径> -c <critic权重路径> -p <pmi权重路径>
```

**参数说明：**
- `--phase evaluate`：评估模式，加载已训练的权重进行测试
- `-a`：Actor 网络权重文件路径（可选）
- `-c`：Critic 网络权重文件路径（可选）
- `-p`：PMI 网络权重文件路径（可选，仅 MAAC-R 需要）
- 未提供的权重将使用随机初始化

---

### 3️⃣ 演示模式（启发式策略演示）

```bash
python .\src\main.py --phase run --method MAAC-R -s 300
```

**参数说明：**
- `--phase run`：演示模式，使用启发式算法（无需权重）
- `--method MAAC-R`：使用 MAAC-R 配置文件
- `-s 300`：运行 300 步（约15秒动画）

#### 🎯 演示模式详解

**演示模式（Run Mode）** 是一个无需训练权重的快速预览模式，使用内置的启发式算法控制UAV行为。

##### ✨ 主要特点

| 特性 | 说明 |
|------|------|
| 🚀 **即开即用** | 无需训练模型，无需加载权重文件 |
| 🧠 **启发式策略** | 基于贪心算法的智能决策 |
| 🎬 **完整可视化** | 生成动画、轨迹图、统计数据 |
| ⚡ **快速执行** | 单轮运行，适合快速测试和演示 |

##### 🤖 启发式策略原理

演示模式使用 **基于方向的贪心策略**（`get_action_by_direction`）：

```python
策略逻辑：
1. 目标评分 = 距离权重 / 目标距离 - 重复追踪惩罚
2. 选择得分最高的目标
3. 计算朝向该目标的最优转向角度
4. 添加随机探索（ε=0.25）和惯性保持（0.3）
```

**决策因素：**
- 🎯 **目标距离**：优先追踪距离较近的目标
- 🚫 **避免扎堆**：当其他UAV已在追踪某目标时降低该目标优先级
- 🎲 **随机探索**：25%概率随机选择动作，避免局部最优
- 🔄 **惯性保持**：30%概率保持当前方向，避免频繁转向

##### 📊 输出内容

与训练/评估模式相同，包含：
- 🎬 `animated/animated_textured_plot_1.mp4` - 完整动画
- 🖼️ `frames/tex_frame_*.png` - 每帧截图
- 📍 `u_xy/u_xy0.csv` - UAV 轨迹数据
- 📍 `t_xy/t_xy0.csv` - 目标轨迹数据
- 📍 `p_xy/p_xy0.csv` - 保护者轨迹数据
- 📊 `covered_target_num/covered_target_num0.csv` - 覆盖率统计
- 📈 `*_return_list.csv` - 奖励曲线数据

##### 🎮 使用场景

| 场景 | 说明 |
|------|------|
| 🔰 **新手入门** | 快速了解系统行为，无需学习训练流程 |
| 👀 **效果预览** | 在训练前预览环境和智能体交互 |
| 🧪 **参数测试** | 快速测试配置参数的影响 |
| 🎓 **教学演示** | 展示多智能体协作和碰撞弹开效果 |
| 🐛 **调试验证** | 验证环境逻辑和碰撞检测是否正常 |

##### ⚖️ 模式对比

| 对比项 | 训练模式 | 评估模式 | 演示模式 |
|--------|----------|----------|----------|
| **需要权重** | ❌ | ✅ | ❌ |
| **训练网络** | ✅ | ❌ | ❌ |
| **决策方式** | 神经网络 | 神经网络 | 启发式算法 |
| **运行轮数** | 多轮（如50轮）| 单轮 | 单轮 |
| **执行速度** | 慢（需训练） | 中等 | 快 |
| **适用场景** | 训练模型 | 测试性能 | 快速预览 |

##### 💡 示例输出

运行演示模式后，终端会显示：

```
use gpus: 1
-----------------------------------------
|This is the summary of config:
|exp_name       : MAAC-R
|environment    : {'n_uav': 3, 'm_targets': 10, 'n_protectors': 3}
|protector      : {'knockback': 200.0, 'arm_thickness': 100.0}
...
-----------------------------------------
```

完成后在 `results/MAAC-R/{实验ID}/` 查看生成的动画和数据。

##### 🔧 调整演示参数

修改 `src/agent/uav.py` 中的启发式参数：

```python
# 随机探索概率（0.0-1.0）
self.epsilon = 0.25        # 降低可减少随机性

# 惯性保持概率（0.0-1.0）
self.continue_tracing = 0.3  # 降低可增加灵活性

# 奖励权重
target_reward_weight = 1.0        # 距离吸引力
repetition_penalty_weight = 0.8   # 重复追踪惩罚
```

---

### ⚙️ GPU/CPU 切换

在 `src\configs\MAAC-R.yaml` 中设置：

```yaml
# 使用 CPU
gpus: -1

# 使用首个 GPU
first_device: 0
gpus: 1

# 使用多个 GPU
gpus: 2  # 使用前2个GPU
```

---

### 🔧 常用配置快速参考

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `environment.n_uav` | 🚁 无人机数量 | 3 |
| `environment.m_targets` | 🎯 目标数量 | 10 |
| `environment.n_protectors` | 🛡️ 保护者数量 | 3 |
| `uav.dp` | 👁️ 无人机探测半径 | 200 |
| `target.capture_radius` | 📍 目标捕获半径 | 120 |
| `protector.safe_radius` | ⚠️ 保护者安全半径 | 200 |
| `protector.knockback` | 💥 弹开强度（可选） | 200 |
| `protector.arm_thickness` | 🔧 手臂碰撞厚度（可选） | 100 |

**💡 提示：** 修改配置后直接重新运行命令即可生效

---

## ⚙️ 详细配置说明（MAAC-R.yaml）

### 🔝 顶层字段
- `exp_name`：🏷️ 实验名称，参与输出目录命名
- `result_dir`：📂 结果根目录（如 `../results/MAAC-R`）
- `seed`：🎲 随机种子
- `cooperative`：🤝 协同奖励系数，控制自利 vs. 共享比例

### 🌍 环境配置 (`environment`)
- `n_uav`：🚁 UAV 数量
- `m_targets`：🎯 目标数量
- `n_protectors`：🛡️ 保护者数量
- `x_max` / `y_max`：📏 地图尺寸（坐标单位）
- `na`：🎮 离散动作空间维数（策略输出的类别数）

### 🚁 UAV 配置 (`uav`)
- `dt`：⏱️ 时间步长
- `v_max`：🚀 最大速度
- `h_max`：🔄 角速度档位（每步最大转角约为 `π / h_max`）
- `dc`：📡 通信/重复追踪判定半径（用于"重复追踪惩罚"）
- `dp`：👁️ 观测/追踪半径（影响目标跟踪奖励触发）
- `alpha` / `beta` / `gamma` / `omega`：⚖️ 奖励分量权重（线性组合为总奖励）

### 🛡️ 保护者配置 (`protector`)
- `safe_radius`：⚠️ 保护者的安全半径（禁区半径），UAV 进入会受罚
  - 值越大越容易触发惩罚；与动画中保护者"阻挡手臂"的半长度一致

### 🎯 目标配置 (`target`)
- `capture_radius`：📍 目标捕获半径（UAV 到目标距离 ≤ 此值即捕获）
  - 判定逻辑在 `environment.step`，被捕获目标会被隐藏

### 🧠 Actor-Critic 配置 (`actor_critic`)
- `buffer_size`：💾 经验回放容量
- `sample_size`：📊 每次采样数量；为 `0` 时通常按每轮 `num_steps * n_uav` 取样
- `actor_lr` / `critic_lr`：📈 学习率
- `hidden_dim`：🔢 隐藏层维度
- `gamma`：💰 折扣因子（0.95）

### 🔗 PMI 网络配置 (`pmi`)
- `hidden_dim`：🔢 PMI 网络隐藏层维度
- `b2_size`：📦 每轮选择的训练样本池大小
- `batch_size`：📊 训练批大小

---

## 🎁 奖励分量与机制

### 💡 奖励计算公式

主要分量在 `UAV.calculate_raw_reward(...)` 与 `Environment.calculate_rewards(...)`：

```python
总奖励 = α × 目标跟踪奖励 
       - β × 重复追踪惩罚 
       - γ × 边界惩罚 
       - ω × 保护者惩罚
```

### 📊 奖励分量详解

| 分量 | 符号 | 权重 | 触发条件 | 作用 |
|------|------|------|----------|------|
| 🎯 目标跟踪奖励 | `target_tracking_reward` | α=0.6 | 距离目标 ≤ dp | 鼓励靠近目标 |
| 🚫 重复追踪惩罚 | `duplicate_tracking_punishment` | β=0.2 | UAV间距 < dc | 避免多UAV扎堆 |
| ⚠️ 边界惩罚 | `boundary_punishment` | γ=0.2 | 靠近地图边界 | 保持安全距离 |
| 🛡️ 保护者惩罚 | `protector_punishment` | ω=0.5 | 距离 < safe_radius | 避开防御区域 |

### 🔧 裁剪与归一化

- **位置：** `src/utils/data_util.py` 的 `clip_and_normalize(...)`
- **功能：** 当原始值超出 `[floor, ceil]`，会被裁剪到边界并归一化用于训练

---

## 📈 结果输出

### 📁 输出目录结构

```
results/MAAC-R/<exp_name_时间戳_随机ID>/
├── 📄 args.yaml                    # 配置快照
├── 📊 *_return_list.csv            # 训练指标
├── 🎬 animated/                    # 动画文件
├── 🖼️ frames/                      # 帧图文件
├── 📍 t_xy/ u_xy/ p_xy/            # 轨迹数据
├── 📊 covered_target_num/          # 覆盖率统计
└── 📈 logs/                        # TensorBoard 日志
```

### 📊 查看训练日志

```bash
```bash
tensorboard --logdir results/MAAC-R/{实验ID}/logs
```

---

## 💥 碰撞弹开效果

本项目实现了 **UAV 碰到保护者手臂后被物理弹开** 的效果。

### 🎯 工作原理

当 UAV 靠近保护者的防御手臂时：
1. **检测碰撞**: 计算 UAV 到手臂线段的最近距离
2. **触发弹开**: 当距离 < `arm_thickness` 时触发
3. **物理推离**: 沿法线方向推离 UAV，推离距离 ≤ `knockback`
4. **边界保护**: 确保 UAV 不被推出地图边界

### ⚙️ 配置参数

在 `src/configs/MAAC-R.yaml` 中配置：

```yaml
protector:
  safe_radius: 200.0      # 保护者安全半径（手臂长度）
  knockback: 200.0        # 💥 弹开强度（最大推离距离）
  arm_thickness: 100.0    # 🔧 手臂碰撞厚度（判定范围）
```

### 📊 参数调优

| 参数 | 推荐范围 | 效果说明 |
|------|----------|----------|
| `knockback` | 100-300 | 弹开强度，值越大推离越远 |
| `arm_thickness` | 50-150 | 碰撞检测范围，值越大越容易触发 |
| `safe_radius` | 150-250 | 手臂长度（建议 ≥ arm_thickness） |

**💡 禁用碰撞**: 将 `knockback` 和 `arm_thickness` 设为 0 即可

---
```

---

## � 碰撞"弹开"效果 ✅ 已实现

本项目已完整实现 **UAV 碰到保护者手臂后被弹开** 的物理效果！

### 🎯 功能说明

当 UAV 靠近保护者的防御手臂时：
1. **检测碰撞**: 计算 UAV 到手臂线段的最近距离
2. **触发弹开**: 距离 < `arm_thickness` 时触发
3. **物理推离**: 沿法线方向推离，距离可达 `knockback`
4. **边界保护**: 确保 UAV 不被推出地图

### ⚙️ 配置参数

```yaml
protector:
  safe_radius: 200.0      # 保护者安全半径（手臂长度）
  knockback: 200.0        # 💥 弹开强度（最大推离距离）
  arm_thickness: 100.0    # 🔧 手臂碰撞厚度（判定范围）
```

### 🧪 测试碰撞效果

运行测试脚本验证功能：

```bash
# 测试碰撞检测
python test_knockback.py

# 生成可视化演示
python demo_knockback.py
```

### 📊 参数调优建议

| 参数 | 推荐范围 | 效果 |
|------|----------|------|
| `knockback` | 100-300 | 控制弹开强度 |
| `arm_thickness` | 50-150 | 控制碰撞检测范围 |
| `safe_radius` | 150-250 | 控制手臂长度 |

### 🎬 可视化效果

在生成的动画中可观察到：
- 🛡️ 保护者带有两条防御手臂
- 💥 UAV 轨迹在碰撞时突然改变方向
- 📍 UAV 被推离保护者的防御区域

### 📖 详细文档

完整实现说明请查看：[KNOCKBACK_GUIDE.md](KNOCKBACK_GUIDE.md)

---

## 🎮 智能体介绍

### 🚁 UAV（无人机 - 追踪者）
- **速度**: 20 单位/步
- **观测半径**: 200（探测目标范围）
- **通信半径**: 500（协作通信范围）
- **动作空间**: 12个离散转向动作
- **任务**: 协同追踪目标，避开保护者

### 🎯 Target（目标 - 被追踪者）
- **速度**: 5 单位/步
- **捕获半径**: 120（被捕获判定距离）
- **行为**: 随机移动策略
- **状态**: 可被UAV捕获（隐藏）

### 🛡️ Protector（保护者 - 防御者）
- **速度**: 5 单位/步
- **安全半径**: 200（禁区半径）
- **惩罚机制**: UAV进入范围会受到惩罚
- **可选功能**: 物理弹开效果

---

## 🧠 算法框架对比

| 方法 | 描述 | 协作机制 | 适用场景 | 推荐度 |
|------|------|----------|----------|--------|
| **MAAC** | 自利策略 | ❌ 无协作 | 简单环境 | ⭐⭐⭐ |
| **MAAC-G** | 全局奖励 | 🤝 平均分配 | 完全协作 | ⭐⭐⭐⭐ |
| **MAAC-R** | PMI网络 | 🧠 智能分配 | 复杂协作 | ⭐⭐⭐⭐⭐ |
| **C-METHOD** | 对比基准 | - | 性能对比 | ⭐⭐ |

**🎯 推荐使用 MAAC-R**，它通过 PMI 网络实现智能奖励分配，在复杂多智能体环境中表现最优。

---

## 📚 使用技巧

### 🎯 实验建议

1. **🔰 初学者**
   - 先用演示模式了解系统行为
   - 使用默认配置训练 50 轮
   - 观察动画理解智能体策略

2. **🔬 研究者**
   - 调整智能体数量进行实验
   - 修改奖励权重观察影响
   - 比较不同方法的性能

3. **🚀 高级用户**
   - 实现自定义奖励函数
   - 扩展智能体行为模型
   - 集成新的协作机制

### ⚡ 性能优化

- **GPU加速**: 在配置文件中启用 GPU
- **批量大小**: 增加 `batch_size` 提高训练效率
- **经验回放**: 调整 `buffer_size` 平衡内存与性能
- **并行训练**: 使用多个 GPU 加速训练

---

## 🤝 致谢

本项目用于 **ME5424 Swarm Robotics and Aerial Robotics** 课程实验，欢迎在此基础上扩展更复杂的协作策略与物理交互。

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 🐛 Issue: [GitHub Issues](https://github.com/XC-CN/5424Project/issues)
- 💬 Discussion: [GitHub Discussions](https://github.com/XC-CN/5424Project/discussions)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！⭐**

Made with ❤️ by ME5424 Team

</div>
