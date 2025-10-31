# 🚁 MARL UAVs-Targets-Tracking

> 🎯 一个基于多智能体强化学习（Multi-Agent Actor-Critic, MAAC）的无人机与目标追踪仿真项目

环境中包含 **🚁 UAV**、**🎯 目标（Target）** 与 **🛡️ 保护者（Protector）**，通过奖励与惩罚设计推动 UAV 学习高效追踪目标并避免进入保护者的"禁区"。

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

## 📦 环境依赖

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

### 1️⃣ 训练模式（保存轨迹与曲线）

```bash
python .\src\main.py --phase train --method MAAC-R -e 50 -s 300 -f 20
```

**参数说明：**
- `-e 50` ：训练轮数（episodes）
- `-s 300` ：每轮步数（steps）
- `-f 20` ：保存/日志频率

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

**说明：** 可只填已有的权重；未提供则使用随机初始化

---

### 3️⃣ 演示模式（内置启发式策略，无需权重）

```bash
python .\src\main.py --phase run --method MAAC-R -s 300
```

**说明：** 保存的动画和帧与训练/评估一致，适合快速预览

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
tensorboard --logdir results/MAAC-R/{实验ID}/logs
```

---

## 🔬 可选扩展：碰撞"弹开"效果

若希望实现 **"UAV 碰到保护者的手臂后被弹开"** 的物理效果：

### 🎯 配置新增

在 `config` 中新增：
- `protector.knockback`：💥 弹开距离
- `protector.arm_thickness`：🔧 手臂碰撞厚度

### 🔧 实现方式

在 `environment.step(...)` 中实现：
1. 计算点到线段的最近距离（碰撞检测）
2. 沿法线方向推离 UAV
3. 推离距离不超过 `knockback` 参数

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
