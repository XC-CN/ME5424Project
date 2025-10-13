# MARL-UAVs-Targets-Tracking

对论文“Improving multi-target cooperative tracking guidance for UAV swarms using multi-agent reinforcement learning”（利用多智能体强化学习改进无人机集群多目标协同跟踪制导）的实现与改进。

![1760356597942](imgs\2d-demo.png)

### 环境配置

可以直接通过 pip 安装所需依赖：

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm tensorboard scipy
pip install imageio[ffmpeg]
```

### 运行代码

```sh
python .\MARL-UAVs-Targets-Tracking\src\main.py
```

### 待办事项

- [X] 原始 MAAC
  - [X] Actor-Critic 框架
- [X] MAAC-R
  - [X] 互惠奖励（结合 PMI 网络）
- [X] MAAC-G
  - [X] 接收全局奖励
- [X] 3D 演示
