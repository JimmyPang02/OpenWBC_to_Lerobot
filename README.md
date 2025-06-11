# OpenWBC_to_Lerobot

将OpenWBC格式的数据集转换为LeRobot格式的工具。

## 功能特性

- 🔄 将OpenWBC数据格式转换为LeRobot兼容格式
- 📊 支持多维度状态向量和动作向量提取
- 🎬 自动从图像序列生成视频文件
- 📈 生成完整的元数据和统计信息
- 🤖 支持多种机器人类型配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python convert_to_lerobot.py \
    --input_dir /path/to/openwbc/dataset \
    --output_dir ./lerobot_dataset \
    --dataset_name "pick_cola" \
    --robot_type "g1" \
    --fps 30
```

### 参数说明

- `--input_dir`: OpenWBC数据集输入目录
- `--output_dir`: LeRobot格式输出目录
- `--dataset_name`: 数据集名称
- `--robot_type`: 机器人类型 (默认: "g1")
- `--fps`: 视频帧率 (默认: 30.0)

## 数据格式说明

### 输入格式 (OpenWBC)
```
dataset/
├── episode_0001/
│   ├── data.json
│   └── colors/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── ...
├── episode_0002/
│   └── ...
```

### 输出格式 (LeRobot)
```
lerobot_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.main/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    └── episodes_stats.jsonl
```

## 模态配置

项目包含 `modality.json` 配置文件，定义了各个模态的数据维度映射：

### 状态向量 (40维)
- left_arm: 7维关节位置
- right_arm: 7维关节位置  
- left_hand: 7维关节位置
- right_hand: 7维关节位置
- left_leg: 6维关节位置
- right_leg: 6维关节位置

### 动作向量 (32维)
- left_arm: 7维关节位置
- right_arm: 7维关节位置
- left_hand: 7维关节位置
- right_hand: 7维关节位置
- base_motion: 4维控制指令

## 示例代码

转换完成后，可以这样使用数据集：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset('./lerobot_dataset')
print(f'数据集大小: {len(dataset)}')

# 获取一个样本
sample = dataset[0]
print("状态维度:", sample['observation.state'].shape)
print("动作维度:", sample['action'].shape)
print("图像形状:", sample['observation.images.main'].shape)
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。 