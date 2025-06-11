# OpenWBC_to_Lerobot

[English](README.md) | [中文](README_CN.md)

A tool for converting OpenWBC format datasets to LeRobot compatible format.

## 🚀 Features

- 🔄 Convert OpenWBC data format to LeRobot compatible format
- 📊 Support multi-dimensional state vectors and action vectors extraction
- 🎬 Automatically generate video files from image sequences
- 📈 Generate complete metadata and statistics
- 🤖 Support multiple robot type configurations

## 📦 Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package

```bash
pip install -e .
```

## 🛠️ Usage

### Basic Usage

```bash
python convert_to_lerobot.py \
    --input_dir /path/to/openwbc/dataset \
    --output_dir ./lerobot_dataset \
    --dataset_name "pick_cola" \
    --robot_type "g1" \
    --fps 30
```

### Command Line Tool

After installation, you can use the command line tool:

```bash
wbc-convert \
    --input_dir /path/to/openwbc/dataset \
    --output_dir ./lerobot_dataset \
    --dataset_name "pick_cola" \
    --robot_type "g1" \
    --fps 30
```

### Parameters

- `--input_dir`: OpenWBC dataset input directory
- `--output_dir`: LeRobot format output directory
- `--dataset_name`: Dataset name
- `--robot_type`: Robot type (default: "g1")
- `--fps`: Video frame rate (default: 30.0)

## 📁 Data Format

### Input Format (OpenWBC)
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

### Output Format (LeRobot)
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

## ⚙️ Modality Configuration

The project includes a `modality.json` configuration file that defines data dimension mappings for each modality:

### State Vector (40 dimensions)
- left_arm: 7D joint positions
- right_arm: 7D joint positions  
- left_hand: 7D joint positions
- right_hand: 7D joint positions
- left_leg: 6D joint positions
- right_leg: 6D joint positions

### Action Vector (32 dimensions)
- left_arm: 7D joint positions
- right_arm: 7D joint positions
- left_hand: 7D joint positions
- right_hand: 7D joint positions
- base_motion: 4D control commands

## 📖 Example Usage

After conversion, you can use the dataset like this:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset('./lerobot_dataset')
print(f'Dataset size: {len(dataset)}')

# Get a sample
sample = dataset[0]
print("State dimensions:", sample['observation.state'].shape)
print("Action dimensions:", sample['action'].shape)
print("Image shape:", sample['observation.images.main'].shape)
```

## 🔧 Advanced Configuration

### Custom Robot Configuration

You can modify the `modality.json` file to support different robot configurations:

```json
{
  "robot_type": "custom_robot",
  "state_dim": 50,
  "action_dim": 40,
  "modalities": {
    "state": {
      "custom_arm": [0, 7],
      "custom_base": [7, 13]
    },
    "action": {
      "custom_arm": [0, 7],
      "custom_base": [7, 13]
    }
  }
}
```

### Batch Processing

For processing multiple datasets:

```python
import os
from pathlib import Path

datasets = [
    "/path/to/dataset1",
    "/path/to/dataset2", 
    "/path/to/dataset3"
]

for dataset_path in datasets:
    dataset_name = Path(dataset_path).name
    os.system(f"""
    python convert_to_lerobot.py \
        --input_dir {dataset_path} \
        --output_dir ./converted/{dataset_name} \
        --dataset_name {dataset_name} \
        --robot_type g1
    """)
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory issues with large datasets**: Use chunk processing for large datasets
   ```bash
   # Process in smaller chunks
   python convert_to_lerobot.py --chunk_size 100 ...
   ```

3. **Video encoding issues**: Install ffmpeg if video generation fails
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu
   sudo apt install ffmpeg
   ```

## 🤝 Contributing

We welcome contributions! Please feel free to submit Issues and Pull Requests to improve this tool.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd OpenWBC_to_Lerobot

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [OpenWBC](https://github.com/your-org/OpenWBC) - Original data collection system
- Community contributors and testers

## 📚 Related Projects

- **[OpenWBC](https://github.com/your-org/OpenWBC)**: Complete robot teleoperation and data collection system
- **[LeRobot](https://github.com/huggingface/lerobot)**: Robot learning framework
- **[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)**: Foundation model for humanoid robots 