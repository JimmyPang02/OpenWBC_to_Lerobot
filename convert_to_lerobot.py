#!/usr/bin/env python3
"""
将OpenWBC格式的数据集转换为LeRobot格式

用法:
python convert_to_lerobot.py \
    --input_dir /Users/jimmyp/Documents/project/OpenWBC/pick_cola \
    --output_dir ./lerobot_dataset \
    --dataset_name "pick_cola" \
    --robot_type "g1" \
    --fps 30
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import ast

def parse_json_data(json_file: Path) -> Dict[str, Any]:
    """解析data.json文件"""
    print(f"解析文件: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_state_vector(states_dict):
    """从states字典中提取40维状态向量"""
    # 按顺序提取: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) + left_leg(6) + right_leg(6)
    state_parts = []
    
    # left_arm qpos (7维)
    if 'left_arm' in states_dict and 'qpos' in states_dict['left_arm']:
        state_parts.extend(states_dict['left_arm']['qpos'])
    
    # right_arm qpos (7维)  
    if 'right_arm' in states_dict and 'qpos' in states_dict['right_arm']:
        state_parts.extend(states_dict['right_arm']['qpos'])
    
    # left_hand qpos (7维)
    if 'left_hand' in states_dict and 'qpos' in states_dict['left_hand']:
        state_parts.extend(states_dict['left_hand']['qpos'])
    
    # right_hand qpos (7维)
    if 'right_hand' in states_dict and 'qpos' in states_dict['right_hand']:
        state_parts.extend(states_dict['right_hand']['qpos'])
    
    # left_leg qpos (6维)
    if 'left_leg' in states_dict and 'qpos' in states_dict['left_leg']:
        state_parts.extend(states_dict['left_leg']['qpos'])
    
    # right_leg qpos (6维)
    if 'right_leg' in states_dict and 'qpos' in states_dict['right_leg']:
        state_parts.extend(states_dict['right_leg']['qpos'])
    
    return np.array(state_parts, dtype=np.float64)

def extract_action_vector(actions_dict):
    """从actions字典中提取32维动作向量"""
    # 按顺序提取: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7) + controller_command(4)
    action_parts = []
    
    # left_arm qpos (7维)
    if 'left_arm' in actions_dict and 'qpos' in actions_dict['left_arm']:
        action_parts.extend(actions_dict['left_arm']['qpos'])
    
    # right_arm qpos (7维)
    if 'right_arm' in actions_dict and 'qpos' in actions_dict['right_arm']:
        action_parts.extend(actions_dict['right_arm']['qpos'])
    
    # left_hand qpos (7维)
    if 'left_hand' in actions_dict and 'qpos' in actions_dict['left_hand']:
        action_parts.extend(actions_dict['left_hand']['qpos'])
    
    # right_hand qpos (7维)
    if 'right_hand' in actions_dict and 'qpos' in actions_dict['right_hand']:
        action_parts.extend(actions_dict['right_hand']['qpos'])
    
    # controller command (4维) - 使用left_leg的qpos (因为left_leg和right_leg是相同的遥控器command)
    if 'left_leg' in actions_dict and 'qpos' in actions_dict['left_leg']:
        # 这里qpos可能是嵌套列表 [[x,y,yaw,z]]，需要展平
        controller_cmd = actions_dict['left_leg']['qpos']
        if isinstance(controller_cmd[0], list):
            controller_cmd = controller_cmd[0]  # 取第一个元素
        action_parts.extend(controller_cmd)
    
    return np.array(action_parts, dtype=np.float64)

def convert_episode(input_episode_dir: Path, episode_idx: int, output_dir: Path, fps: float, task_map: Dict[str, int] = None) -> Dict[str, Any]:
    """转换单个episode"""
    print(f"转换episode {episode_idx}...")
    
    # 读取JSON数据
    json_path = input_episode_dir / "data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"找不到data.json文件: {json_path}")
    
    episode_data = parse_json_data(json_path)
    
    # 检查必要的字段
    if "data" not in episode_data:
        raise ValueError(f"Episode {episode_idx}: data.json缺少data字段")
    
    data_frames = episode_data["data"]
    task_info = episode_data.get("text", {})
    
    # 创建episode数据
    episode_length = len(data_frames)
    frames_data = []
    
    # 获取图像文件列表
    colors_dir = input_episode_dir / "colors"
    if colors_dir.exists():
        color_files = sorted([f for f in colors_dir.glob("*.jpg") if f.is_file()])
    else:
        color_files = []
    
    for frame_idx, frame_data in enumerate(data_frames):
        # 确定task_index
        task_description = task_info.get("goal", "default_task") if task_info else "default_task"
        current_task_index = task_map.get(task_description, 0) if task_map else 0
        
        processed_frame = {
            "episode_index": episode_idx,
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps,  # 时间戳必须与视频帧精确对应：第frame_idx帧 = frame_idx/fps秒
            "task_index": current_task_index,
        }
        
        # 添加states数据作为observation.state
        if "states" in frame_data and frame_data["states"]:
            # 根据实际数据结构调整，这里假设states是个字典或列表
            states = frame_data["states"]
            if isinstance(states, dict):
                # 如果是字典，提取left_arm的数据
                if "left_arm" in states:
                    processed_frame["observation.state"] = extract_state_vector(states)
                else:
                    # 转换为列表
                    processed_frame["observation.state"] = extract_state_vector(states)
            else:
                # 如果已经是列表
                processed_frame["observation.state"] = extract_state_vector(states)
        
        # 添加图像路径（如果存在）
        if frame_idx < len(color_files):
            # 这里我们先记录原始图像路径，稍后会转换为视频
            processed_frame["image_path"] = str(color_files[frame_idx])
        
        # 添加action数据
        if "actions" in frame_data and frame_data["actions"]:
            actions = frame_data["actions"]
            if isinstance(actions, dict):
                # 如果是字典，提取left_arm的数据
                if "left_arm" in actions:
                    processed_frame["action"] = extract_action_vector(actions)
                else:
                    # 转换为列表
                    processed_frame["action"] = extract_action_vector(actions)
            else:
                # 如果已经是列表
                processed_frame["action"] = extract_action_vector(actions)
        
        # 添加episode结束标志和next.reward
        processed_frame["next.done"] = frame_idx == episode_length - 1
        processed_frame["next.reward"] = 0.0  # 添加默认reward
        
        # 为了兼容GR00T格式，添加任务描述相关字段
        if task_info and "goal" in task_info:
            # 根据官方格式，task_description应该是task_index（int64）
            processed_frame["annotation.human.action.task_description"] = current_task_index
        
        # 添加annotation validity字段，表示人工标注的有效性
        processed_frame["annotation.human.validity"] = 1
        
        frames_data.append(processed_frame)
    
    return {
        "frames": frames_data,
        "length": episode_length,
        "episode_index": episode_idx,
        "task_info": task_info
    }

def create_videos_from_images(input_dir: Path, output_videos_dir: Path, episode_data: List[Dict], fps: float, code_type: str='h264'):
    """从图像序列创建视频文件，返回图像尺寸"""
    print("创建视频文件...")
    
    image_shape = None  # 用于存储图像尺寸信息
    
    # 为每个episode创建视频
    for ep_data in episode_data:
        episode_idx = ep_data["episode_index"]
        episode_dir = input_dir / f"episode_{episode_idx+1:04d}"  # episode目录从1开始编号
        colors_dir = episode_dir / "colors"
        
        if not colors_dir.exists():
            print(f"警告: Episode {episode_idx} 没有colors目录，跳过视频创建")
            continue
        
        # 获取图像文件
        image_files = sorted([f for f in colors_dir.glob("*.jpg") if f.is_file()])
        if not image_files:
            print(f"警告: Episode {episode_idx} 没有图像文件，跳过视频创建")
            continue
        
        # 确定视频输出路径
        video_output_dir = output_videos_dir / "chunk-000" / "observation.images.ego_view"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_output_dir / f"episode_{episode_idx:06d}.mp4"
        
        # 读取第一张图像确定尺寸
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"警告: 无法读取图像 {image_files[0]}，跳过episode {episode_idx}")
            continue
        
        height, width = first_img.shape[:2]
        
        # 如果这是第一次获取图像尺寸，记录下来
        if image_shape is None:
            image_shape = [3, height, width]  # RGB, 高度, 宽度
        
        # 创建视频写入器 - 使用H.264编码以获得更好的兼容性
        if code_type=='h264':
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        elif code_type=='mp4v':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        try:
            # 按顺序写入图像，确保视频帧索引与原始frame_idx对应
            # 第frame_idx张图像 → 视频第frame_idx帧 → timestamp = frame_idx/fps
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    video_writer.write(img)
                else:
                    print(f"警告: 无法读取图像 {img_path}")
            
            print(f"创建视频: {video_path}")
        finally:
            video_writer.release()
    
    return image_shape

def create_parquet_files(episode_data: List[Dict], output_data_dir: Path, videos_dir: Path):
    """创建Parquet数据文件"""
    print("创建Parquet数据文件...")
    
    data_output_dir = output_data_dir / "chunk-000"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_frames = []
    global_index = 0
    
    for ep_data in episode_data:
        episode_idx = ep_data["episode_index"]
        
        for frame in ep_data["frames"]:
            frame_data = frame.copy()
            frame_data["index"] = global_index
            
            # 删除图像路径字段，因为parquet中不存储图像路径
            if "image_path" in frame_data:
                del frame_data["image_path"]  # 删除临时字段
            
            all_frames.append(frame_data)
            global_index += 1
    
    # 创建DataFrame并保存每个episode
    for ep_data in episode_data:
        episode_idx = ep_data["episode_index"]
        
        # 筛选该episode的frames
        episode_frames = [f for f in all_frames if f["episode_index"] == episode_idx]
        
        if episode_frames:
            df = pd.DataFrame(episode_frames)
            parquet_path = data_output_dir / f"episode_{episode_idx:06d}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"保存数据文件: {parquet_path}")

def create_metadata_files(episode_data: List[Dict], output_dir: Path, dataset_name: str, robot_type: str, fps: float, image_shape=None, code_type: str='h264'):
    """创建元数据文件"""
    print("创建元数据文件...")
    
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    total_episodes = len(episode_data)
    total_frames = sum(ep["length"] for ep in episode_data)
    
    # 从第一个episode获取数据维度信息
    if episode_data:
        first_frame = episode_data[0]["frames"][0]
        state_dim = len(first_frame.get("observation.state", []))
        action_dim = len(first_frame.get("action", []))
    else:
        state_dim = action_dim = 7  # 默认值
    
    # 设置图像形状，如果没有提供则使用默认值
    if image_shape is None:
        image_shape = [3, 480, 640]  # 默认形状：RGB, 高度, 宽度
    
    # 收集任务数量
    unique_tasks = set()
    for ep_data in episode_data:
        task_info = ep_data.get("task_info", {})
        task_description = task_info.get("goal", f"{dataset_name}_task")
        unique_tasks.add(task_description)
    
    # 创建info.json - 基于NVIDIA Isaac GR00T官方格式
    # 调整图像shape为[height, width, channel]格式
    if image_shape:
        video_shape = [image_shape[1], image_shape[2], image_shape[0]]  # [height, width, channel]
    else:
        video_shape = [480, 640, 3]
    
    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(unique_tasks),
        "total_videos": 1,  # 我们只有一个视频流
        "total_chunks": 1,  # 使用单个chunk
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": video_shape,
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": code_type,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [state_dim],
                "names": [f"motor_{i}" for i in range(state_dim)]
            },
            "action": {
                "dtype": "float64", 
                "shape": [action_dim],
                "names": [f"motor_{i}" for i in range(action_dim)]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1]
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "annotation.human.validity": {
                "dtype": "int64",
                "shape": [1]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "index": {
                "dtype": "int64",
                "shape": [1]
            },
            "next.reward": {
                "dtype": "float64",
                "shape": [1]
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1]
            }
        }
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # 创建modality.json
    modality = {
        "state": {
            "left_arm": {
                "start": 0,
                "end": 7
            },
            "right_arm": {
                "start": 7,
                "end": 14
            },
            "left_hand": {
                "start": 14,
                "end": 21
            },
            "right_hand": {
                "start": 21,
                "end": 28
            },
            "left_leg": {
                "start": 28,
                "end": 34
            },
            "right_leg": {
                "start": 34,
                "end": 40
            }
        },
        "action": {
            "left_arm": {
                "start": 0,
                "end": 7
            },
            "right_arm": {
                "start": 7,
                "end": 14
            },
            "left_hand": {
                "start": 14,
                "end": 21
            },
            "right_hand": {
                "start": 21,
                "end": 28
            },
            "base_motion": {
                "start": 28,
                "end": 32
            }
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            }
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {}
        }
    }
    
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    
    # 创建episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_data in episode_data:
            # 获取任务描述
            task_info = ep_data.get("task_info", {})
            task_description = task_info.get("goal", f"{dataset_name}_task")
            
            episode_info = {
                "episode_index": ep_data["episode_index"],
                "length": ep_data["length"],
                "tasks": [task_description]
            }
            f.write(json.dumps(episode_info) + "\n")
    
    # 创建tasks.jsonl  
    with open(meta_dir / "tasks.jsonl", "w") as f:
        # 收集所有唯一的任务
        unique_tasks = set()
        for ep_data in episode_data:
            task_info = ep_data.get("task_info", {})
            task_description = task_info.get("goal", f"{dataset_name}_task")
            unique_tasks.add(task_description)
        
        for idx, task in enumerate(unique_tasks):
            task_info = {
                "task_index": idx,
                "task": task
            }
            f.write(json.dumps(task_info) + "\n")
    
    # 计算真实的episode统计
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for ep_data in episode_data:
            episode_idx = ep_data["episode_index"]
            frames = ep_data["frames"]
            
            # 提取observation.state和action数据
            obs_states = []
            actions = []
            
            for frame in frames:
                if "observation.state" in frame and frame["observation.state"] is not None:
                    obs_state = frame["observation.state"]
                    # 如果是数组类型，直接添加
                    if isinstance(obs_state, np.ndarray):
                        obs_states.append(obs_state.tolist())
                    # 如果是字典格式，提取qpos
                    elif isinstance(obs_state, dict) and "qpos" in obs_state:
                        obs_states.append(obs_state["qpos"])
                    elif isinstance(obs_state, list):
                        obs_states.append(obs_state)
                if "action" in frame and frame["action"] is not None:
                    action = frame["action"]
                    # 如果是数组类型，直接添加
                    if isinstance(action, np.ndarray):
                        actions.append(action.tolist())
                    # 如果是字典格式，提取qpos
                    elif isinstance(action, dict) and "qpos" in action:
                        actions.append(action["qpos"])
                    elif isinstance(action, list):
                        actions.append(action)
            
            # 计算统计数据
            stats = {
                "episode_index": episode_idx,
                "stats": {}
            }
            
            # 计算observation.state统计
            if obs_states:
                # 确保obs_states是一个数值数组
                valid_obs_states = []
                for obs in obs_states:
                    if obs and isinstance(obs, list) and len(obs) > 0:
                        # 确保是数值类型
                        numeric_obs = [float(x) for x in obs if isinstance(x, (int, float))]
                        if numeric_obs:
                            valid_obs_states.append(numeric_obs)
                
                if valid_obs_states:
                    obs_array = np.array(valid_obs_states, dtype=np.float64)
                    stats["stats"]["observation.state"] = {
                        "max": obs_array.max(axis=0).tolist(),
                        "min": obs_array.min(axis=0).tolist(),
                        "mean": obs_array.mean(axis=0).tolist(),
                        "std": obs_array.std(axis=0).tolist()
                    }
                else:
                    # 如果没有有效数据，使用默认值
                    stats["stats"]["observation.state"] = {
                        "max": [0.0] * state_dim,
                        "min": [0.0] * state_dim,
                        "mean": [0.0] * state_dim,
                        "std": [0.0] * state_dim
                    }
            else:
                # 如果没有数据，使用默认值
                stats["stats"]["observation.state"] = {
                    "max": [0.0] * state_dim,
                    "min": [0.0] * state_dim,
                    "mean": [0.0] * state_dim,
                    "std": [0.0] * state_dim
                }
            
            # 计算action统计
            if actions:
                # 确保actions是一个数值数组
                valid_actions = []
                for action in actions:
                    if action and isinstance(action, list) and len(action) > 0:
                        # 确保是数值类型
                        numeric_action = [float(x) for x in action if isinstance(x, (int, float))]
                        if numeric_action:
                            valid_actions.append(numeric_action)
                
                if valid_actions:
                    action_array = np.array(valid_actions, dtype=np.float64)
                    stats["stats"]["action"] = {
                        "max": action_array.max(axis=0).tolist(),
                        "min": action_array.min(axis=0).tolist(),
                        "mean": action_array.mean(axis=0).tolist(),
                        "std": action_array.std(axis=0).tolist()
                    }
                else:
                    # 如果没有有效数据，使用默认值
                    stats["stats"]["action"] = {
                        "max": [0.0] * action_dim,
                        "min": [0.0] * action_dim,
                        "mean": [0.0] * action_dim,
                        "std": [0.0] * action_dim
                    }
            else:
                # 如果没有数据，使用默认值
                stats["stats"]["action"] = {
                    "max": [0.0] * action_dim,
                    "min": [0.0] * action_dim,
                    "mean": [0.0] * action_dim,
                    "std": [0.0] * action_dim
                }
            
            f.write(json.dumps(stats) + "\n")
    
    print("元数据文件创建完成")

def main():
    parser = argparse.ArgumentParser(description="将OpenWBC数据集转换为LeRobot格式")
    parser.add_argument("--input_dir", type=str, required=True, help="输入数据集目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--robot_type", type=str, default="g1", help="机器人类型")
    parser.add_argument("--fps", type=float, default=30.0, help="视频帧率")
    parser.add_argument("--video_enc", type=str, default='h264', help="视频编码格式, 支持h264或mp4v")
    parser.add_argument("--filter_file", type=str, default="filter.txt", help="包含允许的episode编号列表的文件名 (相对于input_dir)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    print('video:', args.video_enc)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filter_file_path = input_dir / args.filter_file
    filter_episode_numbers = None
    if filter_file_path.exists():
        try:
            with open(filter_file_path, 'r', encoding='utf-8') as f_filter:
                file_content = f_filter.read().strip()
                filter_episode_numbers = ast.literal_eval(file_content)
                if not isinstance(filter_episode_numbers, list) or not all(isinstance(n, int) for n in filter_episode_numbers):
                    print(f"警告: {filter_file_path} 内容格式不正确, 应为整数列表。将处理所有episode。")
                    filter_episode_numbers = None
                else:
                    print(f"从 {filter_file_path} 加载了 {len(filter_episode_numbers)} 个要处理的episode编号。")
        except Exception as e:
            print(f"警告: 读取或解析 {filter_file_path} 失败: {e}。将处理所有episode。")
            filter_episode_numbers = None
    else:
        print(f"提示: 过滤文件 {filter_file_path} 未找到。将尝试处理所有episode。")

    all_potential_episode_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    episode_dirs_to_process = []
    if filter_episode_numbers is not None:
        for ep_dir in all_potential_episode_dirs:
            try:
                # Assuming episode name is like "episode_01" or "episode_1"
                ep_num_str = ep_dir.name.split('_')[-1]
                ep_num = int(ep_num_str)
                if ep_num in filter_episode_numbers:
                    episode_dirs_to_process.append(ep_dir)
            except ValueError:
                print(f"警告: 无法从目录名 {ep_dir.name} 解析episode编号。")
        print(f"根据过滤文件，将处理 {len(episode_dirs_to_process)} 个episodes。")
    else:
        episode_dirs_to_process = all_potential_episode_dirs
        print(f"将处理所有找到的 {len(episode_dirs_to_process)} 个episodes。")

    if not episode_dirs_to_process:
        print(f"错误: 在 {input_dir} 中找不到符合条件的episode目录进行处理。")
        return
    
    # 首先收集所有任务信息以创建task_map
    print("分析任务信息...")
    unique_tasks = set()
    for episode_dir in episode_dirs_to_process:
        data_json = episode_dir / "data.json"
        if data_json.exists():
            try:
                episode_data_raw = parse_json_data(data_json)
                task_info = episode_data_raw.get("text", {})
                task_description = task_info.get("goal", "default_task")
                unique_tasks.add(task_description)
            except Exception as e:
                print(f"警告: 读取episode {episode_dir} 的任务信息失败: {e}")
                unique_tasks.add("default_task")
    
    # 创建task_map
    task_map = {task: idx for idx, task in enumerate(sorted(unique_tasks))}
    print(f"发现 {len(task_map)} 个不同的任务")
    
    # 转换每个episode
    episode_data = []
    for i, episode_dir in enumerate(episode_dirs_to_process):
        try:
            ep_data = convert_episode(episode_dir, i, output_dir, args.fps, task_map)
            episode_data.append(ep_data)
        except Exception as e:
            print(f"错误: 转换episode {episode_dir} 失败: {e}")
            continue
    
    if not episode_data:
        print("错误: 没有成功转换任何episode")
        return
    
    # 创建视频文件
    videos_dir = output_dir / "videos"
    image_shape = create_videos_from_images(input_dir, videos_dir, episode_data, args.fps, code_type=args.video_enc)

    
    # 创建Parquet数据文件  
    data_dir = output_dir / "data"
    create_parquet_files(episode_data, data_dir, videos_dir)
    
    # 创建元数据文件
    create_metadata_files(episode_data, output_dir, args.dataset_name, args.robot_type, args.fps, image_shape, code_type=args.video_enc)
    
    print(f"\n✅ 转换完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 转换了 {len(episode_data)} 个episodes")
    print(f"🎬 总帧数: {sum(ep['length'] for ep in episode_data)}")
    
    print(f"\n📋 使用方法:")
    print(f"from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
    print(f"dataset = LeRobotDataset('{output_dir}')")
    print(f"print(f'数据集大小: {{len(dataset)}}')")

if __name__ == "__main__":
    main() 