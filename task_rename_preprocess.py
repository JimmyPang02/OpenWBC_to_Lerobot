import os
import json
import argparse

def update_goal_in_dataset(dataset_name, task_description):
    # 获取所有以 episode_ 开头的子目录
    for episode in sorted(os.listdir(dataset_name)):
        episode_path = os.path.join(dataset_name, episode)
        data_file = os.path.join(episode_path, "data.json")

        if os.path.isdir(episode_path) and os.path.isfile(data_file):
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 确保 "text" 和 "goal" 存在
                if "text" in data and "goal" in data["text"]:
                    old_goal = data["text"]["goal"]
                    data["text"]["goal"] = task_description

                    with open(data_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)

                    print(f"Updated goal in {data_file}")
                else:
                    print(f"'text' or 'goal' not found in {data_file}")

            except Exception as e:
                print(f"Error processing {data_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace goal text in dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--task", type=str, required=True, help="New goal description to set.")

    args = parser.parse_args()

    update_goal_in_dataset(args.dataset_path, args.task)
