# data_process_script.py
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- 1. 请在这里配置 ---

# [输入] 我们刚刚生成的训练原材料所在的目录
# 注意：这个路径是相对于您运行此脚本的位置
SOURCE_ROI_DIR = Path("./training_rois/")

# [输出] 最终用于PaddleOCR训练的标准数据集的输出目录
OUTPUT_DATA_DIR = Path("./train_data/")

# [配置] 验证集所占的比例 (例如 0.2 代表 20%)
VALIDATION_SPLIT_RATIO = 0.1

# --- 配置结束 ---


def process_data():
    """
    整合、清洗、并划分数据集，为PaddleOCR训练做准备。
    """
    print("--- PaddleOCR 数据集预处理脚本 ---")

    # 1. 检查输入目录是否存在
    if not SOURCE_ROI_DIR.is_dir():
        print(f"\n错误: 源目录 '{SOURCE_ROI_DIR}' 未找到。")
        print("请确保您已运行 data_generator_client.py 并且服务端已生成数据。")
        return

    # 2. 创建或清空输出目录
    output_images_dir = OUTPUT_DATA_DIR / "images"
    if OUTPUT_DATA_DIR.exists():
        print(f"\n警告: 输出目录 '{OUTPUT_DATA_DIR}' 已存在，将清空并重新创建。")
        shutil.rmtree(OUTPUT_DATA_DIR)

    print(f"正在创建空的输出目录: {output_images_dir}")
    output_images_dir.mkdir(parents=True)

    # 3. 扫描所有会话目录，整合数据
    master_label_list = []
    session_dirs = [d for d in SOURCE_ROI_DIR.iterdir() if d.is_dir()]

    print(f"\n找到 {len(session_dirs)} 个会话目录，开始整合...")

    with tqdm(total=len(session_dirs), desc="整合会话") as pbar:
        for session_dir in session_dirs:
            label_file = session_dir / "label.txt"
            if not label_file.is_file():
                continue

            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '\t' not in line:
                        continue

                    roi_filename, label_text = line.split('\t', 1)
                    source_roi_path = session_dir / roi_filename

                    if not source_roi_path.is_file():
                        tqdm.write(f"  跳过：图片文件未找到 {source_roi_path}")
                        continue

                    # 创建一个新的、唯一的文件名，避免冲突
                    # 格式: 会话名_原始文件名.png
                    new_roi_filename = f"{session_dir.name}_{roi_filename}"
                    dest_roi_path = output_images_dir / new_roi_filename

                    # 复制图片文件
                    shutil.copyfile(source_roi_path, dest_roi_path)

                    # 生成PaddleOCR格式的标签行： "相对路径\t标签"
                    relative_path = Path("images") / new_roi_filename
                    master_label_list.append(f"{relative_path.as_posix()}\t{label_text}")

            pbar.update(1)

    if not master_label_list:
        print("\n错误: 未能从源目录中收集到任何有效的标签和图片。")
        return

    print(f"\n整合完成！共找到 {len(master_label_list)} 个有效的ROI样本。")

    # 4. 随机打乱并划分数据集
    print("\n正在随机打乱并划分数据集...")
    random.shuffle(master_label_list)

    split_index = int(len(master_label_list) * (1 - VALIDATION_SPLIT_RATIO))
    train_list = master_label_list[:split_index]
    val_list = master_label_list[split_index:]

    # 5. 写入最终的标签文件
    train_label_path = OUTPUT_DATA_DIR / "train_list.txt"
    val_label_path = OUTPUT_DATA_DIR / "val_list.txt"

    print(f"正在写入训练集标签文件 ({len(train_list)} 条)...")
    with open(train_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))

    print(f"正在写入验证集标签文件 ({len(val_list)} 条)...")
    with open(val_label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_list))

    print("\n--- 数据准备完成！ ---")
    print(f"总样本数: {len(master_label_list)}")
    print(f"  - 训练集: {len(train_list)} 条，保存在 {train_label_path}")
    print(f"  - 验证集: {len(val_list)} 条，保存在 {val_label_path}")
    print(f"所有图片文件已统一存放在: {output_images_dir}")
    print("\n下一步：请将整个 'train_data' 文件夹移动到 'PaddleOCR-release-3.0.1' 目录下，然后配置 .yml 文件准备开始训练。")


if __name__ == "__main__":
    process_data()