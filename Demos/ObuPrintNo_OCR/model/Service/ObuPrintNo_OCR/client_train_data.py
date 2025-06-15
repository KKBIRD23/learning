# data_generator_client.py
import requests
import os
import uuid
from pathlib import Path
from tqdm import tqdm
import time

# --- 1. 请在这里配置 ---

# 您部署的OBU识别服务的地址
# SERVER_URL_PREDICT = "http://127.0.0.1:5000/predict"
SERVER_URL_PREDICT = "http://172.19.205.247:5000/predict"

# 包含您所有原始OBU图片的总文件夹路径 (可以有多个)
# 脚本会自动递归扫描这些文件夹下所有的图片
SOURCE_IMAGE_DIRS = [
    r"../../../../DATA/PIC",
    # r"D:\OBU_Raw_Images\Batch2",
]

# 允许的图片文件扩展名
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# --- 配置结束 ---


def find_all_image_files(source_dirs: list[str]) -> list[Path]:
    """递归查找所有指定目录下的图片文件。"""
    image_paths = []
    for source_dir in source_dirs:
        p = Path(source_dir)
        if not p.is_dir():
            print(f"警告: 目录 '{source_dir}' 不存在，已跳过。")
            continue
        print(f"正在扫描目录: {p}...")
        for file_path in p.rglob('*'):
            if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                image_paths.append(file_path)
    return image_paths

def trigger_server_processing(image_path: Path):
    """
    将单张图片发送到服务器，触发其进行处理和保存ROI。
    为每次请求生成一个独立的会话ID，以保持逻辑清晰。
    """
    # 使用一个基于文件名的、可读的 session_id，方便追溯
    session_id = f"gen_{image_path.stem}_{uuid.uuid4().hex[:8]}"

    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (image_path.name, f, 'image/jpeg')}
            data_payload = {'session_id': session_id}

            response = requests.post(SERVER_URL_PREDICT, files=files_payload, data=data_payload, timeout=180)

            if response.status_code == 200:
                # 服务端成功处理，我们的目的就达到了
                return True, f"OK ({response.status_code})"
            else:
                # 记录服务端返回的错误信息
                error_info = response.text
                return False, f"FAIL ({response.status_code}): {error_info[:150]}"

    except requests.exceptions.RequestException as e:
        return False, f"FAIL (Network Error): {e}"
    except Exception as e:
        return False, f"FAIL (Client Error): {e}"

if __name__ == "__main__":
    print("--- OBU训练数据生成客户端 ---")
    print(f"将向此服务器发送请求: {SERVER_URL_PREDICT}")

    # 1. 查找所有图片
    all_images = find_all_image_files(SOURCE_IMAGE_DIRS)
    if not all_images:
        print("\n错误: 在指定目录中未找到任何图片文件。请检查 SOURCE_IMAGE_DIRS 配置。")
        exit()

    print(f"\n共找到 {len(all_images)} 张图片。准备开始处理...")

    # 2. 循环处理
    success_count = 0
    fail_count = 0
    with tqdm(total=len(all_images), desc="处理进度") as pbar:
        for image_file in all_images:
            pbar.set_description(f"处理中: {image_file.name}")

            is_success, status_msg = trigger_server_processing(image_file)

            if is_success:
                success_count += 1
            else:
                fail_count += 1
                # 打印详细错误，以便排查
                tqdm.write(f"[失败] 图片: {image_file.name} -> {status_msg}")

            pbar.update(1)
            # 加一个小延时，避免瞬间请求量过大冲垮服务
            time.sleep(0.05)

    print("\n--- 处理完成 ---")
    print(f"成功处理: {success_count} 张")
    print(f"失败处理: {fail_count} 张")
    print("\n所有ROI图片和标签已由服务端自动保存在其 'process_photo/training_rois/' 目录下。")
    print("下一步：请运行 data_process_script.py 脚本来整合这些生成的数据。")