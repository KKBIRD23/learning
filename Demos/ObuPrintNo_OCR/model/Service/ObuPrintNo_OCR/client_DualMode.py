# client_DualMode.py (V8.0_Final)
import requests
import os
import uuid
import time
import cv2
import numpy as np
import json
from datetime import datetime
import base64

# --- 配置 ---
SERVER_URL_PREDICT = "http://127.0.0.1:5000/predict"      # 本机测试地址
SERVER_URL_FINALIZE = "http://127.0.0.1:5000/session/finalize"    # 本机测试地址
SERVER_URL__REFRESH = "http://127.0.0.1:5000/refresh-cache"  # 本机测试地址
# SERVER_URL_PREDICT = "http://172.16.252.18:5000/predict"    #   生产测试环境
# SERVER_URL_FINALIZE = "http://172.16.252.18:5000/session/finalize"  #   生产测试环境
# SERVER_URL__REFRESH = "http://172.16.252.18:5000/refresh-cache"  #   生产测试环境

# 请将这里的路径指向您要测试的图片文件夹
IMAGE_PATHS_TO_UPLOAD = [
    r"../../../../DATA/PIC/1pic/1/1.jpg",
    r"../../../../DATA/PIC/1pic/1/2.jpg",
    r"../../../../DATA/PIC/1pic/1/3.jpg",
    r"../../../../DATA/PIC/1pic/1/4.jpg"
]

# --- 辅助函数：显示Base64编码的图像 ---
def display_base64_image(base64_string: str, window_name: str = "Annotated Frame"):
    """解码Base64字符串并用OpenCV显示图像。"""
    if not base64_string:
        print("  客户端显示：无Base64图像数据可显示。")
        return
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            # 调整窗口大小以适应屏幕
            screen_height, screen_width = 1080, 1920 # 假设一个常见的屏幕分辨率
            img_height, img_width = img.shape[:2]
            scale = min(screen_width / img_width, screen_height / img_height, 1)
            if scale < 1:
                display_width = int(img_width * scale * 0.8) # 留出一些边距
                display_height = int(img_height * scale * 0.8)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, display_width, display_height)
            cv2.imshow(window_name, img)
            cv2.waitKey(100)
        else:
            print(f"  客户端显示：解码Base64图像数据失败。")
    except Exception as e:
        print(f"  客户端显示：显示Base64图像时发生错误: {e}")

# --- 核心请求函数 ---
def send_image_for_prediction(image_path: str, session_id_to_use: str, frame_counter: int):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None
    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {'session_id': session_id_to_use}

            response = requests.post(SERVER_URL_PREDICT, files=files_payload, data=data_payload, timeout=180)
            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', "
                  f"服务端状态码: {response.status_code}")

            if response.status_code == 200:
                response_json = response.json()
                print(f"  服务端消息: {response_json.get('message')}")

                confirmed = response_json.get('confirmed_results', [])
                pending = response_json.get('pending_results', [])
                print(f"  实时结果: {len(confirmed)} 个确信, {len(pending)} 个待定。")

                # 打印确信列表
                if confirmed:
                    print("  --- 确信列表 (目击次数 >= 2) ---")
                    for item in confirmed:
                        print(f"    - {item.get('text')} (目击 {item.get('count')} 次)")

                # 打印待定列表
                if pending:
                    print("  --- 待定列表 (目击次数 = 1) ---")
                    for item in pending:
                        print(f"    - {item.get('text')}")

                current_frame_base64_img = response_json.get('current_frame_annotated_image_base64')
                if current_frame_base64_img:
                    print("  客户端显示：当前帧标注图 (绿:确信, 黄:待定, 红:失败)...")
                    display_base64_image(current_frame_base64_img, f"Frame {frame_counter} - Annotated")
                else:
                    print("  客户端显示：未收到当前帧标注图数据。")

                return response_json
            else:
                try: print(f"  服务端错误详情: {response.json()}")
                except requests.exceptions.JSONDecodeError: print(f"  服务端原始响应 (非JSON): {response.text}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"客户端：会话 {session_id_to_use}, 请求图片 '{os.path.basename(image_path)}' 时发生网络请求错误: {e}")
        return None

def finalize_session(session_id_to_finalize: str):
    """调用终审接口获取最终结果。"""
    print("\n" + "="*40)
    print(f"客户端：所有图片已发送，正在为会话 {session_id_to_finalize} 请求最终结果...")
    print("="*40)
    try:
        headers = {'Content-Type': 'application/json'}
        data_payload = json.dumps({'session_id': session_id_to_finalize})
        response = requests.post(SERVER_URL_FINALIZE, data=data_payload, headers=headers, timeout=60)

        print(f"客户端：终审接口状态码: {response.status_code}")
        if response.status_code == 200:
            response_json = response.json()
            print(f"  服务端消息: {response_json.get('message')}")
            final_count = response_json.get('total_count', 0)
            final_results = response_json.get('final_results', [])
            print(f"\n  会话终审完成！最终识别总数: {final_count} 个。")
            print("  --- 最终OBU列表 ---")
            for idx, item in enumerate(final_results):
                print(f"    {idx+1}. {item.get('text')} (总目击 {item.get('count')} 次)")
            return response_json
        else:
            try: print(f"  服务端错误详情: {response.json()}")
            except requests.exceptions.JSONDecodeError: print(f"  服务端原始响应 (非JSON): {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"客户端：请求终审接口时发生网络请求错误: {e}")
        return None

def refresh_server_cache(api_key: str):
    """
    调用 /refresh-cache 接口来触发服务端的缓存刷新。
    """

    print("\n" + "="*40)
    print(f"客户端：正在调用缓存刷新接口...")
    print("="*40)

    try:
        # --- 核心：在请求头中加入 X-API-KEY ---
        headers = {
            'X-API-KEY': api_key
        }

        # 发送POST请求，不需要任何请求体
        response = requests.post(SERVER_URL__REFRESH, headers=headers, timeout=10)

        print(f"客户端：缓存刷新接口状态码: {response.status_code}")

        # 检查响应内容
        try:
            response_json = response.json()
            print(f"  服务端消息: {response_json.get('message')}")
            if response.status_code == 202:
                print("  操作成功：服务端已接受刷新任务，并将在后台执行。")
            elif response.status_code == 200:
                 print(f"  操作成功：缓存已同步刷新。新数量: {response_json.get('count')}")
            else:
                print(f"  操作失败详情: {response_json.get('error')}")

        except requests.exceptions.JSONDecodeError:
            print(f"  服务端返回了非JSON格式的响应: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"客户端：请求缓存刷新接口时发生网络错误: {e}")

if __name__ == "__main__":
    # --- 步骤 1: 定义会话和安全密钥 ---
    current_batch_session_id = str(uuid.uuid4())
    # 请确保这个API密钥与您服务端 config.py 中 REFRESH_API_KEY 的值一致
    API_KEY_TO_USE = "Vfj@1234.wq"

    print(f"客户端：开始新的扫描会话，ID: {current_batch_session_id}")

    # --- 步骤 2: (可选) 在扫描前，先触发一次缓存刷新 ---
    # 您可以取消下面这行代码的注释，来测试刷新功能
    # refresh_server_cache(API_KEY_TO_USE)
    # input("缓存刷新请求已发送，按回车开始图片识别...")

    # --- 步骤 3: 循环发送图片 ---
    frame_count = 0
    for img_path in IMAGE_PATHS_TO_UPLOAD:
        frame_count += 1
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 未找到，跳过。")
            continue

        image_basename = os.path.basename(img_path)
        print(f"\n客户端：准备发送图片 {frame_count}/{len(IMAGE_PATHS_TO_UPLOAD)}: '{image_basename}'")

        send_image_for_prediction(
            img_path,
            current_batch_session_id,
            frame_count
        )

        if frame_count < len(IMAGE_PATHS_TO_UPLOAD):
            print("-" * 30)

    # --- 步骤 4: 调用终审接口 ---
    finalize_session(current_batch_session_id)

    # --- 步骤 5:(可选)在所有操作后，再次触发缓存刷新进行测试 ---
    # 可以取消下面这行代码的注释，来再次测试刷新功能
    refresh_server_cache(API_KEY_TO_USE)

    cv2.destroyAllWindows()
    print("\n客户端测试完成。")