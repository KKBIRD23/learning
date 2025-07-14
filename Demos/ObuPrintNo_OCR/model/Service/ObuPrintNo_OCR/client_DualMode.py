# client_DualMode.py (V21.4 - "Libra")
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
# SERVER_URL_PREDICT = "http://127.0.0.1:5000/predict"
# SERVER_URL_FINALIZE = "http://127.0.0.1:5000/session/finalize"
# SERVER_URL_CONFIRM = "http://127.0.0.1:5000/session/confirm_segment"
# SERVER_URL_REFRESH = "http://127.0.0.1:5000/refresh-cache"
SERVER_URL_PREDICT = "http://10.50.15.68:5000/predict"
SERVER_URL_FINALIZE = "http://10.50.15.68:5000/session/finalize"
SERVER_URL_CONFIRM = "http://10.50.15.68:5000/session/confirm_segment"
SERVER_URL_REFRESH = "http://10.50.15.68:5000/refresh-cache"

# --- 测试模式配置 ---
TEST_MODE = 'full_plate' # 可选: 'full_plate', 'scattered'

# --- 测试图片路径配置 ---
IMAGE_PATHS_TO_UPLOAD = [
    r"../../../../DATA/PIC/1pic/1/1.jpg",
    r"../../../../DATA/PIC/1pic/1/2.jpg",
    r"../../../../DATA/PIC/1pic/1/3.jpg",
    r"../../../../DATA/PIC/1pic/1/4.jpg"
]

# --- 【核心修改】新增一个辅助函数，用于打印可读的JSON日志 ---
# --- 辅助函数 ---
def print_sanitized_log(response_json: dict, title: str):
    """打印关键信息，避免base64刷屏。"""
    print(f"--- {title} ---")
    if not isinstance(response_json, dict):
        print(f"  响应 (非JSON): {response_json}")
        return

    status = response_json.get('session_status')
    print(f"  状态: {status}")

    if status == 'awaiting_confirmation':
        candidates = response_json.get('candidate_segments', [])
        print(f"  候选号段: {candidates}")
    else:
        confirmed_len = len(response_json.get('confirmed_results', []))
        pending_len = len(response_json.get('pending_results', []))
        print(f"  结果: {confirmed_len} 个确信, {pending_len} 个待定")

    base64_key = 'current_frame_annotated_image_base64'
    if base64_key in response_json and response_json[base64_key]:
        print("  标注图: [已接收]")
    else:
        print("  标注图: [未接收]")
    print("--------------------")

def display_base64_image(base64_string: str, window_name: str = "Annotated Frame"):
    if not base64_string:
        print("  客户端显示：无Base64图像数据可显示。")
        return
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow(window_name, img)
            cv2.waitKey(100)
        else:
            print(f"  客户端显示：解码Base64图像数据失败。")
    except Exception as e:
        print(f"  客户端显示：显示Base64图像时发生错误: {e}")

# --- API请求函数 ---
def send_image_for_prediction(image_path: str, session_id_to_use: str, mode: str = None):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None
    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {'session_id': session_id_to_use}
            if mode:
                data_payload['recognition_mode'] = mode
            response = requests.post(SERVER_URL_PREDICT, files=files_payload, data=data_payload, timeout=180)
            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', 服务端状态码: {response.status_code}")
            if response.status_code == 200 or response.status_code == 409:
                return response.json()
            else:
                try: print(f"  服务端错误详情: {response.json()}")
                except requests.exceptions.JSONDecodeError: print(f"  服务端原始响应 (非JSON): {response.text}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"客户端：会话 {session_id_to_use}, 请求时发生网络错误: {e}")
        return None

def confirm_chosen_segment(session_id_to_use: str, chosen_key: str):
    print("\n" + "="*40)
    print(f"客户端：正在为会话 {session_id_to_use} 确认号段: {chosen_key}")
    print("="*40)
    try:
        headers = {'Content-Type': 'application/json'}
        data_payload = json.dumps({'session_id': session_id_to_use, 'chosen_segment_key': chosen_key})
        response = requests.post(SERVER_URL_CONFIRM, data=data_payload, headers=headers, timeout=60)
        print(f"客户端：确认接口状态码: {response.status_code}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  服务端错误详情: {response.json()}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"客户端：请求确认接口时发生网络错误: {e}")
        return None

def finalize_session(session_id_to_finalize: str):
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
        print(f"客户端：请求终审接口时发生网络错误: {e}")
        return None

if __name__ == "__main__":
    current_batch_session_id = str(uuid.uuid4())
    print(f"客户端：开始新的扫描会话，ID: {current_batch_session_id}, 模式: {TEST_MODE}")

    for i, img_path in enumerate(IMAGE_PATHS_TO_UPLOAD):
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 未找到，跳过。")
            continue

        print(f"\n客户端：准备发送图片 {i+1}/{len(IMAGE_PATHS_TO_UPLOAD)}: '{os.path.basename(img_path)}'")

        mode_to_send = TEST_MODE if i == 0 else None
        response_json = send_image_for_prediction(img_path, current_batch_session_id, mode_to_send)

        if not response_json:
            print("客户端：请求失败，终止会话。")
            break

        print_sanitized_log(response_json, f"帧 {i+1} /predict 响应")

        if response_json.get('session_status') == 'awaiting_confirmation':
            print("客户端：服务端要求人工仲裁！")

            candidates = response_json.get('candidate_segments', [])
            if not candidates:
                print("错误：服务端要求仲裁但未提供候选号段。终止测试。")
                break

            print("\n  --- 请操作员确认号段 ---")
            for idx, key in enumerate(candidates):
                print(f"    选项 {idx+1}: {key} {'(首要推荐)' if idx == 0 else ''}")

            chosen_key = None
            while not chosen_key:
                choice = input(f"请输入选项编号 (1-{len(candidates)})，或直接按回车选择首要推荐: ")
                if choice == "":
                    chosen_key = candidates[0]
                    break
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(candidates):
                        chosen_key = candidates[choice_idx]
                        break
                    else:
                        print("无效的选项编号，请重新输入。")
                except ValueError:
                    print("无效输入，请输入数字。")

            print(f"  (用户已选择) 确认号段为: {chosen_key}")

            confirm_response = confirm_chosen_segment(current_batch_session_id, chosen_key)
            if confirm_response:
                print("客户端：号段锁定成功。")
                response_json = confirm_response
                print_sanitized_log(response_json, "确认后首帧结果")
            else:
                print("客户端：号段确认失败，终止测试。")
                break

        display_base64_image(response_json.get('current_frame_annotated_image_base64'), f"Frame {i+1}")

        if i < len(IMAGE_PATHS_TO_UPLOAD) - 1:
            print("-" * 30)
            time.sleep(0.5)

    finalize_session(current_batch_session_id)
    cv2.destroyAllWindows()
    print("\n客户端测试完成。")