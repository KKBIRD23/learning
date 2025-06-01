# client.py
import requests
import os
import uuid
import time # For time.sleep

# --- 配置 ---
SERVER_URL = "http://127.0.0.1:5000/predict"

# 图片路径列表 - 请确保这些图片存在于相对于 client.py 的正确路径
# 例如，如果 client.py 在 Service 目录, DATA 在 Service 的上两级目录的 ObuPrintNo_OCR 下
IMAGE_PATHS_TO_UPLOAD = [
    r"../../../DATA/PIC/1pic/1-1.jpg", # 假设您有这些图片用于测试会话
    r"../../../DATA/PIC/1pic/1-2.jpg",
    r"../../../DATA/PIC/1pic/1-3.jpg",
    r"../../../DATA/PIC/1pic/1-4.jpg",
    # r"../../../DATA/PIC/2.jpg", # 可以用2.jpg和3.jpg测试不同会话
    # r"../../../DATA/PIC/3.jpg",
]
# 或者只用一张图片多次发送来测试增量填充（如果图片内容本身支持）
# IMAGE_PATHS_TO_UPLOAD = [r"../../DATA/PIC/2.jpg"] * 3


def send_image_for_prediction(image_path, session_id_to_use, box_type_to_send=None):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None # Return None on failure

    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {'session_id': session_id_to_use}

            if box_type_to_send: # (未来扩展)
                data_payload['box_type'] = box_type_to_send

            print(f"客户端：会话 {session_id_to_use}, 发送图片 '{os.path.basename(image_path)}' 到 {SERVER_URL} ...")
            response = requests.post(SERVER_URL, files=files_payload, data=data_payload, timeout=60)

            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', 服务端状态码: {response.status_code}")
            if response.status_code == 200:
                response_json = response.json()
                print(f"  服务端消息: {response_json.get('message')}")
                print(f"  会话状态: {response_json.get('session_status')}")
                # print(f"  OBU矩阵: {response_json.get('obu_matrix')}") # 可以取消注释以查看完整矩阵
                # 打印矩阵的简略形式
                matrix = response_json.get('obu_matrix')
                if matrix:
                    print("  矩阵概览:")
                    for row in matrix:
                        print(f"    {[str(item)[:10] + '...' if len(str(item)) > 10 else str(item) for item in row]}")
                return response_json # Return JSON response for further checks
            else:
                try:
                    print(f"  服务端错误: {response.json()}")
                except requests.exceptions.JSONDecodeError:
                    print(f"  服务端原始响应: {response.text}")
                return None

    except requests.exceptions.RequestException as e:
        print(f"客户端：会话 {session_id_to_use}, 请求图片 '{os.path.basename(image_path)}' 时发生错误: {e}")
        return None
    except Exception as e:
        print(f"客户端：会话 {session_id_to_use}, 发送图片 '{os.path.basename(image_path)}' 时发生未知错误: {e}")
        return None

if __name__ == "__main__":
    # --- 测试场景1: 对同一批次OBU连续发送多张不同角度的图片 ---
    print("\n--- 测试场景1: 同一会话，多张图片 ---")
    session1_id = str(uuid.uuid4())
    print(f"客户端：开始新的扫描会话 1，ID: {session1_id}")

    # 使用您准备的针对同一版OBU的多张照片
    # 假设这些图片在 IMAGE_PATHS_TO_UPLOAD 的前几张
    session1_image_paths = IMAGE_PATHS_TO_UPLOAD[:4] # 例如用前4张图模拟对同一批次的拍摄
    if not session1_image_paths:
        print("警告: 测试场景1没有足够的图片路径，请在 IMAGE_PATHS_TO_UPLOAD 中配置。")

    for img_path in session1_image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 测试场景1的图片 {img_path} 未找到，跳过。")
            continue
        json_response = send_image_for_prediction(img_path, session1_id)
        if json_response and json_response.get("session_status") == "completed":
            print(f"客户端：会话 {session1_id} 已完成！")
            # break # 可以选择在会话完成后停止发送该会话的图片
        time.sleep(2) # 模拟用户操作间隔
    print(f"客户端：会话 {session1_id} 的所有指定图片已发送。")

    # --- 测试场景2: 开始一个新的会话处理另一批OBU (例如用2.jpg) ---
    if len(IMAGE_PATHS_TO_UPLOAD) > 4: #确保有不同的图片用于新会话
        print("\n\n--- 测试场景2: 新的会话 ---")
        session2_id = str(uuid.uuid4())
        print(f"客户端：开始新的扫描会话 2，ID: {session2_id}")
        image_for_session2 = IMAGE_PATHS_TO_UPLOAD[4] # 例如用第5张图
        if not os.path.exists(image_for_session2):
            print(f"警告: 测试场景2的图片 {image_for_session2} 未找到，跳过。")
        else:
            send_image_for_prediction(image_for_session2, session2_id)
        print(f"客户端：会话 {session2_id} 的图片已发送。")

    print("\n客户端测试完成。")