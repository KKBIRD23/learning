# client.py
import requests
import os
import uuid
import time # For time.sleep
import cv2 # <--- 新增 OpenCV import
import numpy as np # <--- 新增 NumPy import
from datetime import datetime # <--- 新增 datetime import (如果之前没有)

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

# --- 新增：矩阵渲染函数 ---
def render_matrix_from_json(status_matrix, texts_map, session_id, image_name_base, frame_num):
    if not status_matrix:
        print("  客户端渲染：状态矩阵为空，无法渲染。")
        return

    # ... (num_rows, num_cols, canvas创建等逻辑不变) ...
    num_rows = len(status_matrix)
    num_cols = len(status_matrix[0]) if num_rows > 0 else 0
    if num_rows == 0 or num_cols == 0:
        print("  客户端渲染：状态矩阵维度为0，无法渲染。")
        return

    cell_size = 70
    padding = 20
    spacing = 5
    text_offset_y = -7
    img_width = num_cols * cell_size + (num_cols - 1) * spacing + 2 * padding
    img_height = num_rows * cell_size + (num_rows - 1) * spacing + 2 * padding
    canvas = np.full((img_height, img_width, 3), (230, 230, 230), dtype=np.uint8)

    # ... (颜色和字体定义不变) ...
    color_success_fill = (0, 180, 0); color_fail_fill = (0, 0, 200); color_unknown_fill = (200, 200, 200); color_unavailable_fill = (240, 240, 240)
    color_text_on_dark_bg = (255, 255, 255); color_text_on_light_bg = (50, 50, 50)
    font_scale = 0.6; font_thickness = 1

    # ... (绘制循环不变) ...
    for r in range(num_rows):
        for c in range(num_cols):
            # ... (cell_x_start, cell_y_start, center_x, center_y 计算) ...
            cell_x_start = padding + c * (cell_size + spacing); cell_y_start = padding + r * (cell_size + spacing)
            center_x = cell_x_start + cell_size // 2; center_y = cell_y_start + cell_size // 2
            status = status_matrix[r][c]; obu_text = texts_map.get(f"{r}_{c}", "")
            current_fill_color = color_unknown_fill; display_text = "?"; current_text_color = color_text_on_light_bg
            if status == 1: current_fill_color = color_success_fill; display_text = obu_text[-4:] if obu_text and len(obu_text) >= 4 else "ERR_S"; current_text_color = color_text_on_dark_bg
            elif status == 2: current_fill_color = color_fail_fill; display_text = "X"; current_text_color = color_text_on_dark_bg
            elif status == -1: current_fill_color = color_unavailable_fill; display_text = ""; current_text_color = color_text_on_light_bg
            elif status == 0: current_fill_color = color_unknown_fill; display_text = "?"; current_text_color = color_text_on_light_bg
            cv2.rectangle(canvas, (cell_x_start, cell_y_start), (cell_x_start + cell_size, cell_y_start + cell_size), current_fill_color, -1)
            cv2.rectangle(canvas, (cell_x_start, cell_y_start), (cell_x_start + cell_size, cell_y_start + cell_size), (100,100,100), 1)
            if display_text:
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = center_x - text_w // 2; text_y = center_y + text_h // 2 + text_offset_y
                cv2.putText(canvas, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, current_text_color, font_thickness, cv2.LINE_AA)

    output_render_dir = "process_photo" # 客户端渲染目录
    # --- 新增调试打印 ---
    current_working_directory = os.getcwd()
    print(f"  客户端渲染：当前工作目录: {current_working_directory}")
    absolute_output_dir = os.path.abspath(output_render_dir)
    print(f"  客户端渲染：将尝试创建/使用目录: {absolute_output_dir}")
    # --- 结束新增 ---

    if not os.path.exists(output_render_dir): # 确保使用相对路径来创建
        try:
            os.makedirs(output_render_dir)
            print(f"  客户端渲染：成功创建目录: {absolute_output_dir}")
        except OSError as e:
            print(f"  客户端渲染：创建目录 {absolute_output_dir} 失败: {e}")
            return # 如果目录创建失败，则不继续保存

    rendered_image_name = f"{output_render_dir}/matrix_{session_id[:8]}_{image_name_base}_f{frame_num}.png"
    absolute_image_path = os.path.abspath(rendered_image_name) # 获取绝对路径用于打印

    print(f"  客户端渲染：准备保存逻辑矩阵图到: {absolute_image_path}") # --- 新增 ---
    try:
        save_success = cv2.imwrite(rendered_image_name, canvas) # imwrite返回True/False
        if save_success:
            print(f"  客户端渲染：逻辑矩阵图已成功保存到: {absolute_image_path}")
        else:
            print(f"  客户端渲染：cv2.imwrite 调用返回 False，保存到 {absolute_image_path} 失败 (请检查路径、权限和图像数据)。")
            # 可以在这里尝试打印 canvas 的 shape 和 dtype，看是否有问题
            print(f"  客户端渲染：Canvas shape: {canvas.shape}, dtype: {canvas.dtype}")

    except Exception as e_imwrite:
        print(f"  客户端渲染：保存逻辑矩阵图到 {absolute_image_path} 时发生异常: {e_imwrite}")

    # ... (cv2.imshow 和 cv2.waitKey 可以保持注释) ...

def send_image_for_prediction(image_path, session_id_to_use, frame_counter, box_type_to_send=None):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None # Return None on failure

    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {'session_id': session_id_to_use}
            # ...
            response = requests.post(SERVER_URL, files=files_payload, data=data_payload, timeout=60)

            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', 服务端状态码: {response.status_code}")
            if response.status_code == 200:
                response_json = response.json()
                print(f"  服务端消息: {response_json.get('message')}")
                print(f"  会话状态: {response_json.get('session_status')}")

                status_matrix = response_json.get('obu_status_matrix')
                texts_map = response_json.get('obu_texts')

                if status_matrix and texts_map is not None: # texts_map可以是空字典
                    image_name_base = os.path.splitext(os.path.basename(image_path))[0]
                    render_matrix_from_json(status_matrix, texts_map, session_id_to_use, image_name_base, frame_counter)
                else:
                    print("  客户端渲染：从服务端接收到的矩阵数据不完整。")
                return response_json
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
    current_batch_session_id = str(uuid.uuid4())
    print(f"客户端：开始新的扫描会话，ID: {current_batch_session_id}")

    frame_count = 0 # 初始化帧计数器
    for img_path in IMAGE_PATHS_TO_UPLOAD:
        frame_count += 1 # 递增帧计数器
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 未找到，跳过。")
            continue
        json_response = send_image_for_prediction(img_path, current_batch_session_id, frame_count) # 传递帧计数器
        if json_response and json_response.get("session_status") == "completed":
            print(f"客户端：会话 {current_batch_session_id} 已完成！")
            # break
        time.sleep(1)

    print(f"客户端：会话 {current_batch_session_id} 的所有指定图片已发送。")

    print("\n客户端测试完成。")