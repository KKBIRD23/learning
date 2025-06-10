# many_client_StateMatrix.py
import requests
import os
import uuid
import time
import cv2
import numpy as np
from datetime import datetime
import base64 # 用于解码Base64图像数据

# --- 配置 ---
SERVER_URL = "http://127.0.0.1:5000/predict"
IMAGE_PATHS_TO_UPLOAD = [
    r"../../../../DATA/PIC/1pic/2-1.jpg",
    r"../../../../DATA/PIC/1pic/2-2.jpg",
    r"../../../../DATA/PIC/1pic/2-3.jpg",
    r"../../../../DATA/PIC/1pic/2-4.jpg",
    # 用于测试第一帧特殊行失败的情况 (如果服务端仍保留整版模式的行政干预)
    # r"../../../../DATA/PIC/2pic/2-1.jpg",
]
# 客户端默认请求的模式 (可以设为 'scattered_cumulative_ocr' 或 'full_layout')
DEFAULT_SESSION_PROCESSING_MODE = 'scattered_cumulative_ocr'


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
            cv2.imshow(window_name, img)
            cv2.waitKey(100) # 短暂显示，或用0等待按键
            # cv2.destroyWindow(window_name) # 如果希望每次都关闭旧窗口
        else:
            print(f"  客户端显示：解码Base64图像数据失败 (cv2.imdecode返回None)。")
    except Exception as e:
        print(f"  客户端显示：显示Base64图像时发生错误: {e}")

# --- 矩阵渲染函数 (为整版模式保留，但当前默认不调用) ---
def render_matrix_from_json(status_matrix, texts_map, session_id, image_name_base, frame_num):
    # ... (这个函数的代码与您之前提供的版本完全相同，此处为了简洁暂时省略)
    # ... (如果需要，我可以再次提供完整的这个函数)
    # ... (它的核心是绘制13x4的矩阵图并保存)
    # --- 为了确保完整性，我将粘贴过来 ---
    if not status_matrix: print("  客户端渲染：状态矩阵为空，无法渲染。"); return
    num_rows = len(status_matrix); num_cols = len(status_matrix[0]) if num_rows > 0 else 0
    if num_rows == 0 or num_cols == 0: print("  客户端渲染：状态矩阵维度为0，无法渲染。"); return
    cell_size = 80; padding = 25; spacing = 7; text_offset_y = -9
    img_width = num_cols * cell_size + (num_cols - 1) * spacing + 2 * padding
    img_height = num_rows * cell_size + (num_rows - 1) * spacing + 2 * padding
    canvas = np.full((img_height, img_width, 3), (230, 230, 230), dtype=np.uint8)
    color_success_fill = (100, 200, 100); color_fail_fill = (100, 100, 220); color_unknown_fill = (200, 200, 200); color_unavailable_fill = (250, 250, 250)
    color_text_on_dark_bg = (255, 255, 255); color_text_on_light_bg = (30, 30, 30)
    font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.5; font_thickness = 1
    for r in range(num_rows):
        for c in range(num_cols):
            cell_x_start = padding + c * (cell_size + spacing); cell_y_start = padding + r * (cell_size + spacing)
            center_x = cell_x_start + cell_size // 2; center_y = cell_y_start + cell_size // 2
            status = status_matrix[r][c]; obu_text = texts_map.get(f"{r}_{c}", "")
            current_fill_color = color_unknown_fill; display_text = "?"; current_text_color = color_text_on_light_bg
            if status == 1: current_fill_color = color_success_fill; display_text = obu_text if len(obu_text) <= 8 else obu_text[-6:]; current_text_color = color_text_on_dark_bg
            elif status == 2: current_fill_color = color_fail_fill; display_text = "X"; current_text_color = color_text_on_dark_bg
            elif status == -1: current_fill_color = color_unavailable_fill; display_text = "";
            elif status == 0: current_fill_color = color_unknown_fill; display_text = "?"; current_text_color = color_text_on_light_bg
            cv2.rectangle(canvas, (cell_x_start, cell_y_start), (cell_x_start + cell_size, cell_y_start + cell_size), current_fill_color, -1)
            cv2.rectangle(canvas, (cell_x_start, cell_y_start), (cell_x_start + cell_size, cell_y_start + cell_size), (100,100,100), 1)
            if display_text:
                (text_w, text_h), _ = cv2.getTextSize(display_text, font_face, font_scale, font_thickness)
                text_x = center_x - text_w // 2; text_y = center_y + text_h // 2
                cv2.putText(canvas, display_text, (text_x, text_y), font_face, font_scale, current_text_color, font_thickness, cv2.LINE_AA)
    output_render_dir = "process_photo"
    if not os.path.exists(output_render_dir):
        try: os.makedirs(output_render_dir)
        except OSError as e: print(f"  客户端渲染：创建目录 {os.path.abspath(output_render_dir)} 失败: {e}"); return
    rendered_image_name = f"{output_render_dir}/matrix_s{session_id[:8]}_p{image_name_base}_f{frame_num}.png"
    absolute_image_path = os.path.abspath(rendered_image_name)
    try:
        save_success = cv2.imwrite(rendered_image_name, canvas)
        if save_success: print(f"  客户端渲染：逻辑矩阵图已成功保存到: {absolute_image_path}")
        else: print(f"  客户端渲染：cv2.imwrite 调用返回 False，保存到 {absolute_image_path} 失败。")
    except Exception as e_imwrite: print(f"  客户端渲染：保存逻辑矩阵图到 {absolute_image_path} 时发生异常: {e_imwrite}")


# --- 核心请求函数 (适配新参数和响应) ---
def send_image_for_prediction(image_path: str, session_id_to_use: str, frame_counter: int,
                              mode_to_use: str, # 现在 mode_to_use 是必须的
                              force_recalibrate_payload: bool =False):
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None
    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {'session_id': session_id_to_use, 'mode': mode_to_use}
            if mode_to_use == 'full_layout' and force_recalibrate_payload:
                data_payload['force_recalibrate'] = 'true'

            response = requests.post(SERVER_URL, files=files_payload, data=data_payload, timeout=180)
            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', "
                  f"请求模式 '{mode_to_use}', 服务端状态码: {response.status_code}")

            if response.status_code == 200:
                response_json = response.json()
                print(f"  服务端消息: {response_json.get('message')}")
                processed_mode = response_json.get('mode_processed', 'unknown')
                print(f"  服务端实际处理模式: {processed_mode}")
                session_status = response_json.get('session_status')
                print(f"  会话状态: {session_status}")

                warnings = response_json.get('warnings')
                if warnings:
                    print(f"  服务端警告:")
                    for warn_item in warnings:
                        print(f"    - {warn_item.get('message')} (code: {warn_item.get('code')})")

                if processed_mode == 'full_layout':
                    status_matrix = response_json.get('obu_status_matrix')
                    texts_map = response_json.get('obu_texts')
                    if status_matrix and texts_map is not None:
                        image_name_base = os.path.splitext(os.path.basename(image_path))[0]
                        render_matrix_from_json(status_matrix, texts_map, session_id_to_use, image_name_base, frame_counter)
                    else: print("  客户端渲染（整版模式）：从服务端接收到的矩阵数据不完整或错误。")

                elif processed_mode == 'scattered_cumulative_ocr': # 适配新的模式名
                    accumulated_results = response_json.get('accumulated_results') # 获取累积结果
                    current_frame_base64_img = response_json.get('current_frame_annotated_image_base64')

                    if accumulated_results is not None:
                        print(f"  累积识别OBU列表 (共 {len(accumulated_results)} 个):")
                        if accumulated_results:
                            for idx, item in enumerate(accumulated_results):
                                # 假设 accumulated_results 中的每个 item 是一个包含 "text" 的字典
                                print(f"    {idx+1}. {item.get('text')}")
                        else:
                            print("    （累积列表为空）")
                    else:
                        print("  客户端（累积零散模式）：未收到 accumulated_results 字段。")

                    if current_frame_base64_img:
                        print("  客户端显示：当前帧标注图...")
                        display_base64_image(current_frame_base64_img, f"Frame {frame_counter} - Annotated")
                    else:
                        print("  客户端显示：未收到当前帧标注图数据。")

                return response_json
            else:
                try: print(f"  服务端错误详情: {response.json()}")
                except requests.exceptions.JSONDecodeError: print(f"  服务端原始响应 (非JSON): {response.text}")
                return None
    except requests.exceptions.Timeout:
        print(f"客户端：会话 {session_id_to_use}, 请求图片 '{os.path.basename(image_path)}' 时发生超时错误。")
        return None
    except requests.exceptions.RequestException as e:
        print(f"客户端：会话 {session_id_to_use}, 请求图片 '{os.path.basename(image_path)}' 时发生网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"客户端：会话 {session_id_to_use}, 发送图片 '{os.path.basename(image_path)}' 时发生未知客户端错误: {e}")
        return None

if __name__ == "__main__":
    current_batch_session_id = str(uuid.uuid4())
    print(f"客户端：开始新的扫描会话，ID: {current_batch_session_id}")

    # --- 会话开始时，确定处理模式 (当前默认且仅支持累积式零散识别) ---
    # 我们将 mode 硬编码为新的默认模式，移除了用户选择
    session_processing_mode = DEFAULT_SESSION_PROCESSING_MODE
    print(f"客户端：本次会话将使用“{session_processing_mode}”模式。")
    # --- 结束模式确定 ---

    # --- 整版模式下的强制校准逻辑 (暂时保留，但当前默认模式下不会触发) ---
    force_recalibrate_for_first_frame_if_full_layout = False
    if session_processing_mode == 'full_layout': # 仅当模式是整版时才询问
        recalibrate_input = input("是否对本次会话的第一帧强制重新校准布局? (输入 'y' 或 'Y' 后回车，否则不校准): ").strip().lower()
        force_recalibrate_for_first_frame_if_full_layout = (recalibrate_input == 'y')
        if force_recalibrate_for_first_frame_if_full_layout:
            print(f"  客户端：本次会话的第一帧（如果是整版模式）将发送强制重新校准请求。")
    # --- 结束校准询问 ---

    frame_count = 0
    for img_path in IMAGE_PATHS_TO_UPLOAD:
        frame_count += 1
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 未找到，跳过。")
            continue

        image_basename = os.path.basename(img_path)
        print(f"\n客户端：准备发送图片 {frame_count}/{len(IMAGE_PATHS_TO_UPLOAD)}: '{image_basename}' (模式: {session_processing_mode})")

        should_force_recalibrate_this_call = False
        if session_processing_mode == 'full_layout' and frame_count == 1 and force_recalibrate_for_first_frame_if_full_layout:
            should_force_recalibrate_this_call = True
            print(f"  客户端：为第一帧 '{image_basename}' 发送强制重新校准请求。")

        json_response = send_image_for_prediction(
            img_path,
            current_batch_session_id,
            frame_count,
            mode_to_use=session_processing_mode,
            force_recalibrate_payload=should_force_recalibrate_this_call
        )

        if json_response:
            session_status = json_response.get("session_status")
            if session_status == "completed": # 整版模式完成
                print(f"客户端：会话 {current_batch_session_id} (整版识别) 已完成！")
            elif session_status == "scattered_recognition_in_progress" or session_status == "scattered_ocr_completed": # 零散模式
                print(f"客户端：图片 '{image_basename}' (累积零散识别) 处理完成。")
            elif session_status == "first_frame_anchor_error": # 整版模式首帧错误
                 print(f"客户端：【重要】图片 '{image_basename}' (整版识别-首帧) 未能满足拍摄规定，处理中断！请检查服务端警告。")
                 print(f"客户端：由于首帧锚定失败，会话 {current_batch_session_id} 可能无法继续，建议重新开始新会话并按规定拍摄首帧。")
                 # break # 可以选择在这里中断后续图片的发送

        if frame_count < len(IMAGE_PATHS_TO_UPLOAD):
            # time.sleep(0.1) # 非常短的延时，避免打印过于密集
            print("-" * 30)
            # 可以在这里加入 input("按回车发送下一张...") 来控制节奏
            # input("按回车发送下一张...")


    print(f"\n客户端：会话 {current_batch_session_id} 的所有指定图片已发送。")
    # 在所有图片发送完毕后，可以尝试关闭OpenCV的显示窗口（如果之前没有逐个关闭）
    cv2.destroyAllWindows()
    print("客户端测试完成。")