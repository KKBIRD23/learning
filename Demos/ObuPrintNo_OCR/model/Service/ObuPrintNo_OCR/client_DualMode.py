# many_client_StateMatrix.py
import requests
import os
import uuid
import time
import cv2
import numpy as np
from datetime import datetime

# --- 配置 ---
SERVER_URL = "http://127.0.0.1:5000/predict"

# 图片路径列表 - 请确保这些图片存在于相对于 client.py 的正确路径
IMAGE_PATHS_TO_UPLOAD = [
    r"../../../../DATA/PIC/1pic/2-1.jpg",
    r"../../../../DATA/PIC/1pic/2-2.jpg",
    r"../../../../DATA/PIC/1pic/2-3.jpg",
    r"../../../../DATA/PIC/1pic/2-4.jpg",
    # 用于测试第一帧特殊行识别失败 (假设这张图片不包含清晰的底部特殊行)
    # r"../../../../DATA/PIC/2pic/2-1.jpg",
]


# --- 矩阵渲染函数 ---
def render_matrix_from_json(status_matrix, texts_map, session_id, image_name_base, frame_num):
    """
    根据服务端返回的状态矩阵和文本映射，生成并保存一个可视化的矩阵图像。
    """
    if not status_matrix:
        print("  客户端渲染：状态矩阵为空，无法渲染。")
        return

    num_rows = len(status_matrix)
    num_cols = len(status_matrix[0]) if num_rows > 0 else 0
    if num_rows == 0 or num_cols == 0:
        print("  客户端渲染：状态矩阵维度为0，无法渲染。")
        return

    # --- 绘图参数 ---
    cell_size = 80  # 稍微增大单元格尺寸
    padding = 25    # 增加内边距
    spacing = 7     # 增加间距
    text_offset_y = -9 # 文本Y轴微调

    img_width = num_cols * cell_size + (num_cols - 1) * spacing + 2 * padding
    img_height = num_rows * cell_size + (num_rows - 1) * spacing + 2 * padding

    # 创建一个浅灰色背景的画布
    canvas = np.full((img_height, img_width, 3), (230, 230, 230), dtype=np.uint8)

    # 定义颜色
    color_success_fill = (100, 200, 100) # 柔和的绿色
    color_fail_fill = (100, 100, 220)    # 柔和的红色
    color_unknown_fill = (200, 200, 200) # 灰色
    color_unavailable_fill = (250, 250, 250) # 更浅的灰色，接近白色

    color_text_on_dark_bg = (255, 255, 255) # 白色文本
    color_text_on_light_bg = (30, 30, 30)   # 深灰色文本

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 # 稍微减小字体以适应更多文本
    font_thickness = 1

    # --- 绘制单元格和文本 ---
    for r in range(num_rows):
        for c in range(num_cols):
            cell_x_start = padding + c * (cell_size + spacing)
            cell_y_start = padding + r * (cell_size + spacing)
            center_x = cell_x_start + cell_size // 2
            center_y = cell_y_start + cell_size // 2

            status = status_matrix[r][c]
            obu_text = texts_map.get(f"{r}_{c}", "") # 从字典获取文本

            current_fill_color = color_unknown_fill
            display_text = "?"
            current_text_color = color_text_on_light_bg

            if status == 1: # 成功识别
                current_fill_color = color_success_fill
                # 显示完整的OBU码，如果太长则只显示部分
                display_text = obu_text if len(obu_text) <= 8 else obu_text[-6:] # 例如最多显示后6位
                current_text_color = color_text_on_dark_bg
            elif status == 2: # OCR失败
                current_fill_color = color_fail_fill
                display_text = "X"
                current_text_color = color_text_on_dark_bg
            elif status == -1: # 不可用
                current_fill_color = color_unavailable_fill
                display_text = "" # 不可用单元格不显示文本
            elif status == 0: # 未知
                current_fill_color = color_unknown_fill
                display_text = "?"
                current_text_color = color_text_on_light_bg

            # 绘制单元格背景
            cv2.rectangle(canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          current_fill_color, -1)
            # 绘制单元格边框
            cv2.rectangle(canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          (100, 100, 100), 1) # 深灰色边框

            # 绘制文本
            if display_text:
                (text_w, text_h), baseline = cv2.getTextSize(display_text, font_face, font_scale, font_thickness)
                # 调整文本位置使其更居中
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2 # OpenCV putText的y是基线位置
                cv2.putText(canvas, display_text, (text_x, text_y),
                            font_face, font_scale, current_text_color, font_thickness, cv2.LINE_AA)

    # --- 保存图像 ---
    output_render_dir = "process_photo"
    if not os.path.exists(output_render_dir):
        try:
            os.makedirs(output_render_dir)
            print(f"  客户端渲染：成功创建目录: {os.path.abspath(output_render_dir)}")
        except OSError as e:
            print(f"  客户端渲染：创建目录 {os.path.abspath(output_render_dir)} 失败: {e}")
            return

    # 使用更明确的文件名
    rendered_image_name = f"{output_render_dir}/matrix_s{session_id[:8]}_p{image_name_base}_f{frame_num}.png"
    absolute_image_path = os.path.abspath(rendered_image_name)

    try:
        save_success = cv2.imwrite(rendered_image_name, canvas)
        if save_success:
            print(f"  客户端渲染：逻辑矩阵图已成功保存到: {absolute_image_path}")
        else:
            print(f"  客户端渲染：cv2.imwrite 调用返回 False，保存到 {absolute_image_path} 失败。")
            print(f"  客户端渲染：Canvas shape: {canvas.shape}, dtype: {canvas.dtype}")
    except Exception as e_imwrite:
        print(f"  客户端渲染：保存逻辑矩阵图到 {absolute_image_path} 时发生异常: {e_imwrite}")

# --- 核心请求函数 (适配新参数和响应) ---
def send_image_for_prediction(image_path: str, session_id_to_use: str, frame_counter: int,
                              mode_to_use: str ='full_layout',
                              force_recalibrate_payload: bool =False):
    """
    发送图片到服务端进行预测。
    Args:
        image_path (str): 要发送的图片的路径。
        session_id_to_use (str): 当前会话的ID。
        frame_counter (int): 当前是第几帧（用于日志和文件名）。
        mode_to_use (str): 'full_layout' 或 'scattered_ocr'。
        force_recalibrate_payload (bool): 是否强制服务端重新校准（仅对full_layout模式的第一帧有效）。
    Returns:
        dict or None: 服务端返回的JSON响应，如果请求失败则返回None。
    """
    if not os.path.exists(image_path):
        print(f"客户端错误：图片文件未找到 - {image_path}")
        return None

    try:
        with open(image_path, 'rb') as f:
            files_payload = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data_payload = {
                'session_id': session_id_to_use,
                'mode': mode_to_use
            }
            if mode_to_use == 'full_layout' and force_recalibrate_payload:
                data_payload['force_recalibrate'] = 'true'

            # 增加超时时间，以防服务端处理时间过长
            response = requests.post(SERVER_URL, files=files_payload, data=data_payload, timeout=180) # 例如3分钟超时

            print(f"客户端：会话 {session_id_to_use}, 图片 '{os.path.basename(image_path)}', "
                  f"模式 '{mode_to_use}', 服务端状态码: {response.status_code}")

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

                # 根据处理模式解析和显示结果
                if processed_mode == 'full_layout':
                    status_matrix = response_json.get('obu_status_matrix')
                    texts_map = response_json.get('obu_texts') # 服务端返回的是 "r_c": "text"

                    # 将服务端返回的 "r_c" 格式的 texts_map 转换为客户端需要的 (r,c) -> text 格式
                    # (或者，我们修改客户端渲染函数以适应 "r_c" 格式，但服务端已按此格式返回，客户端应适配)
                    # 服务端返回的已经是 "r_c": "text" 格式，render_matrix_from_json 也应适配
                    # (我之前提供的 render_matrix_from_json 用的就是 f"{r}_{c}" 作为键，所以这里应该没问题)

                    if status_matrix and texts_map is not None: # texts_map可以是空字典
                        image_name_base = os.path.splitext(os.path.basename(image_path))[0]
                        render_matrix_from_json(status_matrix, texts_map, session_id_to_use, image_name_base, frame_counter)
                    else:
                        print("  客户端渲染（整版模式）：从服务端接收到的矩阵数据不完整或错误。")

                elif processed_mode == 'scattered_ocr':
                    scattered_results = response_json.get('scattered_results')
                    if scattered_results is not None:
                        print(f"  零散识别结果 (共 {len(scattered_results)} 个):")
                        if scattered_results:
                            for idx, item in enumerate(scattered_results):
                                ocr_conf_str = f"{item.get('ocr_confidence', 'N/A'):.2f}" if isinstance(item.get('ocr_confidence'), float) else 'N/A'
                                yolo_score_str = f"{item.get('yolo_score', 'N/A'):.2f}" if isinstance(item.get('yolo_score'), float) else 'N/A'
                                print(f"    {idx+1}. 文本: {item.get('text')}, "
                                      f"位置(xyxy): {item.get('bbox_xyxy')}, "
                                      f"OCR置信度: {ocr_conf_str}, "
                                      f"YOLO得分: {yolo_score_str}")
                        else:
                            print("    （当前图片未识别到有效OBU码）")
                    else:
                        print("  客户端（零散模式）：未收到 scattered_results 字段或格式错误。")

                return response_json
            else: # 非200状态码
                try:
                    print(f"  服务端错误详情: {response.json()}")
                except requests.exceptions.JSONDecodeError:
                    print(f"  服务端原始响应 (非JSON): {response.text}")
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

    # --- 在会话开始时，一次性确定处理模式 ---
    session_mode_prompt = (
        "\n请为本次会话选择处理模式 (直接回车默认为 '整版识别'):\n"
        "  1. 整版识别 (full_layout) - 需要精确布局和多帧累积\n"
        "  2. 零散识别 (scattered_ocr) - 快速识别图中所有OBU，不关心布局\n"
        "请输入选项 (1 或 2): "
    )
    session_mode_choice = input(session_mode_prompt).strip()
    session_processing_mode = 'full_layout'
    if session_mode_choice == '2':
        session_processing_mode = 'scattered_ocr'
    print(f"客户端：本次会话将使用“{session_processing_mode}”模式。")
    # --- 结束模式选择 ---

    force_recalibrate_for_first_frame_if_full_layout = False
    if session_processing_mode == 'full_layout':
        recalibrate_input = input("是否对本次会话的第一帧强制重新校准布局 (仅对整版模式有效)? (输入 'y' 或 'Y' 后回车，否则不校准): ").strip().lower()
        force_recalibrate_for_first_frame_if_full_layout = (recalibrate_input == 'y')
        if force_recalibrate_for_first_frame_if_full_layout:
            print(f"  客户端：本次会话的第一帧（如果是整版模式）将发送强制重新校准请求。")

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
        elif session_processing_mode == 'full_layout':
             print(f"  客户端：为图片 '{image_basename}' 发送常规“整版识别”处理请求。")
        else: # scattered_ocr mode
             print(f"  客户端：为图片 '{image_basename}' 发送“零散识别”处理请求。")

        json_response = send_image_for_prediction(
            img_path,
            current_batch_session_id,
            frame_count,
            mode_to_use=session_processing_mode,
            force_recalibrate_payload=should_force_recalibrate_this_call
        )

        if json_response:
            session_status = json_response.get("session_status")
            if session_status == "completed":
                print(f"客户端：会话 {current_batch_session_id} (整版识别) 已完成！")
            elif session_status == "scattered_ocr_completed":
                print(f"客户端：图片 '{image_basename}' (零散识别) 处理完成。")
            elif session_status == "first_frame_anchor_error": # 处理服务端返回的特定错误状态
                 print(f"客户端：【重要】图片 '{image_basename}' (整版识别-首帧) 未能满足拍摄规定，处理中断！请检查服务端警告。")
                 print(f"客户端：由于首帧锚定失败，会话 {current_batch_session_id} 可能无法继续，建议重新开始新会话并按规定拍摄首帧。")
                 # 可以选择在这里 break 中断后续图片的发送，如果首帧失败则不继续
                 # break

        if frame_count < len(IMAGE_PATHS_TO_UPLOAD):
            # time.sleep(0.2) # 可以加一个非常短的延时，或者直接进入下一轮
            print("-" * 30)

    print(f"\n客户端：会话 {current_batch_session_id} 的所有指定图片已发送。")
    print("客户端测试完成。")