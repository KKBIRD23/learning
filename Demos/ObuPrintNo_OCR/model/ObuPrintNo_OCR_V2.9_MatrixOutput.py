# coding: utf-8
"""
OBU (车载单元) 镭标码识别与矩阵输出脚本
版本: v2.9.3_Smart_Grid_Generation
功能:
- 核心: 智能识别特殊行, 基于YOLO锚点和布局先验进行透视感知的理想网格推断, 并填充识别结果。
- YOLO检测条码作为OBU锚点。
- PaddleOCR识别数字。
- 输出最终的OBU矩阵。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr
import onnxruntime
from collections import Counter
from scipy.spatial.distance import cdist
from datetime import datetime

# --- V2.9.3 配置参数 ---
VERSION = "v2.9.3_Smart_Grid_Generation"
IMAGE_PATHS = [
    r"../../DATA/PIC/1.JPG",
    r"../../DATA/PIC/2.JPG",
    r"../../DATA/PIC/3.JPG"
]
BASE_OUTPUT_DIR = "./output_v2.9_smart_grid"
TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP_NOW}_{VERSION}")
os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True)

# --- PaddleOCR 初始化相关参数 ---
LANG_CFG = 'en'; USE_TEXTLINE_ORIENTATION_CFG = False; USE_DOC_ORIENTATION_CLASSIFY_CFG = False
USE_DOC_UNWARPING_CFG = False; OCR_VERSION_CFG = None
TEXT_DETECTION_MODEL_DIR_CFG = None; TEXT_RECOGNITION_MODEL_DIR_CFG = None
TEXT_DETECTION_MODEL_NAME_CFG = None; TEXT_RECOGNITION_MODEL_NAME_CFG = None
PADDLE_OCR_FINE_PARAMS = {"text_det_limit_side_len": 960, "text_det_thresh": 0.3,
                          "text_det_box_thresh": 0.6, "text_rec_score_thresh": 0.5}

# --- OBU码筛选规则 ---
OBU_CODE_PREFIX_FILTER_CFG = "5001"; OBU_CODE_LENGTH_FILTER_CFG = 16

# --- YOLOv8 相关配置 ---
YOLO_ONNX_MODEL_PATH_CFG = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
YOLO_CONFIDENCE_THRESHOLD_CFG = 0.25; YOLO_IOU_THRESHOLD_CFG = 0.45
YOLO_INPUT_WIDTH_CFG = 640; YOLO_INPUT_HEIGHT_CFG = 640

# --- 矩阵与布局先验配置 ---
LAYOUT_CONFIG = {
    "total_obus": 50,               # 期望的总OBU数量
    "regular_rows_count": 12,       # 常规行的数量
    "regular_cols_count": 4,        # 常规行每行的OBU数量
    "special_row_cols_count": 2,    # 特殊行（只有2个OBU）的列数
    "expected_total_rows": 13,      # 期望的总行数 (常规行 + 特殊行)
    "special_row_exists": True      # <--- 新增这一行，明确告知有特殊行
}

# --- 算法相关阈值 ---
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5 # Y坐标差异阈值 = 平均框高 * 此因子
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.75 # 匹配距离阈值因子 (乘以平均YOLO条码宽度)
MIN_YOLO_ANCHORS_FOR_LAYOUT = 10 # 至少需要的YOLO锚点数
MIN_OBUS_FOR_RELIABLE_ROW = 2 # 一行中至少要有这么多OBU才认为它对间距估计有贡献

# --- 全局变量 ---
paddle_ocr_engine_global = None; yolo_session_global = None

# --- 函数定义 ---
# initialize_paddleocr, load_yolo_model (与V2.9.2一致)
# preprocess_for_yolo, non_max_suppression_global, postprocess_yolo_detections, get_yolo_barcode_anchors (来自V2.9.2)
# get_box_center_and_dims (来自V2.9.2)
# draw_ocr_results_refined (来自V2.8.4, 用于临时可视化PaddleOCR原始结果)
# print_matrix_to_console (来自V2.9.1)
# --- 请确保这些辅助函数已正确包含在脚本中 ---
# (为了聚焦核心算法，我再次省略这些辅助函数的代码，假设它们已正确无误)
# ... (Paste the helper functions from V2.9.2/V2.8.4 here) ...
def initialize_paddleocr():
    global paddle_ocr_engine_global
    init_params = {'lang': LANG_CFG, 'use_textline_orientation': USE_TEXTLINE_ORIENTATION_CFG, 'use_doc_orientation_classify': USE_DOC_ORIENTATION_CLASSIFY_CFG, 'use_doc_unwarping': USE_DOC_UNWARPING_CFG, 'ocr_version': OCR_VERSION_CFG, 'text_detection_model_dir': TEXT_DETECTION_MODEL_DIR_CFG, 'text_recognition_model_dir': TEXT_RECOGNITION_MODEL_DIR_CFG,'text_detection_model_name': TEXT_DETECTION_MODEL_NAME_CFG, 'text_recognition_model_name': TEXT_RECOGNITION_MODEL_NAME_CFG, **PADDLE_OCR_FINE_PARAMS }
    ocr_params_final_filtered = {k: v for k, v in init_params.items() if v is not None}
    print(f"\n正在使用以下参数初始化PaddleOCR: {ocr_params_final_filtered}")
    try: paddle_ocr_engine_global = paddleocr.PaddleOCR(**ocr_params_final_filtered); print("PaddleOCR引擎初始化成功。"); return True
    except Exception as e: print(f"PaddleOCR引擎初始化失败: {e}"); paddle_ocr_engine_global = None; return False

def load_yolo_model():
    global yolo_session_global
    if not os.path.exists(YOLO_ONNX_MODEL_PATH_CFG): print(f"错误: YOLO ONNX模型未找到: {YOLO_ONNX_MODEL_PATH_CFG}"); return False
    try: print(f"正在加载YOLO模型: {YOLO_ONNX_MODEL_PATH_CFG}"); yolo_session_global = onnxruntime.InferenceSession(YOLO_ONNX_MODEL_PATH_CFG, providers=['CPUExecutionProvider']); print("YOLO模型加载成功。"); return True
    except Exception as e: print(f"YOLO模型加载失败: {e}"); yolo_session_global = None; return False

def preprocess_for_yolo(img_data, target_h, target_w):
    img_height_orig, img_width_orig = img_data.shape[:2]; ratio = min(target_w / img_width_orig, target_h / img_height_orig); new_w, new_h = int(img_width_orig * ratio), int(img_height_orig * ratio); resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR); canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8); pad_x = (target_w - new_w) // 2; pad_y = (target_h - new_h) // 2; canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img; input_tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0; input_tensor = np.expand_dims(input_tensor, axis=0); return input_tensor, ratio, pad_x, pad_y

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold):
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    if not isinstance(scores, np.ndarray) or scores.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]; areas=(x2-x1+1e-6)*(y2-y1+1e-6); order=scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i);_ = order.size;order = order[1:]
        if _ == 1: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order]);xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h;ovr=inter/(areas[i]+areas[order]-inter+1e-6)
        inds=np.where(ovr<=iou_threshold)[0];order=order[inds]
    return keep

def postprocess_yolo_detections(outputs_onnx, conf_threshold, iou_threshold,original_shape_hw, model_input_shape_hw,ratio_preproc, pad_x_preproc, pad_y_preproc):
    raw_output_tensor = np.squeeze(outputs_onnx[0]);
    if raw_output_tensor.ndim != 2: print(f"错误: YOLO输出张量维度不为2. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate = [], []
    expected_attributes = 4 + 1
    for pred_data in predictions_to_iterate:
        if len(pred_data) != expected_attributes: continue
        final_confidence = float(pred_data[4])
        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = pred_data[:4]; x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2]); scores_candidate.append(final_confidence)
    if not boxes_candidate: return []
    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)
    final_barcode_boxes_xyxy = []; orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx); box_model_coords = boxes_candidate[idx]
        box_no_pad_x1,box_no_pad_y1 = box_model_coords[0]-pad_x_preproc,box_model_coords[1]-pad_y_preproc; box_no_pad_x2,box_no_pad_y2 = box_model_coords[2]-pad_x_preproc,box_model_coords[3]-pad_y_preproc
        if ratio_preproc == 0: continue
        orig_x1,orig_y1 = box_no_pad_x1/ratio_preproc,box_no_pad_y1/ratio_preproc; orig_x2,orig_y2 = box_no_pad_x2/ratio_preproc,box_no_pad_y2/ratio_preproc
        final_x1,final_y1 = int(np.clip(orig_x1,0,orig_w)),int(np.clip(orig_y1,0,orig_h)); final_x2,final_y2 = int(np.clip(orig_x2,0,orig_w)),int(np.clip(orig_y2,0,orig_h))
        final_barcode_boxes_xyxy.append([final_x1, final_y1, final_x2, final_y2])
    return final_barcode_boxes_xyxy

def get_yolo_barcode_anchors(image):
    if not yolo_session_global: print("错误: YOLO会话未初始化。"); return [], 0.0
    input_tensor, ratio, pad_x, pad_y = preprocess_for_yolo(image, YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG)
    input_name = yolo_session_global.get_inputs()[0].name
    t_start = time.time(); outputs = yolo_session_global.run(None, {input_name: input_tensor}); yolo_predict_time = time.time() - t_start
    print(f"  YOLO predict() 耗时 {yolo_predict_time:.3f}s")
    detected_barcode_boxes = postprocess_yolo_detections(outputs_onnx=outputs, conf_threshold=YOLO_CONFIDENCE_THRESHOLD_CFG, iou_threshold=YOLO_IOU_THRESHOLD_CFG, original_shape_hw=image.shape[:2], model_input_shape_hw=(YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG), ratio_preproc=ratio, pad_x_preproc=pad_x, pad_y_preproc=pad_y)
    print(f"  YOLO检测到 {len(detected_barcode_boxes)} 个条码框。")
    return detected_barcode_boxes, yolo_predict_time

def get_box_center_and_dims(box_xyxy_or_poly):
    if box_xyxy_or_poly is None: return None, None, None, None
    if isinstance(box_xyxy_or_poly, (list, np.ndarray)) and len(box_xyxy_or_poly) == 0: return None, None, None, None
    if len(box_xyxy_or_poly) == 4 and isinstance(box_xyxy_or_poly[0], (int, float, np.int32, np.float32)):
        x1, y1, x2, y2 = box_xyxy_or_poly; return int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2-x1), int(y2-y1)
    elif isinstance(box_xyxy_or_poly, (list, np.ndarray)) and len(box_xyxy_or_poly) > 0 and isinstance(box_xyxy_or_poly[0], (list, np.ndarray)):
        points = np.array(box_xyxy_or_poly, dtype=np.int32)
        if len(points) > 0: x, y, w, h = cv2.boundingRect(points); return x + w // 2, y + h // 2, w, h
    # print(f"警告: get_box_center_and_dims 接收到无法解析的box格式: {box_xyxy_or_poly}"); # 减少不必要的打印
    return None, None, None, None

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.png"):
    img_out = image.copy();_c = cv2
    if img_out is None: print(f"错误: 用于绘制的输入图像为None。无法保存到 {output_path}"); return
    if not all_ocr_data and not potential_obu_data :
        print(f"没有OCR数据可以绘制到 {output_path}.")
        try: _c.imwrite(output_path, img_out); print(f"无OCR数据, 底图已保存到: {output_path}")
        except Exception as e_save: print(f"保存底图失败 {output_path}: {e_save}")
        return
    if all_ocr_data:
        for item in all_ocr_data:
            box_polygon = item.get('box')
            if box_polygon is not None and isinstance(box_polygon, (list, np.ndarray)) and len(box_polygon) > 0 :
                try: points = np.array(box_polygon, dtype=np.int32); _c.polylines(img_out, [points], isClosed=True, color=(0, 180, 0), thickness=1)
                except Exception as e_draw_poly: print(f"警告: 无法为检测框 {box_polygon} 绘制多边形. 错误: {e_draw_poly}")
    drawn_potential_text_count = 0
    if potential_obu_data:
        for item in potential_obu_data:
            text = item['text']; box_polygon = item.get('box')
            if box_polygon is None or not isinstance(box_polygon, (list, np.ndarray)) or len(box_polygon) == 0: continue
            try:
                points = np.array(box_polygon, dtype=np.int32); _c.polylines(img_out, [points], isClosed=True, color=(255, 0, 0), thickness=3)
                label = f"{text}"; text_anchor_x = points[0][0]; text_anchor_y = points[0][1] - 10
                if text_anchor_y < 15 : text_anchor_y = points[0][1] + 25
                (text_width, text_height), baseline = _c.getTextSize(label, _c.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                _c.rectangle(img_out, (text_anchor_x, text_anchor_y - text_height - baseline + 1), (text_anchor_x + text_width, text_anchor_y + baseline -1), (220,220,220), -1)
                _c.putText(img_out, label, (text_anchor_x, text_anchor_y), _c.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 0), 2)
                drawn_potential_text_count +=1
            except Exception as e_draw_potential: print(f"警告: 无法为潜在OBU绘制检测框 {box_polygon}. 错误: {e_draw_potential}")
        if drawn_potential_text_count > 0: print(f"已在图上绘制 {drawn_potential_text_count} 个潜在OBU的文本。")
    try: _c.imwrite(output_path, img_out); print(f"OCR可视化结果已保存到: {output_path}")
    except Exception as e: print(f"保存可视化图片失败 {output_path}: {e}")

def print_matrix_to_console(matrix, strategy_name=""):
    if not matrix or (isinstance(matrix, list) and not matrix[0]): print(f"  {strategy_name} 生成的矩阵为空，无法打印。"); return
    print(f"\n--- {strategy_name} OBU识别矩阵 ({len(matrix)}行 x {len(matrix[0]) if matrix and matrix[0] else 0}列) ---")
    for row in matrix:
        row_display = ["  红  " if item == "未识别" else (f" {item[-4:]} " if isinstance(item, str) and item.startswith("5001") and len(item)>4 else f" {str(item)[:4]:^4} ") for item in row]
        print(" ".join(row_display))
    print("---------------------------------------------")

# --- 核心：智能网格生成与填充 ---
def build_matrix_smart_grid(yolo_anchors_input, paddle_results_input, layout_config, image_wh):
    """
    核心函数：通过YOLO锚点、特殊行识别和布局先验，精确推断理想网格，并用PaddleOCR结果填充。
    """
    print("  正在执行智能网格矩阵构建...")
    if not yolo_anchors_input or len(yolo_anchors_input) < MIN_YOLO_ANCHORS_FOR_LAYOUT:
        print(f"  YOLO锚点数量 ({len(yolo_anchors_input)}) 不足 ({MIN_YOLO_ANCHORS_FOR_LAYOUT}个)，无法进行可靠布局推断。")
        # 返回一个符合预期总行数和常规列数的空矩阵
        return [["YOLO锚点不足"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0

    yolo_anchors_sorted = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))

    yolo_rows_grouped = []
    # ... (YOLO行分组逻辑与您V2.9.3版本一致，确保 current_row_for_grouping 在循环外初始化或正确处理)
    if not yolo_anchors_sorted: return [["无有效YOLO锚点"]*layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])],0
    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors_sorted if a['h'] > 0]) if any(a['h'] > 0 for a in yolo_anchors_sorted) else 30
    y_threshold = avg_h_yolo * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR
    current_row_for_grouping = [yolo_anchors_sorted[0]]
    for i in range(1, len(yolo_anchors_sorted)):
        if abs(yolo_anchors_sorted[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold:
            current_row_for_grouping.append(yolo_anchors_sorted[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
            current_row_for_grouping = [yolo_anchors_sorted[i]]
    if current_row_for_grouping: yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
    print(f"  YOLO锚点初步分为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")

    # --- 步骤3b: 生成理想坑位坐标 (改进的占位逻辑，尝试生成接近50个) ---
    print("  警告: 理想坑位生成逻辑仍在优化中，当前为初步实现。")
    ideal_grid_slots = []

    # 尝试从YOLO行中获取一些全局参数
    avg_obu_w_overall = np.mean([a['w'] for a in yolo_anchors_sorted if a['w'] > 0]) if any(a['w'] > 0 for a in yolo_anchors_sorted) else 100
    avg_obu_h_overall = np.mean([a['h'] for a in yolo_anchors_sorted if a['h'] > 0]) if any(a['h'] > 0 for a in yolo_anchors_sorted) else 40

    # 估算行Y坐标 (如果YOLO行数不足，则基于平均行高进行补充)
    estimated_row_y_coords = []
    if len(yolo_rows_grouped) >= 1:
        for r_group in yolo_rows_grouped:
            estimated_row_y_coords.append(np.mean([a['cy'] for a in r_group]))
        # 如果YOLO行数少于预期的13行，尝试补充
        while len(estimated_row_y_coords) < layout_config["expected_total_rows"] and len(estimated_row_y_coords) > 0:
            estimated_row_y_coords.append(estimated_row_y_coords[-1] + avg_obu_h_overall * 1.5) # 简单向下延伸
    else: # 如果YOLO完全没分出行，用一个非常粗略的估计
        img_h = image_wh[0]
        for r in range(layout_config["expected_total_rows"]):
            estimated_row_y_coords.append((r + 0.5) * (img_h / layout_config["expected_total_rows"]))

    # 尝试识别特殊行 (非常简化的逻辑)
    special_row_is_at_top = None
    if yolo_rows_grouped:
        if len(yolo_rows_grouped[0]) == layout_config["special_row_cols_count"]: special_row_is_at_top = True
        elif len(yolo_rows_grouped[-1]) == layout_config["special_row_cols_count"]: special_row_is_at_top = False

    # 生成理想坑位 (更努力地凑齐50个)
    current_obu_idx = 0
    for r_log in range(layout_config["expected_total_rows"]):
        cols_this_row = layout_config["regular_cols_count"]
        is_special = False
        if layout_config["special_row_exists"]:
            if special_row_is_at_top is True and r_log == 0:
                cols_this_row = layout_config["special_row_cols_count"]; is_special = True
            elif special_row_is_at_top is False and r_log == layout_config["expected_total_rows"] - 1:
                cols_this_row = layout_config["special_row_cols_count"]; is_special = True

        # 获取当前行的Y坐标
        current_cy = estimated_row_y_coords[r_log] if r_log < len(estimated_row_y_coords) else estimated_row_y_coords[-1] + avg_obu_h_overall

        # 获取当前行的X坐标参考 (从对应的YOLO行，或用全局平均)
        # 这是一个很大的简化，理想情况下每行的X起点和间距都应考虑透视
        x_coords_this_row = []
        if r_log < len(yolo_rows_grouped) and yolo_rows_grouped[r_log]:
            # 使用当前YOLO行内的X坐标作为参考
            ref_yolo_row = yolo_rows_grouped[r_log]
            for c_log in range(cols_this_row):
                if c_log < len(ref_yolo_row):
                    x_coords_this_row.append(ref_yolo_row[c_log]['cx'])
                elif x_coords_this_row: # 如果YOLO检测不足，尝试外推
                    x_coords_this_row.append(x_coords_this_row[-1] + avg_obu_w_overall * 1.1)
                else: # 如果行首就缺失，用一个粗略的图像左边距
                    x_coords_this_row.append(avg_obu_w_overall * (c_log + 0.5))
        else: # 如果YOLO中没有对应行，则基于全局平均X分布生成
            img_w = image_wh[1]
            # 粗略地使列居中分布
            total_width_of_row = cols_this_row * avg_obu_w_overall + (cols_this_row - 1) * (avg_obu_w_overall * 0.1)
            start_x_this_row = (img_w - total_width_of_row) / 2 + avg_obu_w_overall / 2
            for c_log in range(cols_this_row):
                 x_coords_this_row.append(start_x_this_row + c_log * (avg_obu_w_overall * 1.1))

        for c_log in range(cols_this_row):
            if current_obu_idx >= layout_config["total_obus"]: break
            ideal_grid_slots.append({
                'logical_row': r_log, 'logical_col': c_log,
                'cx': int(x_coords_this_row[c_log] if c_log < len(x_coords_this_row) else (img_w/2)), # 保护
                'cy': int(current_cy),
                'w': int(avg_obu_w_overall), 'h': int(avg_obu_h_overall)
            })
            current_obu_idx += 1
        if current_obu_idx >= layout_config["total_obus"]: break

    if not ideal_grid_slots:
        print("  未能生成理想坑位坐标。");
        return [["无理想坑位"]*layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0
    print(f"  已生成 {len(ideal_grid_slots)} 个理想坑位坐标。")

    # 4. 将PaddleOCR识别结果填充到理想坑位
    # (填充逻辑与V2.9.2基本一致，但现在final_matrix的维度是固定的13x4)
    final_matrix = [["未识别"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])]
    matrix_filled_count = 0

    paddle_results_with_center = []
    if paddle_results_input:
        for pr in paddle_results_input:
            cx, cy, pw, ph = get_box_center_and_dims(pr['box'])
            if cx is not None: paddle_results_with_center.append({**pr, 'cx': cx, 'cy': cy, 'w':pw, 'h':ph, 'used': False})

    if ideal_grid_slots and paddle_results_with_center:
        ideal_coords = np.array([[slot['cx'], slot['cy']] for slot in ideal_grid_slots])
        paddle_coords = np.array([[p['cx'], p['cy']] for p in paddle_results_with_center])

        if paddle_coords.size == 0: print("  警告: 没有有效的PaddleOCR中心点用于匹配。")
        else:
            dist_matrix = cdist(ideal_coords, paddle_coords) # ideal_slots x paddle_results

            # 为每个paddle_result找到最佳的ideal_slot (避免一个slot被多个paddle结果填充)
            # 或者为每个ideal_slot找到最佳的paddle_result (当前做法)

            # 使用匈牙利算法或简单的贪婪匹配（确保一对一）会更好，但先用简单距离
            for i_slot, slot in enumerate(ideal_grid_slots):
                log_r, log_c = slot['logical_row'], slot['logical_col']

                # 确定当前逻辑格子的正确列数（特殊行只有2列）
                cols_for_current_logical_row = layout_config["regular_cols_count"]
                if layout_config["special_row_exists"]:
                    is_current_log_row_special = False
                    if special_row_is_at_top is True and log_r == 0: is_current_log_row_special = True
                    elif special_row_is_at_top is False and log_r == layout_config["expected_total_rows"] - 1: is_current_log_row_special = True
                    if is_current_log_row_special: cols_for_current_logical_row = layout_config["special_row_cols_count"]

                if log_r >= layout_config["expected_total_rows"] or log_c >= cols_for_current_logical_row : continue

                best_paddle_idx = -1; min_dist_to_slot = float('inf')
                # 使用slot中估算的宽度作为距离阈值参考
                max_dist_thresh = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * slot.get('w', avg_obu_w_overall)

                for j_paddle, p_obu in enumerate(paddle_results_with_center):
                    if p_obu['used']: continue
                    if i_slot < dist_matrix.shape[0] and j_paddle < dist_matrix.shape[1]:
                        current_dist = dist_matrix[i_slot, j_paddle]
                        if current_dist < max_dist_thresh and current_dist < min_dist_to_slot:
                            min_dist_to_slot = current_dist; best_paddle_idx = j_paddle

                if best_paddle_idx != -1:
                    final_matrix[log_r][log_c] = paddle_results_with_center[best_paddle_idx]['text']
                    paddle_results_with_center[best_paddle_idx]['used'] = True; matrix_filled_count += 1

    print(f"  智能网格方案: 构建矩阵 {len(final_matrix)}x{len(final_matrix[0]) if final_matrix else 0}, 填充OBU数: {matrix_filled_count}")
    return final_matrix, matrix_filled_count

# --- 主程序 (与V2.9.2类似，调用 build_matrix_smart_grid) ---
if __name__ == "__main__":
    # ... (与V2.9.2的主程序结构基本一致，主要是调用 build_matrix_smart_grid)
    overall_start_time = time.time()
    print(f"--- OBU识别与矩阵输出工具 {VERSION} ---"); print(f"输出目录: {os.path.abspath(CURRENT_RUN_OUTPUT_DIR)}")
    if not initialize_paddleocr(): exit()
    if not load_yolo_model(): print("警告: YOLO模型加载失败。")
    for image_path_current in IMAGE_PATHS:
        print(f"\n\n========== 处理图片: {image_path_current} ==========")
        img_filename_base = os.path.splitext(os.path.basename(image_path_current))[0]
        original_image = cv2.imread(image_path_current)
        if original_image is None: print(f"错误: 无法读取图片 {image_path_current}"); continue
        print(f"\n--- 步骤1: PaddleOCR 文本检测与识别 ---")
        t_start_paddle = time.time(); ocr_prediction_result = paddle_ocr_engine_global.predict(original_image); paddle_predict_time = time.time() - t_start_paddle
        print(f"PaddleOCR predict() 完成, 耗时: {paddle_predict_time:.3f}s")
        all_paddle_ocr_data = []; potential_obu_list_paddle = []
        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]; dt_polys = ocr_result_object.get('dt_polys'); rec_texts = ocr_result_object.get('rec_texts'); rec_scores = ocr_result_object.get('rec_scores')
            if rec_texts and rec_scores and dt_polys:
                max_items = 0
                if rec_texts and rec_scores and dt_polys:
                    max_items = min(len(rec_texts), len(rec_scores), len(dt_polys))
                    if len(rec_texts) != max_items or len(rec_scores) != max_items or len(dt_polys) != max_items : print(f"  警告: PaddleOCR原始输出长度不匹配: texts({len(rec_texts)}), scores({len(rec_scores)}), boxes({len(dt_polys)}). 按最短 {max_items} 处理。")
                for i in range(max_items):
                    item_data = {"text": str(rec_texts[i]), "score": float(rec_scores[i]), "box": dt_polys[i]}
                    all_paddle_ocr_data.append(item_data)
                    text_check = item_data['text'].strip()
                    if text_check.startswith(OBU_CODE_PREFIX_FILTER_CFG) and len(text_check) == OBU_CODE_LENGTH_FILTER_CFG and text_check.isdigit() and item_data['score'] >= PADDLE_OCR_FINE_PARAMS['text_rec_score_thresh']:
                        potential_obu_list_paddle.append(item_data)
        print(f"PaddleOCR 原始有效文本 {len(all_paddle_ocr_data)} 条, 内容筛选后潜在OBU {len(potential_obu_list_paddle)} 个。")
        yolo_anchors_for_matrix = []
        if yolo_session_global:
            print(f"\n--- 步骤2: YOLO 条码锚点检测 ---")
            yolo_barcode_boxes_xyxy, _ = get_yolo_barcode_anchors(original_image.copy())
            for box in yolo_barcode_boxes_xyxy:
                cx, cy, w, h = get_box_center_and_dims(box)
                if cx is not None: yolo_anchors_for_matrix.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box})

        final_matrix, filled_count = build_matrix_smart_grid(yolo_anchors_for_matrix, potential_obu_list_paddle, LAYOUT_CONFIG, original_image.shape[:2])
        print_matrix_to_console(final_matrix, f"智能网格矩阵 - {img_filename_base}")

        temp_viz_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"paddle_viz_{img_filename_base}_{VERSION}.png")
        if original_image is not None: draw_ocr_results_refined(original_image, all_paddle_ocr_data, potential_obu_list_paddle, temp_viz_path)
    overall_end_time = time.time(); total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。"); print(f"-------------------------------------------------")