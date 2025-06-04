# coding: utf-8
"""
OBU (车载单元) 镭标码识别与矩阵输出脚本
版本: v2.9.5_YOLO_Direct_Grid
功能:
- 实现“YOLO直接网格填充”策略：YOLO检测结果直接定义矩阵结构。
- PaddleOCR识别数字并填充到YOLO网格中。
- 新增对比分析：比较PaddleOCR原始检测框与YOLO条码框的位置。
- 可视化两种结果：最终填充的矩阵，以及YOLO与PaddleOCR检测框的对比。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr
import onnxruntime
from collections import Counter
from scipy.spatial.distance import cdist # 仍然用于匹配
from datetime import datetime

# --- V2.9.5 配置参数 ---
VERSION = "v2.9.5_YOLO_Direct_Grid"
IMAGE_PATHS = [
    r"../../DATA/PIC/1.JPG", # 请确保路径正确
    r"../../DATA/PIC/2.JPG",
    r"../../DATA/PIC/3.JPG",
]
BASE_OUTPUT_DIR = "./output_v2.9_direct_grid"
TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP_NOW}_{VERSION}")
os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True)

# --- PaddleOCR 初始化相关参数 (使用黄金参数) ---
LANG_CFG = 'en'; USE_TEXTLINE_ORIENTATION_CFG = False; USE_DOC_ORIENTATION_CLASSIFY_CFG = False
USE_DOC_UNWARPING_CFG = False; OCR_VERSION_CFG = None
TEXT_DETECTION_MODEL_DIR_CFG = None; TEXT_RECOGNITION_MODEL_DIR_CFG = None
TEXT_DETECTION_MODEL_NAME_CFG = None; TEXT_RECOGNITION_MODEL_NAME_CFG = None
PADDLE_OCR_FINE_PARAMS = {"text_det_limit_side_len": 960, "text_det_thresh": 0.3,
                          "text_det_box_thresh": 0.6, "text_rec_score_thresh": 0.5}

# --- OBU码筛选规则 (PaddleOCR后处理) ---
OBU_CODE_PREFIX_FILTER_CFG = "5001"; OBU_CODE_LENGTH_FILTER_CFG = 16

# --- YOLOv8 相关配置 ---
YOLO_ONNX_MODEL_PATH_CFG = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
YOLO_CONFIDENCE_THRESHOLD_CFG = 0.25; YOLO_IOU_THRESHOLD_CFG = 0.45
YOLO_INPUT_WIDTH_CFG = 640; YOLO_INPUT_HEIGHT_CFG = 640

# --- 矩阵构建与匹配参数 ---
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5
PADDLE_OBU_TO_YOLO_MAX_DIST_FACTOR = 0.8 # Paddle数字中心与YOLO条码上方预期数字区中心的最大匹配距离因子 (乘以YOLO条码宽度)
# 预期的数字区域相对于YOLO条码框的偏移和尺寸 (用于匹配)
DIGIT_AREA_RELATIVE_Y_OFFSET_FROM_YOLO = -0.8
DIGIT_AREA_SEARCH_HEIGHT_FACTOR_FROM_YOLO = 1.0
DIGIT_AREA_SEARCH_WIDTH_FACTOR_FROM_YOLO = 1.2

# --- 全局变量 ---
paddle_ocr_engine_global = None; yolo_session_global = None

# --- 函数定义 ---
# initialize_paddleocr, load_yolo_model, preprocess_for_yolo, non_max_suppression_global,
# postprocess_yolo_detections, get_yolo_barcode_anchors, get_box_center_and_dims,
# print_matrix_to_console 与V2.9.2/V2.9.1一致，此处省略。
# draw_ocr_results_refined 也可以复用，但我们可能需要一个新的绘图函数用于对比。
# --- 请确保这些辅助函数已正确包含在脚本中 ---
# ... (Paste helper functions from V2.9.2 here) ...
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
    return None, None, None, None

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.png"):
    # (与V2.8.4版本一致)
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


# --- 新增：YOLO直接网格填充的矩阵构建 ---
def build_matrix_yolo_direct_fill(yolo_barcode_boxes, potential_paddle_obus):
    """
    策略B：基于YOLO检测到的条码框直接构建物理网格，并用PaddleOCR结果填充。
    Args:
        yolo_barcode_boxes (list): YOLO检测到的条码框列表 [[x1,y1,x2,y2], ...]
        potential_paddle_obus (list): PaddleOCR筛选后的OBU结果列表
                                     [{"text":..., "score":..., "box":poly, 'cx', 'cy', 'w', 'h'}, ...]
    Returns:
        list: 二维矩阵, int: 填充的OBU数量
    """
    print("  正在执行YOLO直接网格填充矩阵构建...")
    if not yolo_barcode_boxes:
        print("  YOLO未检测到条码，无法构建直接填充矩阵。")
        return [], 0

    # 1. 为YOLO框计算中心点和尺寸，并按Y坐标主序，X坐标次序排序，形成行分组
    yolo_anchors_with_info = []
    for box in yolo_barcode_boxes:
        cx, cy, w, h = get_box_center_and_dims(box)
        if cx is not None:
            yolo_anchors_with_info.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box})

    if not yolo_anchors_with_info: return [], 0
    yolo_anchors_with_info.sort(key=lambda a: (a['cy'], a['cx']))

    yolo_rows_grouped = []
    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors_with_info if a['h'] > 0]) if any(a['h'] > 0 for a in yolo_anchors_with_info) else 30
    y_threshold = avg_h_yolo * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    current_row = [yolo_anchors_with_info[0]]
    for i in range(1, len(yolo_anchors_with_info)):
        if abs(yolo_anchors_with_info[i]['cy'] - current_row[-1]['cy']) < y_threshold:
            current_row.append(yolo_anchors_with_info[i])
        else:
            yolo_rows_grouped.append(sorted(current_row, key=lambda a: a['cx']))
            current_row = [yolo_anchors_with_info[i]]
    if current_row: yolo_rows_grouped.append(sorted(current_row, key=lambda a: a['cx']))

    num_rows = len(yolo_rows_grouped)
    num_cols = 0
    if yolo_rows_grouped and any(yolo_rows_grouped): # 确保yolo_rows_grouped不为空且内部行不为空
        num_cols = max(len(r) for r in yolo_rows_grouped) if yolo_rows_grouped else 0

    if num_rows == 0 or num_cols == 0:
        print("  YOLO未能形成有效行列结构用于直接填充。")
        return [], 0

    matrix = [["未识别"] * num_cols for _ in range(num_rows)]
    filled_count = 0

    # 为PaddleOCR结果也计算中心点 (如果之前没算的话)
    paddle_obus_with_center = []
    for obu in potential_paddle_obus:
        if 'cx' not in obu or 'cy' not in obu: # 确保有中心点
            cx, cy, _, _ = get_box_center_and_dims(obu['box'])
            if cx is not None:
                paddle_obus_with_center.append({**obu, 'cx':cx, 'cy':cy, 'used':False})
        else:
            paddle_obus_with_center.append({**obu, 'used':False})


    for r_idx, yolo_row in enumerate(yolo_rows_grouped):
        for c_idx, yolo_anchor in enumerate(yolo_row):
            # 定义此YOLO锚点对应的数字区域预期中心
            expected_digit_cx = yolo_anchor['cx']
            expected_digit_cy = yolo_anchor['cy'] + int(yolo_anchor['h'] * DIGIT_AREA_RELATIVE_Y_OFFSET_FROM_YOLO)

            best_match_paddle_obu = None
            min_dist_sq = (PADDLE_OBU_TO_YOLO_MAX_DIST_FACTOR * yolo_anchor['w'])**2

            for p_obu in paddle_obus_with_center:
                if p_obu['used']: continue
                dist_sq = (p_obu['cx'] - expected_digit_cx)**2 + (p_obu['cy'] - expected_digit_cy)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_match_paddle_obu = p_obu

            if best_match_paddle_obu:
                matrix[r_idx][c_idx] = best_match_paddle_obu['text']
                best_match_paddle_obu['used'] = True
                filled_count += 1

    print(f"  YOLO直接网格填充方案: 构建矩阵 {num_rows}x{num_cols}, 填充OBU数: {filled_count}")
    return matrix, filled_count

# --- 新增：YOLO与PaddleOCR检测框对比分析与可视化 ---
def analyze_and_draw_yolo_paddle_comparison(image, yolo_barcode_boxes, all_paddle_ocr_data, output_path):
    """
    对比YOLO条码框和PaddleOCR原始文本检测框，并进行可视化。
    Args:
        image: 原始图像
        yolo_barcode_boxes: YOLO检测的条码框 [[x1,y1,x2,y2], ...]
        all_paddle_ocr_data: PaddleOCR所有原始识别结果 [{"text":..., "score":..., "box":poly}, ...]
        output_path: 对比可视化图片的保存路径
    """
    print("  正在执行YOLO与PaddleOCR检测框对比分析...")
    img_compare_viz = image.copy()

    # 绘制YOLO条码框 (例如用红色)
    for y_box in yolo_barcode_boxes:
        cv2.rectangle(img_compare_viz, (y_box[0], y_box[1]), (y_box[2], y_box[3]), (0, 0, 255), 2) # 红色

    # 绘制PaddleOCR原始检测框 (例如用绿色)
    # 并且可以尝试标记那些与YOLO条码框（上方数字区域）匹配度高的PaddleOCR框
    yolo_match_count = 0
    paddle_matched_to_yolo_indices = set()

    for y_idx, y_box in enumerate(yolo_barcode_boxes):
        y_cx, y_cy, y_w, y_h = get_box_center_and_dims(y_box)
        if y_cx is None: continue

        # 定义YOLO条码框上方的预期数字区域 (用于匹配)
        expected_digit_center_y = y_cy + int(y_h * DIGIT_AREA_RELATIVE_Y_OFFSET_FROM_YOLO)
        search_half_h = int(y_h * DIGIT_AREA_SEARCH_HEIGHT_FACTOR_FROM_YOLO / 2)
        search_half_w = int(y_w * DIGIT_AREA_SEARCH_WIDTH_FACTOR_FROM_YOLO / 2)

        expected_digit_roi = [y_cx - search_half_w, expected_digit_center_y - search_half_h,
                              y_cx + search_half_w, expected_digit_center_y + search_half_h]

        found_match_for_this_yolo = False
        for p_idx, p_data in enumerate(all_paddle_ocr_data):
            if p_idx in paddle_matched_to_yolo_indices: continue # 避免重复匹配

            p_poly = p_data['box']
            p_cx, p_cy, _, _ = get_box_center_and_dims(p_poly)
            if p_cx is None: continue

            # 简单匹配：如果PaddleOCR文本中心落在YOLO条码上方的预期数字ROI内
            if expected_digit_roi[0] <= p_cx <= expected_digit_roi[2] and \
               expected_digit_roi[1] <= p_cy <= expected_digit_roi[3]:

                # 高亮这个匹配上的PaddleOCR框 (例如用蓝色)
                points = np.array(p_poly, dtype=np.int32)
                cv2.polylines(img_compare_viz, [points], isClosed=True, color=(255, 100, 0), thickness=3) # 蓝色表示匹配
                paddle_matched_to_yolo_indices.add(p_idx)
                yolo_match_count +=1
                found_match_for_this_yolo = True
                break # 一个YOLO框只匹配一个最佳的PaddleOCR框（或第一个满足条件的）

    # 绘制其他未匹配的PaddleOCR框 (保持绿色)
    for p_idx, p_data in enumerate(all_paddle_ocr_data):
        if p_idx not in paddle_matched_to_yolo_indices:
            p_poly = p_data['box']
            if p_poly is not None and isinstance(p_poly, (list, np.ndarray)) and len(p_poly) > 0 :
                try:
                    points = np.array(p_poly, dtype=np.int32)
                    cv2.polylines(img_compare_viz, [points], isClosed=True, color=(0, 200, 0), thickness=1) # 稍亮的绿色
                except: pass # 忽略绘制错误

    print(f"  YOLO与PaddleOCR对比：{yolo_match_count} 个YOLO锚点区域内找到了匹配的PaddleOCR文本检测框。")
    try:
        cv2.imwrite(output_path, img_compare_viz)
        print(f"  YOLO与PaddleOCR检测框对比可视化已保存到: {output_path}")
    except Exception as e:
        print(f"  保存YOLO与PaddleOCR对比可视化图片失败: {e}")


# --- 主程序 (V2.9.5) ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"--- OBU识别与矩阵输出工具 {VERSION} ---")
    print(f"输出目录: {os.path.abspath(CURRENT_RUN_OUTPUT_DIR)}")

    if not initialize_paddleocr(): exit()
    if not load_yolo_model(): print("警告: YOLO模型加载失败。")

    for image_path_current in IMAGE_PATHS:
        print(f"\n\n========== 处理图片: {image_path_current} ==========")
        img_filename_base = os.path.splitext(os.path.basename(image_path_current))[0]
        original_image = cv2.imread(image_path_current)
        if original_image is None: print(f"错误: 无法读取图片 {image_path_current}"); continue

        # --- PaddleOCR 识别 ---
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
                    all_paddle_ocr_data.append(item_data) # 存储所有原始检测结果
                    text_check = item_data['text'].strip()
                    if text_check.startswith(OBU_CODE_PREFIX_FILTER_CFG) and len(text_check) == OBU_CODE_LENGTH_FILTER_CFG and text_check.isdigit() and item_data['score'] >= PADDLE_OCR_FINE_PARAMS['text_rec_score_thresh']:
                        potential_obu_list_paddle.append(item_data) # 存储筛选后的OBU候选
        print(f"PaddleOCR 原始有效文本 {len(all_paddle_ocr_data)} 条, 内容筛选后潜在OBU {len(potential_obu_list_paddle)} 个。")

        # --- YOLO 检测 ---
        yolo_barcodes_list_xyxy = [] # 存储YOLO检测到的条码框 [x1,y1,x2,y2]
        if yolo_session_global:
            print(f"\n--- 步骤2: YOLO 条码锚点检测 ---")
            yolo_barcodes_list_xyxy, _ = get_yolo_barcode_anchors(original_image.copy())

        # --- 策略B：YOLO直接网格填充 ---
        if yolo_barcodes_list_xyxy: # 只有当YOLO检测到东西时才执行
            matrix_direct_yolo, filled_direct_yolo = build_matrix_yolo_direct_fill(yolo_barcodes_list_xyxy, potential_obu_list_paddle)
            print_matrix_to_console(matrix_direct_yolo, f"策略B (YOLO直接网格) - {img_filename_base}")
            # 可视化这个策略的结果 (可以画YOLO框和匹配上的文本)
            # ... (需要一个新的绘图函数 draw_yolo_direct_fill_matrix)
        else:
            print("YOLO未检测到条码，跳过YOLO直接网格填充策略。")

        # --- 对比分析与可视化 ---
        comparison_viz_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"compare_yolo_paddle_det_{img_filename_base}_{VERSION}.png")
        analyze_and_draw_yolo_paddle_comparison(original_image, yolo_barcodes_list_xyxy, all_paddle_ocr_data, comparison_viz_path)

        # 临时的PaddleOCR结果可视化 (方便查看原始识别)
        temp_paddle_viz_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"temp_paddle_raw_viz_{img_filename_base}_{VERSION}.png")
        if original_image is not None:
            # 创建一个只包含原始检测框（不含筛选后）的列表用于draw_ocr_results_refined的all_ocr_data参数
            raw_paddle_for_drawing = [{"text": item["text"], "score": item["score"], "box": item["box"]} for item in all_paddle_ocr_data]
            draw_ocr_results_refined(original_image, raw_paddle_for_drawing, potential_obu_list_paddle, temp_paddle_viz_path)


    overall_end_time = time.time(); total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。"); print(f"-------------------------------------------------")