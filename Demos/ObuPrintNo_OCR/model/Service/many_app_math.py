# D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\Service\app.py
import os
import cv2
import numpy as np
import onnxruntime
import time
import traceback
import paddlex as pdx
import re
import multiprocessing
from datetime import datetime
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import atexit
import logging
from logging.handlers import RotatingFileHandler
from collections import Counter
from scipy.spatial.distance import cdist
import uuid # For generating session IDs if needed

# --- 配置 ---
VERSION = "v2.5.6_flask_session_progressive_fill_complete"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LOG_DIR = "log"
LOG_FILE = "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_MODEL_PATH_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "BarCode_Detect", "BarCode_Detect_dynamic.onnx")
SERVER_REC_MODEL_DIR_CFG_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_infer")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MIN_DETECTION_AREA = 2000
MAX_DETECTION_AREA = 0.1

DIGIT_ROI_Y_OFFSET_FACTOR = -0.15
DIGIT_ROI_HEIGHT_FACTOR = 0.7
DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05
TARGET_OCR_INPUT_HEIGHT = 48
NUM_OCR_WORKERS_CONFIG = 4

PROCESS_PHOTO_DIR = "process_photo"
SAVE_PROCESS_PHOTOS = True
PROCESS_PHOTO_JPG_QUALITY = 85

LAYOUT_CONFIG = {
    "total_obus": 50, "regular_rows_count": 12, "regular_cols_count": 4,
    "special_row_cols_count": 2, "expected_total_rows": 13
}
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.75
MIN_YOLO_ANCHORS_FOR_LAYOUT = 10
COCO_CLASSES = ['Barcode']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

onnx_session = None
ocr_processing_pool = None
actual_num_ocr_workers = 1
session_data_store = {}

def setup_logging():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("Flask应用日志系统已启动。")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_box_center_and_dims(box_xyxy):
    if box_xyxy is None or len(box_xyxy) != 4: return None, None, None, None
    x1, y1, x2, y2 = box_xyxy; w = x2 - x1; h = y2 - y1
    return int(x1 + w / 2), int(y1 + h / 2), int(w), int(h)

def preprocess_onnx_for_main(img_data, target_shape_hw):
    img_h_orig, img_w_orig = img_data.shape[:2]; target_h, target_w = target_shape_hw
    ratio = min(target_w / img_w_orig, target_h / img_h_orig)
    new_w, new_h = int(img_w_orig * ratio), int(img_h_orig * ratio)
    resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), ratio, pad_x, pad_y

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold):
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]
    areas=(x2-x1+1e-6)*(y2-y1+1e-6); order=scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i); order = order[1:]
        if order.size == 0: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order])
        xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h
        ovr=inter/(areas[i]+areas[order]-inter+1e-6)
        order=order[np.where(ovr<=iou_threshold)[0]]
    return keep

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold,
                                   original_shape_hw, model_input_shape_hw,
                                   ratio_preproc, pad_x_preproc, pad_y_preproc,
                                   num_classes=1): # <--- 确保 num_classes 参数在这里
    logger = current_app.logger if current_app else logging.getLogger(__name__)
    raw_output_tensor = np.squeeze(outputs_onnx[0])
    if raw_output_tensor.ndim != 2:
        logger.error(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}")
        return []

    # YOLOv8通常输出是 [batch, num_predictions, 4_coords + 1_obj_conf + num_classes_conf]
    # 或者 [batch, 4_coords + 1_obj_conf + num_classes_conf, num_predictions] -> 转置后处理
    # 我们假设处理后是 num_predictions 行, (4+1+num_classes) 列 或类似结构
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor

    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []

    # 确定预期的属性数量，这取决于YOLO模型的输出格式
    # 常见的YOLOv5/v8格式是 cx,cy,w,h, objectness_confidence, class1_score, class2_score, ...
    # 如果是单类别且没有明确的objectness，可能是 cx,cy,w,h, class_confidence
    expected_attributes_with_obj = 4 + 1 + num_classes
    expected_attributes_simple_conf = 4 + 1

    actual_attributes = predictions_to_iterate.shape[1]

    for i_pred, pred_data in enumerate(predictions_to_iterate):
        box_coords_raw = pred_data[:4]
        final_confidence = 0.0
        class_id = 0 # Default for single class or if class determination fails

        if actual_attributes == expected_attributes_with_obj:
            objectness = float(pred_data[4])
            class_scores_all = pred_data[5:]
            if num_classes == 1:
                final_confidence = objectness * float(class_scores_all[0])
                # class_id remains 0
            else: # Multi-class scenario
                class_id = np.argmax(class_scores_all)
                max_class_score = float(class_scores_all[class_id])
                final_confidence = objectness * max_class_score
        elif actual_attributes == expected_attributes_simple_conf and num_classes == 1:
            # This case assumes the 5th element is directly the class confidence for the single class
            final_confidence = float(pred_data[4])
            # class_id remains 0
        else:
            if i_pred == 0: # Log warning only once per call if format is unexpected
                logger.warning(
                    f"YOLO postprocess: Unexpected number of attributes per prediction ({actual_attributes}). "
                    f"Expected {expected_attributes_simple_conf} (for simple conf) or {expected_attributes_with_obj} (with objectness). "
                    f"Attempting to use attribute 4 as confidence."
                )
            # Fallback: try to interpret the 5th element as confidence if other checks fail
            if actual_attributes >= 5:
                 final_confidence = float(pred_data[4])
            else:
                 logger.error(f"YOLO postprocess: Prediction data too short ({actual_attributes} attributes) to extract confidence.")
                 continue # Skip this prediction


        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = box_coords_raw
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2])
            scores_candidate.append(final_confidence)
            class_ids_candidate.append(class_id) # Store determined class_id

    if not boxes_candidate:
        return []

    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)

    final_detections = []
    orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx)
        box_model_coords = boxes_candidate[idx]
        score = scores_candidate[idx]
        class_id_val = class_ids_candidate[idx] # Use the stored class_id

        box_no_pad_x1 = box_model_coords[0] - pad_x_preproc
        box_no_pad_y1 = box_model_coords[1] - pad_y_preproc
        box_no_pad_x2 = box_model_coords[2] - pad_x_preproc
        box_no_pad_y2 = box_model_coords[3] - pad_y_preproc

        if ratio_preproc == 0: continue

        orig_x1 = box_no_pad_x1 / ratio_preproc
        orig_y1 = box_no_pad_y1 / ratio_preproc
        orig_x2 = box_no_pad_x2 / ratio_preproc
        orig_y2 = box_no_pad_y2 / ratio_preproc

        final_x1 = int(np.clip(orig_x1, 0, orig_w))
        final_y1 = int(np.clip(orig_y1, 0, orig_h))
        final_x2 = int(np.clip(orig_x2, 0, orig_w))
        final_y2 = int(np.clip(orig_y2, 0, orig_h))

        final_detections.append([final_x1, final_y1, final_x2, final_y2, score, class_id_val])

    return final_detections

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    img_out = image.copy() # Ensure all drawing happens on a copy
    # ... (rest of draw_detections from v2.5.5, ensure it uses img_out)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out

_worker_ocr_predictor = None
def init_ocr_worker(ocr_model_dir):
    global _worker_ocr_predictor; worker_pid = os.getpid()
    print(f"[Worker PID {worker_pid}] Initializing OCR predictor: {ocr_model_dir}")
    try:
        _worker_ocr_predictor = pdx.inference.create_predictor(model_dir=ocr_model_dir, model_name='PP-OCRv5_server_rec', device='cpu')
        print(f"[Worker PID {worker_pid}] OCR predictor initialized.")
    except Exception as e: print(f"[Worker PID {worker_pid}] OCR predictor init FAILED: {e}\n{traceback.format_exc()}"); _worker_ocr_predictor = None

def ocr_task_for_worker(task_data):
    global _worker_ocr_predictor; roi_idx, _, img_bgr = task_data; pid = os.getpid(); start_t = time.time()
    if img_bgr is None: return roi_idx, {'rec_text': 'PREPROC_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t}
    if _worker_ocr_predictor is None: return roi_idx, {'rec_text': 'WORKER_INIT_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t}
    try:
        res_gen = _worker_ocr_predictor.predict([img_bgr]); res_list = next(res_gen, None); final_res = {'rec_text': '', 'rec_score': 0.0}
        if res_list and isinstance(res_list, list) and len(res_list) > 0: final_res = res_list[0]
        elif res_list and isinstance(res_list, dict): final_res = res_list
        return roi_idx, {**final_res, 'pid': pid, 'duration': time.time() - start_t}
    except Exception as e: return roi_idx, {'rec_text': 'PREDICT_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t, 'error': str(e)}

def generate_ideal_layout_and_matrix(yolo_anchors_input, current_layout_config, image_wh, logger):
    logger.info("智能布局 V4.1: 基于可靠YOLO行构建局部确认网格 (修正变量名)...") # 版本号微调以便追踪
    if not yolo_anchors_input or len(yolo_anchors_input) < MIN_YOLO_ANCHORS_FOR_LAYOUT / 2:
        logger.warning(f"智能布局 V4.1: YOLO锚点数量 ({len(yolo_anchors_input)}) 过少。")
        return None, None

    # 1. YOLO锚点行分组
    yolo_anchors_sorted_by_y = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))
    yolo_rows_grouped = []
    avg_h_yolo_for_grouping = np.mean([a['h'] for a in yolo_anchors_sorted_by_y if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted_by_y) else 30
    y_threshold_for_grouping = avg_h_yolo_for_grouping * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR
    if not yolo_anchors_sorted_by_y:
        logger.warning("智能布局 V4.1: 无有效YOLO锚点进行行分组。")
        return None, None
    current_row_for_grouping = [yolo_anchors_sorted_by_y[0]]
    for i in range(1, len(yolo_anchors_sorted_by_y)):
        if abs(yolo_anchors_sorted_by_y[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold_for_grouping:
            current_row_for_grouping.append(yolo_anchors_sorted_by_y[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
            current_row_for_grouping = [yolo_anchors_sorted_by_y[i]]
    if current_row_for_grouping:
        yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
    logger.info(f"智能布局 V4.1: YOLO锚点行分组为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")
    if not yolo_rows_grouped:
        logger.warning("智能布局 V4.1: 行分组后结果为空。")
        return None, None

    # 2. 筛选可靠物理行
    reliable_physical_rows = [row for row in yolo_rows_grouped if len(row) >= 2]
    if not reliable_physical_rows:
        logger.warning("智能布局 V4.1: 未找到足够可靠的物理行。")
        return None, None
    logger.info(f"智能布局 V4.1: 筛选出 {len(reliable_physical_rows)} 个可靠物理行。")

    # 3. 推断常规列数和特殊行位置 (与V4相同)
    inferred_regular_cols = current_layout_config["regular_cols_count"]
    # ... (此处省略详细的 inferred_regular_cols 和 special_row_is_at_logical_top 判断逻辑，与V4版本一致)
    possible_regular_col_counts = [len(r) for r in reliable_physical_rows if len(r) != current_layout_config["special_row_cols_count"]]
    if possible_regular_col_counts:
        mode_res = Counter(possible_regular_col_counts).most_common(1)
        if mode_res and mode_res[0][0] > 0: inferred_regular_cols = mode_res[0][0]
    logger.info(f"智能布局 V4.1: 从可靠行推断常规列数为: {inferred_regular_cols}")
    physical_bottom_special_row_idx = -1
    if len(reliable_physical_rows[-1]) == current_layout_config["special_row_cols_count"]:
        if len(reliable_physical_rows) > 1 and abs(len(reliable_physical_rows[-2]) - inferred_regular_cols) <=1 :
            physical_bottom_special_row_idx = len(reliable_physical_rows) - 1
        elif len(reliable_physical_rows) == 1: physical_bottom_special_row_idx = 0
    if physical_bottom_special_row_idx != -1: logger.info(f"智能布局 V4.1: 物理底部检测到特殊行 (索引 {physical_bottom_special_row_idx} in reliable_physical_rows)。")


    # 4. 计算平均物理行高 (与V4相同)
    avg_physical_row_height = avg_h_yolo_for_grouping * 1.2
    if len(reliable_physical_rows) > 1:
        y_diffs = [reliable_physical_rows[i+1][0]['cy'] - reliable_physical_rows[i][0]['cy'] for i in range(len(reliable_physical_rows)-1)]
        if y_diffs: avg_physical_row_height = np.mean(y_diffs)
    if avg_physical_row_height <=0: avg_physical_row_height = avg_h_yolo_for_grouping * 1.2
    logger.info(f"智能布局 V4.1: 平均物理行高估算为: {avg_physical_row_height:.1f}")

    # 5. 生成理想坑位 (只为可靠物理行中的锚点生成，并赋予逻辑坐标)
    ideal_grid_slots = [] # <--- 确保初始化！！！
    # assigned_yolo_anchors_to_logical_slots = [] # 这个变量在V4中定义了但未使用，可以移除或按需使用

    expected_total_logical_rows = current_layout_config["expected_total_rows"]

    for i_physical_row, physical_row_anchors in enumerate(reliable_physical_rows):
        estimated_logical_row = -1
        if physical_bottom_special_row_idx != -1:
            logical_idx_of_bottom_special = expected_total_logical_rows - 1
            physical_row_diff_from_special = i_physical_row - physical_bottom_special_row_idx
            estimated_logical_row = logical_idx_of_bottom_special + physical_row_diff_from_special
        else:
            estimated_logical_row = (expected_total_logical_rows - len(reliable_physical_rows)) + i_physical_row
            if estimated_logical_row < 0 : estimated_logical_row = i_physical_row

        if not (0 <= estimated_logical_row < expected_total_logical_rows):
            logger.warning(f"智能布局 V4.1: 物理行 {i_physical_row} 估算的逻辑行 {estimated_logical_row} 超出范围，跳过。")
            continue

        is_this_logical_row_special = (estimated_logical_row == expected_total_logical_rows - 1)

        logical_col_indices_to_assign = []
        if is_this_logical_row_special and len(physical_row_anchors) == current_layout_config["special_row_cols_count"]:
            if inferred_regular_cols == 4 and current_layout_config["special_row_cols_count"] == 2:
                logical_col_indices_to_assign = [1, 2]
            else:
                logical_col_indices_to_assign = list(range(len(physical_row_anchors)))
        else:
            logical_col_indices_to_assign = list(range(len(physical_row_anchors)))

        for i_anchor_in_row, yolo_anchor in enumerate(physical_row_anchors):
            if i_anchor_in_row < len(logical_col_indices_to_assign):
                logical_col = logical_col_indices_to_assign[i_anchor_in_row]

                ideal_grid_slots.append({ # <--- 使用 ideal_grid_slots.append
                    'logical_row': estimated_logical_row,
                    'logical_col': logical_col,
                    'cx': int(yolo_anchor['cx']),
                    'cy': int(yolo_anchor['cy']),
                    'w': int(yolo_anchor['w']),
                    'h': int(yolo_anchor['h'])
                })
                # assigned_yolo_anchors_to_logical_slots.append({**yolo_anchor, 'logical_row': estimated_logical_row, 'logical_col': logical_col})

    if not ideal_grid_slots: # <--- 使用 ideal_grid_slots
        logger.warning("智能布局 V4.1: 未能从可靠行生成任何理想坑位。")
        return None, None
    logger.info(f"智能布局 V4.1: 基于可靠行共生成 {len(ideal_grid_slots)} 个高置信度理想坑位。") # <--- 使用 ideal_grid_slots

    # 6. 构建初始矩阵
    initial_obu_matrix = [["无检测"] * inferred_regular_cols for _ in range(expected_total_logical_rows)]
    for slot in ideal_grid_slots: # <--- 使用 ideal_grid_slots
        r, c = slot['logical_row'], slot['logical_col']
        if 0 <= r < expected_total_logical_rows and 0 <= c < inferred_regular_cols:
            initial_obu_matrix[r][c] = "未识别"
            if r == expected_total_logical_rows - 1 and \
               current_layout_config["special_row_cols_count"] == 2 and \
               inferred_regular_cols == 4:
                if c == 0 or c == 3:
                    initial_obu_matrix[r][c] = "---"

    return ideal_grid_slots, initial_obu_matrix

def fill_matrix_incrementally(matrix, slots, paddle_res, processed_texts, logger):
    logger.info("增量填充矩阵...")
    filled_new = 0
    if not paddle_res or not slots: return matrix, filled_new
    slot_coords = np.array([[s['cx'],s['cy']] for s in slots])
    paddle_coords = np.array([[p['cx'],p['cy']] for p in paddle_res if p.get('cx') is not None])
    if paddle_coords.size == 0: return matrix, filled_new
    dist_m = cdist(slot_coords, paddle_coords)
    avg_w_slot = np.mean([s['w'] for s in slots if s.get('w',0)>0]) or 100

    for j_paddle, p_obu in enumerate(paddle_res):
        if p_obu.get('cx') is None or p_obu['text'] in processed_texts: continue
        best_slot_i = -1; min_d = float('inf')
        for i_slot, s_info in enumerate(slots):
            if j_paddle < dist_m.shape[1] and i_slot < dist_m.shape[0]:
                curr_d = dist_m[i_slot, j_paddle]
                max_d_thr = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * s_info.get('w', avg_w_slot)
                if curr_d < max_d_thr and curr_d < min_d: min_d=curr_d; best_slot_i=i_slot
        if best_slot_i != -1:
            s_fill = slots[best_slot_i]; r,c = s_fill['logical_row'],s_fill['logical_col']
            if r<len(matrix) and c<len(matrix[r]) and (matrix[r][c]=="未识别" or not matrix[r][c].startswith("5001")):
                matrix[r][c] = p_obu['text']; processed_texts.add(p_obu['text']); filled_new+=1
                logger.info(f"OBU '{p_obu['text']}' 填入矩阵 [{r},{c}]")
    logger.info(f"本轮新填充 {filled_new} 个OBU")
    return matrix, filled_new

def draw_obu_matrix_on_image(obu_matrix, layout_cfg, logger): # From v2.5.5
    # ... (Full function body from previous correct version) ...
    if not obu_matrix: logger.warning("矩阵数据为空，无法生成逻辑矩阵图。"); return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)
    num_r = layout_cfg.get("expected_total_rows", len(obu_matrix)); num_c = layout_cfg.get("regular_cols_count", len(obu_matrix[0]) if obu_matrix else 0)
    if num_r == 0 or num_c == 0: logger.warning("矩阵维度为0，无法生成逻辑矩阵图。"); return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)
    cell_sz=60; pad=15; space=10; txt_off_y=-5
    img_w = num_c*cell_sz+(num_c-1)*space+2*pad; img_h = num_r*cell_sz+(num_r-1)*space+2*pad
    canvas = np.full((img_h, img_w, 3), (255,255,255), dtype=np.uint8)
    c_ok=(0,180,0); c_fail=(0,0,200); c_txt_ok=(255,255,255); c_txt_fail=(255,255,255); c_empty=(220,220,220)
    f_scale=0.7; f_thick=2
    for r_idx in range(num_r):
        actual_cols_r = num_c; special_cfg_cols = layout_cfg.get("special_row_cols_count", num_c)
        if r_idx < len(obu_matrix):
            actual_cols_data = len(obu_matrix[r_idx])
            if actual_cols_data != num_c and actual_cols_data == special_cfg_cols: actual_cols_r = actual_cols_data
            elif actual_cols_data != num_c : logger.warning(f"逻辑图:行{r_idx}数据列{actual_cols_data}与常规{num_c}或特殊{special_cfg_cols}不符"); actual_cols_r = actual_cols_data
        for c_idx in range(num_c):
            cx_start=pad+c_idx*(cell_sz+space); cy_start=pad+r_idx*(cell_sz+space)
            cen_x=cx_start+cell_sz//2; cen_y=cy_start+cell_sz//2
            cell_color=c_empty; disp_txt=""; txt_color=(0,0,0)
            if c_idx < actual_cols_r and r_idx < len(obu_matrix):
                txt_slot = obu_matrix[r_idx][c_idx]
                if isinstance(txt_slot, str):
                    if txt_slot=="未识别" or txt_slot.startswith("锚点不足") or txt_slot.startswith("无YOLO锚点") or txt_slot.startswith("无理想坑位"): disp_txt="X"; cell_color=c_fail; txt_color=c_txt_fail
                    elif txt_slot.startswith("5001") and len(txt_slot)>4: disp_txt=txt_slot[-4:]; cell_color=c_ok; txt_color=c_txt_ok
                    else: disp_txt=txt_slot[:4]; cell_color=c_fail; txt_color=c_txt_fail
                else: disp_txt="N/A"; cell_color=c_fail; txt_color=c_txt_fail
            else: disp_txt="-"
            cv2.rectangle(canvas,(cx_start,cy_start),(cx_start+cell_sz,cy_start+cell_sz),cell_color,-1)
            cv2.rectangle(canvas,(cx_start,cy_start),(cx_start+cell_sz,cy_start+cell_sz),(50,50,50),1)
            if disp_txt:
                (tw,th),_ = cv2.getTextSize(disp_txt,cv2.FONT_HERSHEY_SIMPLEX,f_scale,f_thick)
                tx=cen_x-tw//2; ty=cen_y+th//2+txt_off_y
                cv2.putText(canvas,disp_txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,f_scale,txt_color,f_thick,cv2.LINE_AA)
    logger.info("逻辑矩阵可视化图绘制完成。")
    return canvas

def process_image_with_ocr_logic(image_path, current_onnx_session, session_id, current_layout_config, min_area_cfg, max_area_cfg):
    logger = current_app.logger
    logger.info(f"会话 {session_id}: 处理图片 {os.path.basename(image_path)} (版本 {VERSION})")
    timing_profile = {}
    t_start_overall_processing = time.time()

    # 1. Read Image
    t_start_img_read = time.time(); original_image = cv2.imread(image_path); timing_profile['1_image_reading'] = time.time() - t_start_img_read
    if original_image is None: logger.error(f"错误: 无法读取图片: {image_path}"); raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img_h, orig_img_w = original_image.shape[:2]; logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    # 2. YOLO Detection & Area Filtering
    actual_max_area_threshold_px = None
    if max_area_cfg is not None:
        if isinstance(max_area_cfg, float) and 0 < max_area_cfg <= 1.0: actual_max_area_threshold_px = (orig_img_h * orig_img_w) * max_area_cfg
        elif isinstance(max_area_cfg, (int, float)) and max_area_cfg > 1: actual_max_area_threshold_px = float(max_area_cfg)

    logger.info("--- 开始整图检测 (YOLO) ---")
    input_cfg = current_onnx_session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640)
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int): model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]

    t_s = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s;
    t_s = time.time(); outputs_main = current_onnx_session.run(None, {input_name: input_tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s;
    detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(COCO_CLASSES)); timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s

    aggregated_boxes_xyxy = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]
    aggregated_scores = [d[4] for d in detections_result_list]
    aggregated_class_ids = [d[5] for d in detections_result_list]
    logger.info(f"YOLO检测完成。找到了 {len(aggregated_boxes_xyxy)} 个原始框。")

    if len(aggregated_boxes_xyxy) > 0 and ((min_area_cfg is not None and min_area_cfg > 0) or actual_max_area_threshold_px is not None):
        t_start_area_filter=time.time(); filtered_boxes,filtered_scores,filtered_ids=[],[],[]; initial_count=len(aggregated_boxes_xyxy)
        for i_box,box_xyxy in enumerate(aggregated_boxes_xyxy):
            b_w,b_h=box_xyxy[2]-box_xyxy[0],box_xyxy[3]-box_xyxy[1]; area=b_w*b_h; valid=True
            if min_area_cfg is not None and min_area_cfg > 0 and area < min_area_cfg: valid=False
            if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid=False
            if valid: filtered_boxes.append(box_xyxy); filtered_scores.append(aggregated_scores[i_box]); filtered_ids.append(aggregated_class_ids[i_box])
        aggregated_boxes_xyxy,aggregated_scores,aggregated_class_ids=filtered_boxes,filtered_scores,filtered_ids; timing_profile['5_area_filtering']=time.time()-t_start_area_filter; logger.info(f"面积筛选后剩 {len(aggregated_boxes_xyxy)} 个框.")
    else: timing_profile['5_area_filtering']=0

    # 3. OCR Preprocessing & Task Preparation
    tasks_for_ocr = []
    # This intermediate list helps map OCR results back to original YOLO detections if needed for drawing annotated image
    ocr_input_metadata = [None] * len(aggregated_boxes_xyxy)

    if len(aggregated_boxes_xyxy) > 0:
        logger.info(f"--- 对 {len(aggregated_boxes_xyxy)} 个YOLO框进行OCR预处理 ---")
        for i, yolo_box_coords in enumerate(aggregated_boxes_xyxy):
            class_id = int(aggregated_class_ids[i]); class_name = COCO_CLASSES[class_id]
            x1_y, y1_y, x2_y, y2_y = yolo_box_coords; h_y, w_y = y2_y-y1_y, x2_y-x1_y
            y1_d_ideal = y1_y + int(h_y * DIGIT_ROI_Y_OFFSET_FACTOR); h_d_ideal = int(h_y * DIGIT_ROI_HEIGHT_FACTOR)
            y2_d_ideal = y1_d_ideal + h_d_ideal
            w_d_exp = int(w_y * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_y = x1_y + w_y / 2.0
            x1_d_ideal = int(cx_y - w_d_exp / 2.0); x2_d_ideal = int(cx_y + w_d_exp / 2.0)
            y1_d_clip,y2_d_clip = max(0,y1_d_ideal),min(orig_img_h,y2_d_ideal)
            x1_d_clip,x2_d_clip = max(0,x1_d_ideal),min(orig_img_w,x2_d_ideal)

            ocr_input_metadata[i] = {"original_index": i, "roi_index": i + 1, "class": class_name,
                                     "bbox_yolo": yolo_box_coords,
                                     "bbox_digit_ocr_clipped": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip],
                                     "confidence_yolo": float(aggregated_scores[i])}
            img_for_ocr = None
            if x2_d_clip > x1_d_clip and y2_d_clip > y1_d_clip:
                digit_roi = original_image[y1_d_clip:y2_d_clip, x1_d_clip:x2_d_clip]
                h_roi, w_roi = digit_roi.shape[:2]
                if h_roi > 0 and w_roi > 0:
                    scale = TARGET_OCR_INPUT_HEIGHT / h_roi; target_w = int(w_roi * scale)
                    if target_w <= 0: target_w = 1
                    resized_roi = cv2.resize(digit_roi, (target_w, TARGET_OCR_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
                    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img_for_ocr = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
                                # NEW: Save OCR input slice if enabled
            if SAVE_PROCESS_PHOTOS and img_for_ocr is not None:
                ocr_slice_dir = os.path.join(PROCESS_PHOTO_DIR, "ocr_slices")
                if not os.path.exists(ocr_slice_dir):
                    os.makedirs(ocr_slice_dir, exist_ok=True)

                # Try to get a tentative text for filename, otherwise use index
                # This part is tricky as actual OCR text is not yet available
                # We can use original_index (i) from the loop
                slice_filename = f"s_{session_id}_idx{i}_roi{i+1}.jpg"
                slice_output_path = os.path.join(ocr_slice_dir, slice_filename)
                try:
                    cv2.imwrite(slice_output_path, img_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY + 5 if PROCESS_PHOTO_JPG_QUALITY <=90 else 95]) # Slightly higher quality for small slices
                    # logger.debug(f"OCR切片图已保存: {slice_output_path}") # Use debug level for less verbose logs
                except Exception as e_save_slice:
                    logger.error(f"保存OCR切片图失败 {slice_output_path}: {e_save_slice}")

            tasks_for_ocr.append((i, i + 1, img_for_ocr))

    # 4. Parallel OCR Processing & Result Consolidation
    final_ocr_results_list = [None] * len(aggregated_boxes_xyxy)
    ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes_xyxy)

    if tasks_for_ocr:
        logger.info(f"提交 {len(tasks_for_ocr)} 个OCR任务...")
        t_ocr_s = time.time()
        ocr_results_indexed_from_pool = [None] * len(tasks_for_ocr)
        global ocr_processing_pool, actual_num_ocr_workers
        if actual_num_ocr_workers > 1 and ocr_processing_pool:
            try:
                pool_res = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                for orig_idx_res, res_dict_ocr in pool_res: ocr_results_indexed_from_pool[orig_idx_res] = res_dict_ocr
            except Exception as e_map: logger.error(f"OCR Pool map error: {e_map}"); # Consider serial fallback
        else: # Serial
            logger.info("OCR串行处理")
            # ... (Serial OCR logic from v2.5.5, populating ocr_results_indexed_from_pool)
            serial_ocr_predictor = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG): raise FileNotFoundError("Serial OCR model dir not found")
                serial_ocr_predictor = pdx.inference.create_predictor(model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG, model_name='PP-OCRv5_server_rec', device='cpu')
                for task_idx_s, task_data_s in enumerate(tasks_for_ocr):
                    orig_idx_s, _, img_data_s = task_data_s
                    if img_data_s is not None:
                        res_gen_s = serial_ocr_predictor.predict([img_data_s]); res_list_s = next(res_gen_s, None)
                        if res_list_s and isinstance(res_list_s, list) and len(res_list_s) > 0: ocr_results_indexed_from_pool[orig_idx_s] = res_list_s[0]
                        elif res_list_s and isinstance(res_list_s, dict): ocr_results_indexed_from_pool[orig_idx_s] = res_list_s
                        else: ocr_results_indexed_from_pool[orig_idx_s] = {'rec_text': '', 'rec_score': 0.0}
                    else: ocr_results_indexed_from_pool[orig_idx_s] = {'rec_text': 'PREPROC_FAIL_SERIAL', 'rec_score': 0.0}
                if serial_ocr_predictor: del serial_ocr_predictor
            except Exception as e_serial_ocr: logger.error(f"Serial OCR error: {e_serial_ocr}")


        timing_profile['7_ocr_processing_total'] = time.time() - t_ocr_s
        logger.info(f"OCR处理完成 ({timing_profile['7_ocr_processing_total']:.3f}s)")

        for i, ocr_dict in enumerate(ocr_results_indexed_from_pool):
            full_res_item = {**(ocr_input_metadata[i] or {})} # Start with YOLO info
            if ocr_dict and isinstance(ocr_dict, dict):
                raw_txt = ocr_dict.get('rec_text', "")
                score = ocr_dict.get('rec_score', 0.0)
                if raw_txt and raw_txt not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_SERIAL']:
                    digits = "".join(re.findall(r'\d', raw_txt))
                    full_res_item["ocr_final_text"] = digits if digits else "N/A_NO_DIGITS"
                    ocr_texts_for_drawing[i] = digits if digits else "ERR"
                else:
                    full_res_item["ocr_final_text"] = raw_txt # Store error code
                    ocr_texts_for_drawing[i] = "ERR"
                full_res_item["ocr_confidence"] = score
            else:
                full_res_item["ocr_final_text"] = "N/A_OCR_FAIL"
                ocr_texts_for_drawing[i] = "N/A"
                full_res_item["ocr_confidence"] = 0.0
            final_ocr_results_list[i] = full_res_item
    else:
        logger.info("无任务进行OCR"); timing_profile['7_ocr_processing_total'] = 0
        # Populate final_ocr_results_list with N/A if no OCR but boxes existed
        for i in range(len(aggregated_boxes_xyxy)):
            final_ocr_results_list[i] = {**(ocr_input_metadata[i] or {}), "ocr_final_text": "N/A_NO_OCR_TASKS", "ocr_confidence": 0.0}


    # 5. 会话处理与矩阵更新
    session = session_data_store.get(session_id)
    obu_matrix_to_return = None

    yolo_anchors_for_layout_gen = []
    for i, box_xyxy in enumerate(aggregated_boxes_xyxy):
        cx, cy, w, h = get_box_center_and_dims(box_xyxy)
        if cx is not None:
            yolo_anchors_for_layout_gen.append({'cx': cx, 'cy': cy, 'w': w, 'h': h,
                                                'box_yolo': box_xyxy, 'score': aggregated_scores[i]})

    paddle_results_for_matrix_fill = []
    for ocr_item in final_ocr_results_list:
        if ocr_item and isinstance(ocr_item.get("ocr_final_text"), str) and \
           ocr_item.get("ocr_final_text") not in ["N/A", "N/A_NO_DIGITS", "N/A_NO_OCR_TASKS", "N/A_OCR_FAIL", "WORKER_INIT_FAIL", "PREDICT_FAIL", "PREPROC_FAIL", "PREPROC_FAIL_SERIAL"]:
            ocr_cx, ocr_cy, ocr_w, ocr_h = get_box_center_and_dims(ocr_item.get('bbox_digit_ocr_clipped'))
            if ocr_cx is not None:
                paddle_results_for_matrix_fill.append({
                    'text': ocr_item['ocr_final_text'], 'score': ocr_item.get('ocr_confidence', 0.0),
                    'cx': ocr_cx, 'cy': ocr_cy, 'w': ocr_w, 'h': ocr_h,
                    'box_ocr_clipped': ocr_item.get('bbox_digit_ocr_clipped')})

    ideal_grid_slots_for_session = None # To store/retrieve ideal slots for the session

    if not session:
        logger.info(f"会话 {session_id}: 新建会话。")
        ideal_slots, initial_matrix = generate_ideal_layout_and_matrix(
            yolo_anchors_input=yolo_anchors_for_layout_gen,
            current_layout_config=current_layout_config,
            image_wh=(orig_img_w, orig_img_h), logger=logger)
        if ideal_slots and initial_matrix:
            ideal_grid_slots_for_session = ideal_slots
            session_data_store[session_id] = {
                "layout_config": current_layout_config, "ideal_grid_slots": ideal_slots,
                "obu_matrix": initial_matrix, "processed_obu_texts": set(),
                "last_activity": datetime.now()}
            session = session_data_store[session_id]
            updated_matrix, _ = fill_matrix_incrementally(
                session["obu_matrix"], session["ideal_grid_slots"],
                paddle_results_for_matrix_fill, session["processed_obu_texts"], logger)
            session["obu_matrix"] = updated_matrix
            obu_matrix_to_return = session["obu_matrix"]
        else:
            logger.error(f"会话 {session_id}: 无法从首图生成布局。")
            obu_matrix_to_return = [["布局失败"]*current_layout_config["regular_cols_count"] for _ in range(current_layout_config["expected_total_rows"])]
    else:
        logger.info(f"会话 {session_id}: 加载现有会话数据。")
        session["last_activity"] = datetime.now()
        ideal_grid_slots_for_session = session["ideal_grid_slots"] # Use stored slots
        updated_matrix, _ = fill_matrix_incrementally(
            session["obu_matrix"], session["ideal_grid_slots"],
            paddle_results_for_matrix_fill, session["processed_obu_texts"], logger)
        session["obu_matrix"] = updated_matrix
        obu_matrix_to_return = session["obu_matrix"]

    timing_profile['9_matrix_session_logic'] = time.time() - (timing_profile.get('7_ocr_processing_total', 0) + \
        timing_profile.get('5_area_filtering',0) + timing_profile.get('3c_fullimg_postprocessing',0) + \
        timing_profile.get('3b_fullimg_inference',0) + timing_profile.get('3a_fullimg_preprocessing',0) + \
        timing_profile.get('1_image_reading',0) + t_start_overall_processing) # More accurate start point for this step

    # 6. & 7. Visualizations
    if SAVE_PROCESS_PHOTOS:
        if len(aggregated_boxes_xyxy) > 0:
            annotated_img = draw_detections(original_image.copy(), np.array(aggregated_boxes_xyxy),
                                            np.array(aggregated_scores), np.array(aggregated_class_ids),
                                            COCO_CLASSES, ocr_texts=ocr_texts_for_drawing,
                                            roi_indices=[item['roi_index'] for item in final_ocr_results_list if item]) # Use actual roi_indices
            # ... (save annotated_img to PROCESS_PHOTO_DIR/annotated_...jpg)
            img_name_base = os.path.splitext(os.path.basename(image_path))[0]
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            annotated_path = os.path.join(PROCESS_PHOTO_DIR, f"annotated_{img_name_base}_{ts}.jpg")
            try: cv2.imwrite(annotated_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY]); logger.info(f"Annotated image saved: {annotated_path}")
            except Exception as e_save_ann: logger.error(f"Failed to save annotated image: {e_save_ann}")
            timing_profile['8_drawing_annotated'] = time.time() - (t_start_overall_processing + sum(timing_profile.get(k,0) for k in ['1_image_reading', '3a_fullimg_preprocessing', '3b_fullimg_inference', '3c_fullimg_postprocessing', '5_area_filtering', '7_ocr_processing_total', '9_matrix_session_logic']))


        if obu_matrix_to_return and ideal_grid_slots_for_session: # Use the slots from the session
            t_matrix_draw_s = time.time()
            matrix_viz = draw_obu_matrix_on_image(obu_matrix_to_return, current_layout_config, logger) # Pass correct layout
            # ... (save matrix_viz to PROCESS_PHOTO_DIR/matrix_viz_...jpg)
            matrix_viz_path = os.path.join(PROCESS_PHOTO_DIR, f"matrix_viz_{img_name_base}_{ts}.jpg")
            try: cv2.imwrite(matrix_viz_path, matrix_viz, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY]); logger.info(f"Matrix viz image saved: {matrix_viz_path}")
            except Exception as e_save_matrix: logger.error(f"Failed to save matrix viz: {e_save_matrix}")
            timing_profile['10_matrix_visualization_drawing'] = time.time() - t_matrix_draw_s
        elif SAVE_PROCESS_PHOTOS:
            logger.warning("Matrix or ideal slots not available for matrix visualization.")

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    logger.info(f"--- Timing profile for {os.path.basename(image_path)} ({session_id}) ---")
    for stage, duration in sorted(timing_profile.items()): logger.info(f"  {stage}: {duration:.3f}s")

    return final_ocr_results_list, obu_matrix_to_return, timing_profile

# --- Flask Routes (修改 /predict 以接收 session_id) ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger

    # 1. 获取 session_id (强制要求客户端提供)
    if 'session_id' not in request.form:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400
    session_id = request.form.get('session_id')

    current_layout_config_for_session = LAYOUT_CONFIG

    # 2. 文件检查和保存
    if 'file' not in request.files:
        logger.warning(f"会话 {session_id}: 请求中未找到文件部分。")
        return jsonify({"error": "No file part in the request", "session_id": session_id}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning(f"会话 {session_id}: 未选择文件。")
        return jsonify({"error": "No selected file", "session_id": session_id}), 400

    if not (file and allowed_file(file.filename)):
        logger.warning(f"会话 {session_id}: 文件类型不允许: {file.filename}")
        return jsonify({"error": "File type not allowed", "session_id": session_id}), 400

    original_filename_for_exc = "N/A" # For use in except block if original_filename not set
    try:
        original_filename = secure_filename(file.filename)
        original_filename_for_exc = original_filename # Set it once available
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{timestamp}{ext}"

        upload_dir = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        logger.info(f"会话 {session_id}: 文件 '{filename}' 已成功保存到 '{filepath}'")

        global onnx_session
        if onnx_session is None:
            logger.error(f"会话 {session_id}: ONNX session 未初始化!")
            return jsonify({"error": "ONNX session not initialized on server", "session_id": session_id}), 500

        # 从 app.config 获取面积筛选参数
        min_area_cfg_val = current_app.config.get('MIN_DETECTION_AREA_CFG', 2000)
        max_area_cfg_val = current_app.config.get('MAX_DETECTION_AREA_CFG', 0.1)

        # 调用核心处理逻辑
        ocr_results_flat_list, obu_matrix_result, timings = process_image_with_ocr_logic(
            filepath,
            onnx_session,
            session_id,
            current_layout_config_for_session,
            min_area_cfg_val, # <--- 参数在这里传递
            max_area_cfg_val  # <--- 参数在这里传递
        )

        response_data = {
            "message": "File processed successfully.",
            "session_id": session_id,
            "received_filename": original_filename,
            "obu_matrix": obu_matrix_result,
            "timing_profile_seconds": timings,
        }

        current_session_state = session_data_store.get(session_id)
        if current_session_state and \
           len(current_session_state.get("processed_obu_texts", set())) >= current_session_state.get("layout_config", {}).get("total_obus", float('inf')):
            response_data["session_status"] = "completed"
            logger.info(f"会话 {session_id}: 所有OBU已识别，会话完成。")
        else:
            response_data["session_status"] = "in_progress"

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_exc}' 时发生严重错误: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- Main Application Runner (与v2.5.4相同) ---
# ... (initialize_onnx_session, cleanup_global_ocr_pool, if __name__ == '__main__': block) ...
def initialize_onnx_session(): # Same as v2.5.4
    global onnx_session; logger = app.logger if app else logging.getLogger(__name__)
    logger.info("--- ONNX 模型初始化开始 ---")
    if not os.path.exists(ONNX_MODEL_PATH_CONFIG): logger.error(f"错误: ONNX模型未找到: {ONNX_MODEL_PATH_CONFIG}"); raise FileNotFoundError(f"ONNX Model not found: {ONNX_MODEL_PATH_CONFIG}")
    try:
        onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH_CONFIG, providers=['CPUExecutionProvider'])
        logger.info(f"ONNX模型加载完成 from {ONNX_MODEL_PATH_CONFIG}")
    except Exception as e: logger.error(f"ONNX模型加载失败: {e}\n{traceback.format_exc()}"); raise
    logger.info("--- ONNX 模型初始化完成 ---")

def cleanup_global_ocr_pool(): # Same as v2.5.4
    global ocr_processing_pool
    if ocr_processing_pool: print("Closing global OCR processing pool..."); ocr_processing_pool.close(); ocr_processing_pool.join(); print("Global OCR processing pool closed.")

if __name__ == '__main__': # Same as v2.5.4, with setup_logging()
    setup_logging()
        # --- Load module-level configurations into app.config ---
    # Do this after app object is created and before it's run.
    # The module-level constants MIN_DETECTION_AREA and MAX_DETECTION_AREA
    # are defined at the top of the script.
    app.config['MIN_DETECTION_AREA_CFG'] = MIN_DETECTION_AREA
    app.config['MAX_DETECTION_AREA_CFG'] = MAX_DETECTION_AREA
    # Also, let's ensure other relevant configs are loaded if needed by routes directly
    # For now, only these two are directly used by the route via process_image_with_ocr_logic's args
    # LAYOUT_CONFIG is used by build_obu_matrix_smart, which is called by process_image_with_ocr_logic
    # So, LAYOUT_CONFIG doesn't strictly need to be in app.config unless a route handler itself needs it.
    # However, it's good practice for app-wide constants.
    app.config['LAYOUT_CONFIG_CFG'] = LAYOUT_CONFIG
    # (If other top-level constants were needed by routes, add them here too)

    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn': multiprocessing.set_start_method('spawn', force=True)
        app.logger.info(f"Multiprocessing start method set to: {multiprocessing.get_start_method()}")
    except RuntimeError as e_mp_start: app.logger.warning(f"Could not set multiprocessing start method ('spawn'): {e_mp_start}. Using default: {multiprocessing.get_start_method(allow_none=True)}")

    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(PROCESS_PHOTO_DIR): os.makedirs(PROCESS_PHOTO_DIR, exist_ok=True)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR, exist_ok=True)

    cpu_cores = os.cpu_count() or 1
    if NUM_OCR_WORKERS_CONFIG <= 0: actual_num_ocr_workers = 1
    elif NUM_OCR_WORKERS_CONFIG > cpu_cores: actual_num_ocr_workers = cpu_cores
    else: actual_num_ocr_workers = NUM_OCR_WORKERS_CONFIG

    initialize_onnx_session()

    if actual_num_ocr_workers > 1:
        if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG):
            app.logger.critical(f"Server OCR model directory for workers not found: {SERVER_REC_MODEL_DIR_CFG_CONFIG}"); actual_num_ocr_workers = 1
        else:
            app.logger.info(f"Initializing global OCR processing pool with {actual_num_ocr_workers} workers...")
            try:
                ocr_processing_pool = multiprocessing.Pool(processes=actual_num_ocr_workers, initializer=init_ocr_worker, initargs=(SERVER_REC_MODEL_DIR_CFG_CONFIG,) )
                app.logger.info("Global OCR processing pool initialized and workers are loading models.")
                atexit.register(cleanup_global_ocr_pool)
            except Exception as e_pool_create: app.logger.critical(f"Failed to create global OCR processing pool: {e_pool_create}\n{traceback.format_exc()}"); ocr_processing_pool = None; actual_num_ocr_workers = 1

    if actual_num_ocr_workers <= 1 and ocr_processing_pool is None : app.logger.info("OCR will run in serial mode.")

    app.logger.info(f"服务版本 {VERSION} 启动中... 监听 0.0.0.0:5000")
    app.logger.info(f"过程图片保存开关 (SAVE_PROCESS_PHOTOS): {SAVE_PROCESS_PHOTOS}")
    if SAVE_PROCESS_PHOTOS: app.logger.info(f"过程图片JPG质量 (PROCESS_PHOTO_JPG_QUALITY): {PROCESS_PHOTO_JPG_QUALITY}")
    app.logger.info(f"智能矩阵布局配置: {LAYOUT_CONFIG}")


    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)