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
from collections import Counter # For build_matrix_smart_grid
from scipy.spatial.distance import cdist # For build_matrix_smart_grid

# --- 配置 ---
# ================== 版本信息 ==================
VERSION = "v2.5.5_flask_smart_matrix_integration"

# ... (UPLOAD_FOLDER, ALLOWED_EXTENSIONS, LOG_DIR, LOG_FILE, etc. from v2.5.4 remain the same) ...
# ================== 服务端上传配置 ==================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ================== 日志配置 ==================
LOG_DIR = "log"
LOG_FILE = "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5

# ================== 模型路径配置 ==================
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_MODEL_PATH_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "BarCode_Detect", "BarCode_Detect_dynamic.onnx")
SERVER_REC_MODEL_DIR_CFG_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_infer")

# ================== YOLOv8 检测参数 ==================
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ================== 检测结果面积筛选配置 (确保这些在这里定义！) ==================
MIN_DETECTION_AREA = 2000
MAX_DETECTION_AREA = 0.1

# ================== 数字ROI裁剪精调参数 (用于OCR前处理) ==================
DIGIT_ROI_Y_OFFSET_FACTOR = -0.15 # 向上偏移YOLO框高度的百分比
DIGIT_ROI_HEIGHT_FACTOR = 0.7    # 数字区域占YOLO框高度的百分比
DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05 # 数字区域宽度在YOLO框宽度基础上的扩展因子

# ================== OCR 服务端识别配置 ==================
TARGET_OCR_INPUT_HEIGHT = 48

# ================== 并行OCR处理配置 ==================
NUM_OCR_WORKERS_CONFIG = 4

# ================== 过程图片保存配置 ==================
PROCESS_PHOTO_DIR = "process_photo_service"
SAVE_PROCESS_PHOTOS = True
PROCESS_PHOTO_JPG_QUALITY = 85

# ================== 智能矩阵布局配置 (来自您的 V2.9.x 脚本) ==================
LAYOUT_CONFIG = {
    "total_obus": 50,
    "regular_rows_count": 12, # 逻辑上的常规行数 (例如12行4列)
    "regular_cols_count": 4,  # 逻辑上的常规列数
    "special_row_cols_count": 2, # 特殊行的列数 (例如底部一行2列)
    "expected_total_rows": 13 # 期望的总逻辑行数 (regular_rows_count + 1 if special row exists)
}
# --- 智能矩阵算法相关阈值 (来自您的 V2.9.x 脚本) ---
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5  # YOLO锚点聚类成行时，Y坐标差异阈值因子（乘以平均锚点高度）
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.75 # OCR结果中心点与理想坑位中心点匹配的最大允许距离因子（乘以理想坑位宽度）
MIN_YOLO_ANCHORS_FOR_LAYOUT = 10 # 进行布局推断所需的最少YOLO锚点数

# ================== 其他配置 ==================
COCO_CLASSES = ['Barcode'] # 假设YOLO模型检测的是'Barcode'类别

# --- Flask 应用实例 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 全局模型实例 ---
onnx_session = None
ocr_processing_pool = None
actual_num_ocr_workers = 1

# --- 日志设置函数 (与v2.5.4相同) ---
def setup_logging():
    # ... (logging setup code from v2.5.4) ...
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("Flask应用日志系统已启动。")

# --- 辅助函数 (部分来自v2.5.4, 新增 get_box_center_and_dims) ---
# ... (allowed_file, clear_process_photo_directory, YOLO pre/post-processing, draw_detections from v2.5.4) ...
# --- BEGIN: Functions copied/adapted ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_process_photo_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            except Exception as e:
                logger = current_app.logger if current_app else logging.getLogger(__name__)
                logger.error(f'Failed to delete {file_path}. Reason: {e}')
    else: os.makedirs(directory, exist_ok=True)

def preprocess_onnx_for_main(img_data, target_shape_hw): # Adapted from previous
    img_height_orig, img_width_orig = img_data.shape[:2]
    target_h, target_w = target_shape_hw
    ratio = min(target_w / img_width_orig, target_h / img_height_orig)
    new_w, new_h = int(img_width_orig * ratio), int(img_height_orig * ratio)
    resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    input_tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, ratio, pad_x, pad_y

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold): # From previous
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

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold, \
                                   original_shape_hw, model_input_shape_hw, \
                                   ratio_preproc, pad_x_preproc, pad_y_preproc, \
                                   num_classes=1): # Adapted from previous
    logger = current_app.logger if current_app else logging.getLogger(__name__)
    raw_output_tensor = np.squeeze(outputs_onnx[0]);
    if raw_output_tensor.ndim != 2: logger.error(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []
    expected_attributes = 5 # For YOLO models with [cx, cy, w, h, conf] per class or [cx,cy,w,h,obj_conf, class_conf_0, ...]
    # This needs to be robust to different YOLO output formats.
    # Assuming single class 'Barcode' and output is [cx,cy,w,h,conf] or [cx,cy,w,h,obj_conf, barcode_conf]
    if predictions_to_iterate.shape[1] == (4 + 1 + num_classes) and num_classes >=1 : # e.g. [cx,cy,w,h, obj_conf, class1_conf, ...]
         expected_attributes = 4 + 1 + num_classes
    elif predictions_to_iterate.shape[1] != 5 and num_classes == 1 : # Fallback for [cx,cy,w,h,conf]
        logger.warning(f"YOLO output attributes {predictions_to_iterate.shape[1]} vs expected 5 or {4+1+num_classes}. Assuming 5 (cx,cy,w,h,conf).")
        expected_attributes = 5


    for i_pred, pred_data in enumerate(predictions_to_iterate):
        if len(pred_data) < 5 : continue # Must have at least cx,cy,w,h,conf

        box_coords_raw = pred_data[:4]
        final_confidence = 0.0
        class_id = 0 # Default to 0 for single class 'Barcode'

        if expected_attributes == 5: # Simple [cx,cy,w,h,conf]
            final_confidence = float(pred_data[4])
        elif expected_attributes == (4 + 1 + num_classes): # [cx,cy,w,h, obj_conf, class1_conf, ...]
            objectness = float(pred_data[4])
            class_scores_all = pred_data[5:]
            if num_classes == 1:
                final_confidence = objectness * float(class_scores_all[0])
            else: # Multi-class (though we only care about 'Barcode')
                # Assuming COCO_CLASSES[0] is 'Barcode'
                # This part might need adjustment if your YOLO model is multi-class and 'Barcode' isn't class 0
                barcode_class_index = 0 # Assuming 'Barcode' is the first and only class we care about
                final_confidence = objectness * float(class_scores_all[barcode_class_index])
                class_id = barcode_class_index # or np.argmax(class_scores_all) if truly multi-class
        else: # Fallback or unexpected format
            if i_pred == 0: logger.warning(f"Unexpected YOLO pred_data length: {len(pred_data)}. Trying to use pred_data[4] as confidence.")
            final_confidence = float(pred_data[4])


        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = box_coords_raw; x1,y1,x2,y2 = cx-w/2,cy-h/2,cx+w/2,cy+h/2
            boxes_candidate.append([x1,y1,x2,y2]); scores_candidate.append(final_confidence); class_ids_candidate.append(class_id)

    if not boxes_candidate: return []
    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)
    final_detections = []; orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx); box_model_coords = boxes_candidate[idx]; score = scores_candidate[idx]; class_id_val = class_ids_candidate[idx]
        box_no_pad_x1,box_no_pad_y1 = box_model_coords[0]-pad_x_preproc,box_model_coords[1]-pad_y_preproc
        box_no_pad_x2,box_no_pad_y2 = box_model_coords[2]-pad_x_preproc,box_model_coords[3]-pad_y_preproc
        if ratio_preproc == 0: continue
        orig_x1,orig_y1 = box_no_pad_x1/ratio_preproc,box_no_pad_y1/ratio_preproc; orig_x2,orig_y2 = box_no_pad_x2/ratio_preproc,box_no_pad_y2/ratio_preproc
        final_x1,final_y1 = np.clip(orig_x1,0,orig_w),np.clip(orig_y1,0,orig_h); final_x2,final_y2 = np.clip(orig_x2,0,orig_w),np.clip(orig_y2,0,orig_h)
        final_detections.append([int(final_x1),int(final_y1),int(final_x2),int(final_y2),score,class_id_val])
    return final_detections

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None): # From previous
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out

# --- (确保此函数放在其他辅助函数定义区域) ---
# --- (确保此函数放在其他辅助函数定义区域) ---
def draw_obu_matrix_on_image(obu_matrix_data, layout_config, logger):
    """
    生成一个纯色背景的逻辑矩阵图，用方块和文本清晰展示OBU识别状态。
    Args:
        obu_matrix_data (list): 二维列表，服务端生成的OBU矩阵。
        layout_config (dict): 布局配置，用于获取矩阵维度。
        logger: Flask app logger instance.
    Returns:
        OpenCV Image: 绘制了逻辑矩阵图的新图像。
    """
    if not obu_matrix_data:
        logger.warning("矩阵数据为空，无法生成逻辑矩阵图。")
        # 返回一个小的空白图像或错误提示图
        return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)

    num_rows = layout_config.get("expected_total_rows", len(obu_matrix_data))
    # 以常规列数为基准宽度，特殊行会在其内部少画单元格
    num_cols = layout_config.get("regular_cols_count", len(obu_matrix_data[0]) if obu_matrix_data else 0)

    if num_rows == 0 or num_cols == 0:
        logger.warning("矩阵维度为0，无法生成逻辑矩阵图。")
        return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)

    cell_size = 60  # 每个单元格的像素大小
    padding = 15    # 画布边缘和单元格之间的间距
    spacing = 10    # 单元格之间的间距
    text_offset_y = -5 # 文本相对于单元格中心的Y偏移

    img_width = num_cols * cell_size + (num_cols - 1) * spacing + 2 * padding
    img_height = num_rows * cell_size + (num_rows - 1) * spacing + 2 * padding

    # 创建纯白色背景画布
    matrix_canvas = np.full((img_height, img_width, 3), (255, 255, 255), dtype=np.uint8)

    color_identified_success = (0, 180, 0)   # 深绿色 - 识别成功
    color_identified_fail = (0, 0, 200)     # 深红色 - 未识别/错误
    color_text_success = (255, 255, 255) # 白色文字 - 成功背景上
    color_text_fail = (255, 255, 255)    # 白色文字 - 失败背景上
    color_empty_slot = (220, 220, 220)  # 浅灰色 - 逻辑上存在但无对应OBU的格子（例如特殊行的空位）

    font_scale = 0.7
    font_thickness = 2

    for r in range(num_rows):
        # 判断当前行是否为特殊行，以及它实际应有的列数
        actual_cols_this_row = num_cols # 默认为常规列数
        is_special_row_config = layout_config.get("special_row_cols_count", num_cols)

        # 一个简化的判断：如果当前行在obu_matrix_data中的列数少于num_cols，则认为是特殊行处理
        # 或者可以根据layout_config更精确判断（例如，首行或末行）
        # 为简化，我们直接使用obu_matrix_data[r]的长度
        if r < len(obu_matrix_data):
            actual_cols_this_row_from_data = len(obu_matrix_data[r])
            # 如果数据中的列数与常规列数不同，且等于特殊行配置的列数，则按特殊行处理
            if actual_cols_this_row_from_data != num_cols and actual_cols_this_row_from_data == is_special_row_config:
                 actual_cols_this_row = actual_cols_this_row_from_data
            elif actual_cols_this_row_from_data != num_cols: # 数据列数与常规不同，但不等于特殊配置，可能数据有误
                 logger.warning(f"逻辑矩阵图: 第 {r} 行数据列数 {actual_cols_this_row_from_data} 与常规列数 {num_cols} 或特殊行列数 {is_special_row_config} 不符。按数据列数绘制。")
                 actual_cols_this_row = actual_cols_this_row_from_data


        for c in range(num_cols): # 我们仍然遍历所有可能的常规列，以便绘制空槽
            cell_x_start = padding + c * (cell_size + spacing)
            cell_y_start = padding + r * (cell_size + spacing)

            center_x = cell_x_start + cell_size // 2
            center_y = cell_y_start + cell_size // 2

            current_cell_color = color_empty_slot
            display_text = ""
            current_text_color = (0,0,0) # Default black for empty slot text if any

            if c < actual_cols_this_row and r < len(obu_matrix_data): # 确保在有效数据范围内
                obu_text_in_slot = obu_matrix_data[r][c]
                if isinstance(obu_text_in_slot, str):
                    if obu_text_in_slot == "未识别" or \
                       obu_text_in_slot.startswith("锚点不足") or \
                       obu_text_in_slot.startswith("无YOLO锚点") or \
                       obu_text_in_slot.startswith("无理想坑位"):
                        display_text = "X"
                        current_cell_color = color_identified_fail
                        current_text_color = color_text_fail
                    elif obu_text_in_slot.startswith("5001") and len(obu_text_in_slot) > 4:
                        display_text = obu_text_in_slot[-4:]
                        current_cell_color = color_identified_success
                        current_text_color = color_text_success
                    else: # Other short texts or unexpected values
                        display_text = obu_text_in_slot[:4]
                        current_cell_color = color_identified_fail # Treat as fail if not standard success
                        current_text_color = color_text_fail
                else: # Not a string
                    display_text = "N/A"
                    current_cell_color = color_identified_fail
                    current_text_color = color_text_fail
            else: # This is a slot that's part of regular grid but empty in a special row
                display_text = "-" # Indicate an empty logical slot

            # 绘制单元格方块
            cv2.rectangle(matrix_canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          current_cell_color, -1) # Filled rectangle
            cv2.rectangle(matrix_canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          (50,50,50), 1) # Black border for cell

            # 绘制文本
            if display_text:
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2 + text_offset_y
                cv2.putText(matrix_canvas, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, current_text_color, font_thickness, cv2.LINE_AA)

    logger.info("逻辑矩阵可视化图绘制完成。")
    return matrix_canvas

def get_box_center_and_dims(box_xyxy): # Adapted from your V2.9.x
    """Calculates center (cx, cy), width (w), and height (h) from an [x1, y1, x2, y2] box."""
    if box_xyxy is None or len(box_xyxy) != 4:
        return None, None, None, None
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return int(cx), int(cy), int(w), int(h)
# --- END: Functions copied/adapted ---


# --- OCR Worker Process Initialization and Task Function (与v2.5.4相同) ---
_worker_ocr_predictor = None
def init_ocr_worker(ocr_model_dir):
    # ... (init_ocr_worker code from v2.5.4) ...
    global _worker_ocr_predictor; worker_pid = os.getpid()
    print(f"[Worker PID {worker_pid}] Initializing OCR predictor with model_dir: {ocr_model_dir}")
    try:
        _worker_ocr_predictor = pdx.inference.create_predictor(model_dir=ocr_model_dir, model_name='PP-OCRv5_server_rec', device='cpu')
        print(f"[Worker PID {worker_pid}] OCR predictor initialized successfully.")
    except Exception as e:
        print(f"[Worker PID {worker_pid}] CRITICAL: Failed to initialize OCR predictor: {e}\n{traceback.format_exc()}"); _worker_ocr_predictor = None

def ocr_task_for_worker(task_data):
    # ... (ocr_task_for_worker code from v2.5.4) ...
    global _worker_ocr_predictor; roi_original_index, roi_display_index, image_bgr_for_ocr = task_data
    worker_pid = os.getpid(); start_time = time.time()
    if image_bgr_for_ocr is None: return roi_original_index, {'rec_text': 'PREPROC_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': time.time() - start_time}
    if _worker_ocr_predictor is None: print(f"[Worker PID {worker_pid}] OCR predictor not available. ROI {roi_display_index} cannot be processed."); return roi_original_index, {'rec_text': 'WORKER_INIT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': time.time() - start_time}
    try:
        result_gen = _worker_ocr_predictor.predict([image_bgr_for_ocr]); recognition_result_list_for_roi = next(result_gen, None)
        final_result_dict = {'rec_text': '', 'rec_score': 0.0}
        if recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, list) and len(recognition_result_list_for_roi) > 0: final_result_dict = recognition_result_list_for_roi[0]
        elif recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, dict): final_result_dict = recognition_result_list_for_roi
        duration = time.time() - start_time
        return roi_original_index, {**final_result_dict, 'pid': worker_pid, 'duration': duration}
    except Exception as e:
        duration = time.time() - start_time; print(f"[Worker PID {worker_pid} | ROI {roi_display_index}] Error during OCR predict: {e}")
        return roi_original_index, {'rec_text': 'PREDICT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': duration}


# --- 智能矩阵构建函数 (移植并适配自您的 V2.9.x build_matrix_smart_grid) ---
def build_obu_matrix_smart(yolo_anchors_input, paddle_results_input, layout_config, image_wh, logger):
    """
    通过YOLO锚点、布局先验，精确推断理想网格，并用PaddleOCR结果填充。
    Args:
        yolo_anchors_input (list): YOLO锚点列表 [{'cx', 'cy', 'w', 'h', 'box_yolo'}, ...]
        paddle_results_input (list): PaddleOCR筛选后的OBU结果列表 [{'text', 'score', 'cx', 'cy', 'w', 'h', 'box_ocr_clipped'}, ...]
        layout_config (dict): 包含布局先验的配置
        image_wh (tuple): 原始图片的宽度和高度 (w, h)
        logger: Flask app logger instance
    Returns:
        list: 二维矩阵, int: 填充的OBU数量
    """
    logger.info("开始执行智能网格矩阵构建...")
    if not yolo_anchors_input or len(yolo_anchors_input) < MIN_YOLO_ANCHORS_FOR_LAYOUT:
        logger.warning(f"YOLO锚点数量 ({len(yolo_anchors_input)}) 不足 ({MIN_YOLO_ANCHORS_FOR_LAYOUT}个)，无法进行可靠布局推断。返回空矩阵。")
        return [["锚点不足"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0

    # 1. YOLO锚点行分组 (与您脚本逻辑类似)
    yolo_anchors_sorted_by_y = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))
    yolo_rows_grouped = []
    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors_sorted_by_y if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted_by_y) else 30 # Default height
    y_threshold = avg_h_yolo * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    if not yolo_anchors_sorted_by_y:
        logger.warning("无有效YOLO锚点进行行分组。")
        return [["无YOLO锚点"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0

    current_row_for_grouping = [yolo_anchors_sorted_by_y[0]]
    for i in range(1, len(yolo_anchors_sorted_by_y)):
        # Compare with the average Y of the current row being built, or simply the last element's Y
        # For simplicity, using last element's Y as in your script's initial grouping
        if abs(yolo_anchors_sorted_by_y[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold:
            current_row_for_grouping.append(yolo_anchors_sorted_by_y[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx'])) # Sort by X within the row
            current_row_for_grouping = [yolo_anchors_sorted_by_y[i]]
    if current_row_for_grouping: # Add the last row
        yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))

    logger.info(f"YOLO锚点精确行分组为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")

    # 2. 识别特殊行和推断常规列数 (与您脚本逻辑类似)
    special_row_is_at_top = None # True if top, False if bottom, None if undetermined
    num_detected_yolo_rows = len(yolo_rows_grouped)
    inferred_regular_cols = layout_config["regular_cols_count"] # Default

    if num_detected_yolo_rows > 0:
        # Try to infer regular columns from rows that are NOT the special_row_cols_count
        possible_regular_col_counts = [len(r) for r in yolo_rows_grouped if len(r) != layout_config["special_row_cols_count"]]
        if not possible_regular_col_counts: # If all rows have special_row_cols_count or no other counts
            possible_regular_col_counts = [len(r) for r in yolo_rows_grouped] # Use all row counts

        if possible_regular_col_counts:
            mode_res = Counter(possible_regular_col_counts).most_common(1)
            if mode_res and mode_res[0][0] > 0: inferred_regular_cols = mode_res[0][0]

        # Identify special row (simplified logic from your script)
        if len(yolo_rows_grouped[0]) == layout_config["special_row_cols_count"] and \
           (num_detected_yolo_rows == 1 or (num_detected_yolo_rows > 1 and abs(len(yolo_rows_grouped[1]) - inferred_regular_cols) <=1 )): # Allow 1 diff for next row
            special_row_is_at_top = True
            logger.info("智能矩阵: 初步判断特殊行在顶部。")
        elif num_detected_yolo_rows > 1 and \
             len(yolo_rows_grouped[-1]) == layout_config["special_row_cols_count"] and \
             abs(len(yolo_rows_grouped[-2]) - inferred_regular_cols) <=1 : # Allow 1 diff for prev row
            special_row_is_at_top = False
            logger.info("智能矩阵: 初步判断特殊行在底部。")
        else:
            logger.warning("智能矩阵: 未能明确判断特殊行位置。默认特殊行在底部（如果存在）。")
            # Default assumption if not clearly identified: special row is at the bottom if total rows match expectation
            if num_detected_yolo_rows == layout_config["expected_total_rows"]:
                 if len(yolo_rows_grouped[-1]) == layout_config["special_row_cols_count"]:
                      special_row_is_at_top = False
                 elif len(yolo_rows_grouped[0]) == layout_config["special_row_cols_count"]: # Check top again if bottom doesn't fit
                      special_row_is_at_top = True
                 else: # Default to bottom if still unclear
                      special_row_is_at_top = False
            else: # If row counts don't match, it's harder. Default to bottom.
                 special_row_is_at_top = False


    logger.info(f"智能矩阵: 推断常规列数: {inferred_regular_cols}, 特殊行是否在顶部: {special_row_is_at_top}")

    # 3. 生成理想坑位坐标 (简化版几何估算，类似您脚本中的直接生成逻辑)
    ideal_grid_slots = []
    avg_obu_w_yolo = np.mean([a['w'] for a in yolo_anchors_input if a.get('w',0)>0]) if any(a.get('w',0)>0 for a in yolo_anchors_input) else 100
    avg_obu_h_yolo = np.mean([a['h'] for a in yolo_anchors_input if a.get('h',0)>0]) if any(a.get('h',0)>0 for a in yolo_anchors_input) else 40

    # Estimate Y coordinates for each logical row based on grouped YOLO rows
    logical_row_y_coords = [0.0] * layout_config["expected_total_rows"]
    if yolo_rows_grouped:
        for i in range(min(len(yolo_rows_grouped), layout_config["expected_total_rows"])):
            logical_row_y_coords[i] = np.mean([a['cy'] for a in yolo_rows_grouped[i]])
        # Extrapolate if not enough YOLO rows detected
        if len(yolo_rows_grouped) < layout_config["expected_total_rows"] and len(yolo_rows_grouped) > 0:
            last_known_y = logical_row_y_coords[len(yolo_rows_grouped)-1]
            for i in range(len(yolo_rows_grouped), layout_config["expected_total_rows"]):
                logical_row_y_coords[i] = last_known_y + (avg_obu_h_yolo + 10) * (i - (len(yolo_rows_grouped) - 1)) # Simple extrapolation
    else: # Fallback if no YOLO rows
        for i in range(layout_config["expected_total_rows"]):
            logical_row_y_coords[i] = (image_wh[0] / layout_config["expected_total_rows"] / 2) + i * (image_wh[0] / layout_config["expected_total_rows"])


    # Estimate X coordinates for each logical column in each row
    # Using a reference row (e.g., the longest detected YOLO row) to set the X scale
    reference_yolo_row_for_x = []
    if yolo_rows_grouped:
        reference_yolo_row_for_x = max(yolo_rows_grouped, key=len, default=None) # Longest row

    start_x_overall = image_wh[1] * 0.1 # Default start X
    col_spacing_overall = avg_obu_w_yolo * 1.1 # Default spacing

    if reference_yolo_row_for_x and len(reference_yolo_row_for_x) >= inferred_regular_cols -1 and inferred_regular_cols > 1: # Need at least a few anchors in ref row
        xs_in_ref_row = [a['cx'] for a in reference_yolo_row_for_x]
        start_x_overall = min(xs_in_ref_row)
        if len(xs_in_ref_row) > 1:
            col_spacing_overall = np.mean(np.diff(sorted(xs_in_ref_row)))
            if col_spacing_overall <= avg_obu_w_yolo * 0.5 : col_spacing_overall = avg_obu_w_yolo * 1.1 # Sanity check
    elif yolo_anchors_input: # Fallback to overall min X if ref row is not good
        all_cx = [a['cx'] for a in yolo_anchors_input]
        if all_cx: start_x_overall = min(all_cx)


    current_obu_count = 0
    for r_logic in range(layout_config["expected_total_rows"]):
        cols_for_this_row = layout_config["regular_cols_count"]
        is_this_row_special = False

        # Determine if this logical row is the special row
        if special_row_is_at_top is True and r_logic == 0:
            cols_for_this_row = layout_config["special_row_cols_count"]
            is_this_row_special = True
        elif special_row_is_at_top is False and r_logic == (layout_config["expected_total_rows"] - 1):
            cols_for_this_row = layout_config["special_row_cols_count"]
            is_this_row_special = True

        current_y = logical_row_y_coords[r_logic]

        # Adjust starting X for special rows to center them relative to regular rows
        x_offset_for_centering = 0
        if is_this_row_special and cols_for_this_row < inferred_regular_cols:
            x_offset_for_centering = (inferred_regular_cols - cols_for_this_row) * col_spacing_overall / 2.0

        for c_logic in range(cols_for_this_row):
            if current_obu_count >= layout_config["total_obus"]: break
            current_x = start_x_overall + x_offset_for_centering + c_logic * col_spacing_overall
            ideal_grid_slots.append({
                'logical_row': r_logic, 'logical_col': c_logic,
                'cx': int(current_x), 'cy': int(current_y),
                'w': int(avg_obu_w_yolo), 'h': int(avg_obu_h_yolo)
            })
            current_obu_count += 1
        if current_obu_count >= layout_config["total_obus"]: break

    if not ideal_grid_slots:
        logger.warning("未能生成理想坑位坐标。")
        return [["无理想坑位"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0
    logger.info(f"已生成 {len(ideal_grid_slots)} 个理想坑位坐标。")

    # 4. 将PaddleOCR识别结果填充到理想坑位 (与您脚本逻辑类似)
    final_matrix = [["未识别"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])]
    matrix_filled_count = 0

    if paddle_results_input and ideal_grid_slots:
        # Prepare coordinates for cdist
        ideal_coords_np = np.array([[slot['cx'], slot['cy']] for slot in ideal_grid_slots])
        # paddle_results_input already has cx, cy
        paddle_coords_np = np.array([[p['cx'], p['cy']] for p in paddle_results_input if p.get('cx') is not None])

        if paddle_coords_np.size == 0:
            logger.warning("没有有效的PaddleOCR中心点用于匹配。")
        else:
            dist_matrix = cdist(ideal_coords_np, paddle_coords_np)

            # Keep track of used paddle results to avoid double matching
            paddle_used_flags = [False] * len(paddle_results_input)

            for i_slot, slot_info in enumerate(ideal_grid_slots):
                log_r, log_c = slot_info['logical_row'], slot_info['logical_col']

                # Ensure logical_col is valid for potentially shorter special rows in the matrix
                if log_r >= len(final_matrix) or log_c >= len(final_matrix[log_r]): # Check against actual row length
                    logger.warning(f"逻辑行列 ({log_r},{log_c}) 超出最终矩阵边界 ({len(final_matrix)}x{len(final_matrix[log_r]) if log_r < len(final_matrix) else 'N/A'}). 跳过此理想坑位。")
                    continue

                best_paddle_idx = -1
                min_dist_to_slot = float('inf')
                # Max distance threshold based on ideal slot width (or avg YOLO anchor width as fallback)
                max_dist_thresh = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * slot_info.get('w', avg_obu_w_yolo)

                for j_paddle, p_obu in enumerate(paddle_results_input):
                    if paddle_used_flags[j_paddle]: continue # Skip already used OCR results
                    if p_obu.get('cx') is None: continue # Skip OCR results without coordinates

                    if i_slot < dist_matrix.shape[0] and j_paddle < dist_matrix.shape[1]:
                        current_dist = dist_matrix[i_slot, j_paddle]
                        if current_dist < max_dist_thresh and current_dist < min_dist_to_slot:
                            min_dist_to_slot = current_dist
                            best_paddle_idx = j_paddle

                if best_paddle_idx != -1:
                    final_matrix[log_r][log_c] = paddle_results_input[best_paddle_idx]['text']
                    paddle_used_flags[best_paddle_idx] = True
                    matrix_filled_count += 1

    logger.info(f"智能网格方案: 构建矩阵 {len(final_matrix)}x{len(final_matrix[0]) if final_matrix else 0}, 填充OBU数: {matrix_filled_count}")
    return final_matrix, matrix_filled_count, ideal_grid_slots


# --- Main Image Processing Function ---
def process_image_with_ocr_logic(image_path, input_onnx_session, min_area_cfg, max_area_cfg):
    logger = current_app.logger
    logger.info(f"--- OBU 检测、识别与矩阵构建开始 ({VERSION}) for image: {image_path} ---")
    timing_profile = {}
    t_start_overall_processing = time.time()

    # 1. Read Image (same as before)
    # ... (image reading logic) ...
    t_start_img_read = time.time(); original_image = cv2.imread(image_path); timing_profile['1_image_reading'] = time.time() - t_start_img_read
    if original_image is None: logger.error(f"错误: 无法读取图片: {image_path}"); raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img_h, orig_img_w = original_image.shape[:2]; logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    # 2. YOLO Detection (same as before)
    # ... (YOLO detection logic, results in aggregated_boxes, aggregated_scores, aggregated_class_ids) ...
    actual_max_area_threshold_px = None
    if max_area_cfg is not None: # 使用参数 max_area_cfg
        if isinstance(max_area_cfg, float) and 0 < max_area_cfg <= 1.0:
            actual_max_area_threshold_px = (orig_img_h * orig_img_w) * max_area_cfg
        elif isinstance(max_area_cfg, (int, float)) and max_area_cfg > 1:
            actual_max_area_threshold_px = float(max_area_cfg)
    logger.info("--- 开始整图检测 (YOLO) ---")
    input_cfg = input_onnx_session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640)
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int): model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]
    t_s = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s;
    t_s = time.time(); outputs_main = input_onnx_session.run(None, {input_name: input_tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s;
    detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(COCO_CLASSES)); timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s
    aggregated_boxes_xyxy = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]; aggregated_scores = [d[4] for d in detections_result_list]; aggregated_class_ids = [d[5] for d in detections_result_list]; logger.info(f"YOLO检测完成。找到了 {len(aggregated_boxes_xyxy)} 个原始框。")

    # Area Filtering (same as before, but on aggregated_boxes_xyxy)
    # ... (area filtering logic, updates aggregated_boxes_xyxy, aggregated_scores, aggregated_class_ids) ...
    if len(aggregated_boxes_xyxy) > 0 and ((min_area_cfg is not None and min_area_cfg > 0) or actual_max_area_threshold_px is not None): # 使用参数 min_area_cfg
        t_start_area_filter=time.time(); filtered_boxes,filtered_scores,filtered_ids=[],[],[]; initial_count=len(aggregated_boxes_xyxy)
        for i_box,box_xyxy in enumerate(aggregated_boxes_xyxy):
            b_w_filter,b_h_filter=box_xyxy[2]-box_xyxy[0],box_xyxy[3]-box_xyxy[1]; area_filter=b_w_filter*b_h_filter; valid_area_filter=True
            if min_area_cfg is not None and min_area_cfg > 0 and area_filter < min_area_cfg: valid_area_filter=False # 使用参数 min_area_cfg
            if actual_max_area_threshold_px is not None and area_filter > actual_max_area_threshold_px: valid_area_filter=False
            if valid_area_filter: filtered_boxes.append(box_xyxy); filtered_scores.append(aggregated_scores[i_box]); filtered_ids.append(aggregated_class_ids[i_box])
        aggregated_boxes_xyxy,aggregated_scores,aggregated_class_ids=filtered_boxes,filtered_scores,filtered_ids; timing_profile['5_area_filtering']=time.time()-t_start_area_filter; logger.info(f"面积筛选完成 (从 {initial_count} 减少到 {len(aggregated_boxes_xyxy)} 个框).")
    else: timing_profile['5_area_filtering']=0


    # 3. OCR Preprocessing and Task Preparation (same as before)
    # ... (OCR task prep, results in tasks_for_ocr and recognized_obu_data_list_intermediate) ...
    tasks_for_ocr = []
    recognized_obu_data_list_intermediate = [None] * len(aggregated_boxes_xyxy) # Based on filtered YOLO boxes
    if len(aggregated_boxes_xyxy) > 0:
        logger.info(f"--- 对 {len(aggregated_boxes_xyxy)} 个YOLO框进行OCR预处理 ---")
        for i, yolo_box_coords in enumerate(aggregated_boxes_xyxy): # Use filtered boxes
            # ... (ROI extraction for OCR using DIGIT_ROI factors, same as v2.5.4) ...
            class_id = int(aggregated_class_ids[i]); class_name_str = COCO_CLASSES[class_id] if COCO_CLASSES and 0 <= class_id < len(COCO_CLASSES) else f"ClassID:{class_id}"
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = [int(c) for c in yolo_box_coords]; h_yolo = y2_yolo - y1_yolo; w_yolo = x2_yolo - x1_yolo
            y1_digit_ideal = y1_yolo + int(h_yolo * DIGIT_ROI_Y_OFFSET_FACTOR); h_digit_ideal = int(h_yolo * DIGIT_ROI_HEIGHT_FACTOR); y2_digit_ideal = y1_digit_ideal + h_digit_ideal
            w_digit_expanded = int(w_yolo * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_yolo = x1_yolo + w_yolo / 2.0; x1_digit_ideal = int(cx_yolo - w_digit_expanded / 2.0); x2_digit_ideal = int(cx_yolo + w_digit_expanded / 2.0)
            y1_d_clip = max(0, y1_digit_ideal); y2_d_clip = min(orig_img_h, y2_digit_ideal); x1_d_clip = max(0, x1_digit_ideal); x2_d_clip = min(orig_img_w, x2_digit_ideal)
            current_box_meta_for_task = {"original_index": i, "roi_index": i + 1, "class": class_name_str, "bbox_yolo": [x1_yolo, y1_yolo, x2_yolo, y2_yolo], "bbox_digit_ocr_ideal": [x1_digit_ideal, y1_digit_ideal, x2_digit_ideal, y2_digit_ideal], "bbox_digit_ocr_clipped": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip], "confidence_yolo": float(aggregated_scores[i])}
            recognized_obu_data_list_intermediate[i] = current_box_meta_for_task
            image_for_ocr_bgr = None; dx1,dy1,dx2,dy2 = current_box_meta_for_task['bbox_digit_ocr_clipped']
            if dx2>dx1 and dy2>dy1:
                digit_roi_color=original_image[dy1:dy2,dx1:dx2]; h_roi_digit, w_roi_digit = digit_roi_color.shape[:2]
                if h_roi_digit > 0 and w_roi_digit > 0:
                    scale_ocr = TARGET_OCR_INPUT_HEIGHT / h_roi_digit; target_w_ocr = int(w_roi_digit * scale_ocr)
                    if target_w_ocr <= 0: target_w_ocr = 1
                    resized_digit_roi_color = cv2.resize(digit_roi_color, (target_w_ocr, TARGET_OCR_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC if scale_ocr > 1 else cv2.INTER_AREA)
                    gray_resized_roi = cv2.cvtColor(resized_digit_roi_color, cv2.COLOR_BGR2GRAY); _, binary_resized_roi = cv2.threshold(gray_resized_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    image_for_ocr_bgr = cv2.cvtColor(binary_resized_roi, cv2.COLOR_GRAY2BGR)
            tasks_for_ocr.append( (i, i + 1, image_for_ocr_bgr) )

    # 4. Parallel OCR Processing (same as before)
    # ... (OCR processing, results in ocr_processed_results_indexed) ...
    ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes_xyxy)
    final_ocr_results_list = [None] * len(aggregated_boxes_xyxy) # This will be the flat list of OCR results
    global ocr_processing_pool, actual_num_ocr_workers
    if tasks_for_ocr:
        # ... (OCR task submission and result consolidation logic from v2.5.4, populating final_ocr_results_list) ...
        logger.info(f"\n--- Submitting {len(tasks_for_ocr)} OCR tasks to pool/serial ---"); t_ocr_start = time.time()
        ocr_processed_results_indexed = [None] * len(tasks_for_ocr)
        if actual_num_ocr_workers > 1 and ocr_processing_pool is not None:
            try:
                parallel_results_with_indices = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                for res_original_idx, res_dict in parallel_results_with_indices: ocr_processed_results_indexed[res_original_idx] = res_dict
            except Exception as e_pool_map: logger.error(f"Error during ocr_processing_pool.map: {e_pool_map}\n{traceback.format_exc()}");
        run_serially = False
        if actual_num_ocr_workers <= 1 or ocr_processing_pool is None: run_serially = True
        if run_serially: # Serial OCR logic from v2.5.4
            logger.info("Performing OCR in serial mode (main process)."); serial_ocr_predictor = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG): raise FileNotFoundError(f"Serial OCR: Model directory not found {SERVER_REC_MODEL_DIR_CFG_CONFIG}")
                serial_ocr_predictor = pdx.inference.create_predictor(model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG, model_name='PP-OCRv5_server_rec', device='cpu')
            except Exception as e_serial_init_ocr: logger.error(f"Error initializing serial OCR predictor: {e_serial_init_ocr}")
            if serial_ocr_predictor:
                for task_idx, task_data_tuple in enumerate(tasks_for_ocr):
                    original_idx, _, img_data = task_data_tuple
                    if img_data is not None:
                        res_gen = serial_ocr_predictor.predict([img_data]); res_list = next(res_gen, None)
                        if res_list and isinstance(res_list, list) and len(res_list) > 0: ocr_processed_results_indexed[original_idx] = res_list[0]
                        elif res_list and isinstance(res_list, dict): ocr_processed_results_indexed[original_idx] = res_list
                        else: ocr_processed_results_indexed[original_idx] = {'rec_text': '', 'rec_score': 0.0}
                    else: ocr_processed_results_indexed[original_idx] = {'rec_text': 'PREPROC_FAIL_MAIN_SERIAL', 'rec_score': 0.0}
                del serial_ocr_predictor
            else:
                for task_idx, task_data_tuple in enumerate(tasks_for_ocr):
                     original_idx, _, _, = task_data_tuple; ocr_processed_results_indexed[original_idx] = {'rec_text': 'SERIAL_OCR_INIT_FAIL', 'rec_score': 0.0}
        timing_profile['7_ocr_processing_total'] = time.time() - t_ocr_start; logger.info(f"--- OCR processing finished ({timing_profile['7_ocr_processing_total']:.3f} 秒) ---")
        for original_idx, recognition_result_dict in enumerate(ocr_processed_results_indexed): # Result consolidation from v2.5.4
            current_box_info = recognized_obu_data_list_intermediate[original_idx]; ocr_text_to_draw = "N/A"
            if recognition_result_dict and isinstance(recognition_result_dict, dict):
                raw_recognized_text = recognition_result_dict.get('rec_text', ""); ocr_score = recognition_result_dict.get('rec_score', 0.0)
                if raw_recognized_text and raw_recognized_text not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_MAIN_SERIAL', 'SERIAL_OCR_INIT_FAIL']:
                    digits_only_text = "".join(re.findall(r'\d', raw_recognized_text))
                    if digits_only_text: ocr_text_to_draw = digits_only_text; current_box_info["ocr_final_text"] = digits_only_text; current_box_info["ocr_confidence"] = ocr_score
                    else: current_box_info["ocr_final_text"] = "N/A_NO_DIGITS"; current_box_info["ocr_confidence"] = ocr_score
                else: current_box_info["ocr_final_text"] = raw_recognized_text; current_box_info["ocr_confidence"] = 0.0
            else: current_box_info["ocr_final_text"] = "N/A_INVALID_RESULT_FORMAT"; current_box_info["ocr_confidence"] = 0.0
            ocr_texts_for_drawing[original_idx] = ocr_text_to_draw; final_ocr_results_list[original_idx] = current_box_info
    else:
        logger.info("No ROIs for OCR."); timing_profile['7_ocr_processing_total'] = 0
        if len(aggregated_boxes_xyxy) > 0: # Populate with N/A if no OCR but boxes existed
            for i in range(len(aggregated_boxes_xyxy)):
                if recognized_obu_data_list_intermediate[i]: final_ocr_results_list[i] = {**recognized_obu_data_list_intermediate[i], "ocr_final_text": "N/A_NO_OCR_PERFORMED", "ocr_confidence": 0.0}
                else: final_ocr_results_list[i] = {"original_index": i, "roi_index": i + 1, "bbox_yolo": aggregated_boxes_xyxy[i], "confidence_yolo": aggregated_scores[i], "class": COCO_CLASSES[aggregated_class_ids[i]] if COCO_CLASSES else "N/A", "ocr_final_text": "N/A_NO_OCR_PERFORMED", "ocr_confidence": 0.0 }


    # 5. Prepare inputs for matrix building
    yolo_anchors_for_matrix = []
    for i, box_xyxy in enumerate(aggregated_boxes_xyxy): # Use filtered YOLO boxes
        cx, cy, w, h = get_box_center_and_dims(box_xyxy)
        if cx is not None:
            yolo_anchors_for_matrix.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box_xyxy, 'score': aggregated_scores[i]})

    paddle_results_for_matrix = []
    for ocr_res_item in final_ocr_results_list:
        if ocr_res_item and ocr_res_item.get("ocr_final_text", "N/A_NO_OCR_PERFORMED") not in ["N/A_NO_OCR_PERFORMED", "N/A_NO_DIGITS", "N/A_INVALID_RESULT_FORMAT", "INIT_FAIL", "PREDICT_FAIL", "PREPROC_FAIL", "WORKER_INIT_FAIL", "SERIAL_OCR_INIT_FAIL"]:
            # Use bbox_digit_ocr_clipped for cx, cy for matching, as it's more precise for the text
            ocr_cx, ocr_cy, ocr_w, ocr_h = get_box_center_and_dims(ocr_res_item.get('bbox_digit_ocr_clipped'))
            if ocr_cx is not None:
                paddle_results_for_matrix.append({
                    'text': ocr_res_item['ocr_final_text'],
                    'score': ocr_res_item.get('ocr_confidence', 0.0),
                    'cx': ocr_cx, 'cy': ocr_cy, 'w': ocr_w, 'h': ocr_h,
                    'box_ocr_clipped': ocr_res_item.get('bbox_digit_ocr_clipped') # Keep original clipped box
                })

    # 6. Build Smart Matrix
    obu_matrix, filled_count, ideal_grid_slots_for_drawing = build_obu_matrix_smart( # MODIFIED: Receive ideal_grid_slots
        yolo_anchors_input=yolo_anchors_for_matrix,
        paddle_results_input=paddle_results_for_matrix,
        layout_config=LAYOUT_CONFIG,
        image_wh=(orig_img_w, orig_img_h),
        logger=logger
    )

    timing_profile['9_matrix_building'] = time.time() - (timing_profile.get('7_ocr_processing_total',0) + t_start_img_read + timing_profile.get('3a_fullimg_preprocessing',0) + timing_profile.get('3b_fullimg_inference',0) + timing_profile.get('3c_fullimg_postprocessing',0) + timing_profile.get('5_area_filtering',0) ) # Approximate


    # 7. Save annotated image (if enabled)
    # ... (image saving logic from v2.5.4, using aggregated_boxes_xyxy and ocr_texts_for_drawing) ...
    if SAVE_PROCESS_PHOTOS and len(aggregated_boxes_xyxy) > 0:
        output_img_to_draw_on = original_image.copy()
        valid_roi_indices_for_drawing = [item["roi_index"] for item in final_ocr_results_list if item and "roi_index" in item]
        output_img_to_draw_on = draw_detections(output_img_to_draw_on, np.array(aggregated_boxes_xyxy), np.array(aggregated_scores), np.array(aggregated_class_ids), COCO_CLASSES, ocr_texts=ocr_texts_for_drawing, roi_indices=valid_roi_indices_for_drawing)
        if not os.path.exists(PROCESS_PHOTO_DIR): os.makedirs(PROCESS_PHOTO_DIR, exist_ok=True)
        output_fn_base = os.path.splitext(os.path.basename(image_path))[0]; timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        final_output_image_name = f"annotated_{output_fn_base}_{timestamp}.jpg"; final_output_path = os.path.join(PROCESS_PHOTO_DIR, final_output_image_name)
        try:
            cv2.imwrite(final_output_path, output_img_to_draw_on, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY])
            logger.info(f"最终结果图已保存到 (JPG): {final_output_path}")
        except Exception as e_imwrite: logger.error(f"保存标注图片失败: {e_imwrite}")
        timing_profile['8_drawing_results_final'] = time.time() - ( (timing_profile.get('7_ocr_processing_total',0) + t_start_img_read + timing_profile.get('3a_fullimg_preprocessing',0) + timing_profile.get('3b_fullimg_inference',0) + timing_profile.get('3c_fullimg_postprocessing',0) + timing_profile.get('5_area_filtering',0) ) + timing_profile.get('9_matrix_building',0) )
    elif len(aggregated_boxes_xyxy) > 0:
        logger.info("SAVE_PROCESS_PHOTOS is False. 跳过保存过程图片。"); timing_profile['8_drawing_results_final'] = 0
    else:
        logger.info("最终未检测到任何OBU ROI，无法进行OCR或绘图。"); timing_profile['8_drawing_results_final'] = 0

    # 8. NEW: Save Matrix Visualization Image (if enabled)
    if SAVE_PROCESS_PHOTOS and obu_matrix and ideal_grid_slots_for_drawing: # MODIFIED: Use the received variable
        t_start_matrix_draw = time.time()
        matrix_viz_image = draw_obu_matrix_on_image(
            obu_matrix,
            LAYOUT_CONFIG,
            logger
        )

        matrix_viz_fn_base = os.path.splitext(os.path.basename(image_path))[0]
        matrix_viz_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        matrix_viz_image_name = f"matrix_viz_{matrix_viz_fn_base}_{matrix_viz_timestamp}.jpg"
        matrix_viz_output_path = os.path.join(PROCESS_PHOTO_DIR, matrix_viz_image_name)
        try:
            cv2.imwrite(matrix_viz_output_path, matrix_viz_image, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY])
            logger.info(f"矩阵可视化图已保存到 (JPG): {matrix_viz_output_path}")
        except Exception as e_imwrite_matrix:
            logger.error(f"保存矩阵可视化图片失败: {e_imwrite_matrix}")
        timing_profile['10_matrix_visualization_drawing'] = time.time() - t_start_matrix_draw
    elif SAVE_PROCESS_PHOTOS:
        logger.warning("SAVE_PROCESS_PHOTOS is True, 但矩阵或理想坑位数据不足，无法生成矩阵可视化图。")

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    logger.info(f"--- 时间分析概要 ({VERSION}) for {os.path.basename(image_path)} ---")
    for stage, duration in sorted(timing_profile.items()):
        logger.info(f"  {stage}: {duration:.3f} 秒")

    # Return both the flat list and the matrix
    return final_ocr_results_list, obu_matrix, timing_profile


# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    # ... (Route logic is mostly the same, but now expects 3 return values from process_image_with_ocr_logic) ...
    logger = current_app.logger
    if 'file' not in request.files: logger.warning("请求中未找到文件部分。"); return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '': logger.warning("未选择文件。"); return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename); timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename); filename = f"{name}_{timestamp}{ext}"
        upload_dir = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        try:
            file.save(filepath)
            logger.info(f"服务端：文件 '{filename}' 已成功保存到 '{filepath}'")
            global onnx_session
            if onnx_session is None:
                logger.error("ONNX session is not initialized!")
                return jsonify({"error": "ONNX session not initialized on server"}), 500

    # Retrieve area detection configs from app.config
            min_area_from_config = current_app.config['MIN_DETECTION_AREA_CFG']
            max_area_from_config = current_app.config['MAX_DETECTION_AREA_CFG']

            ocr_results_flat_list, obu_matrix_result, timings = process_image_with_ocr_logic(
                filepath,
                onnx_session,
                min_area_from_config,  # Pass value retrieved from app.config
                max_area_from_config   # Pass value retrieved from app.config
            )

            return jsonify({
                "message": "File processed successfully.",
                "received_filename": original_filename,
                "saved_filepath": filepath,
                "ocr_results_list": ocr_results_flat_list, # Keep the detailed flat list
                "obu_matrix": obu_matrix_result,          # Add the new smart matrix
                "timing_profile_seconds": timings
            }), 200
        except FileNotFoundError as e_fnf:
            logger.error(f"文件处理错误 (FileNotFound): {e_fnf}\n{traceback.format_exc()}")
            return jsonify({"error": f"File processing error: {str(e_fnf)}"}), 500
        except Exception as e:
            logger.error(f"处理图片时发生严重错误: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"An unexpected error occurred during processing: {str(e)}"}), 500
    else:
        logger.warning(f"文件类型不允许: {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400

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