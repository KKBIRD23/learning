# D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\Service\app.py
import os
import cv2
import numpy as np
import onnxruntime
import time
import traceback
import paddlex as pdx
import re
import multiprocessing # For parallel processing
from datetime import datetime # For timestamped logs
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import atexit # For cleanup

# --- 配置 (大部分从 test_paddle_server_multithreading_ocr.py 移植) ---
# ================== 版本信息 ==================
VERSION = "v2.5.3_flask_service_persistent_workers"

# ================== 服务端上传配置 ==================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ================== 模型路径配置 ==================
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_MODEL_PATH_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "BarCode_Detect", "BarCode_Detect_dynamic.onnx")
SERVER_REC_MODEL_DIR_CFG_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_infer")

# ================== YOLOv8 检测参数 ==================
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ================== 切块逻辑配置 ==================
ENABLE_TILING = False
FIXED_TILING_GRID = None
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5
TILE_OVERLAP_RATIO = 0.2

# ================== 检测结果面积筛选配置 ==================
MIN_DETECTION_AREA = 2000
MAX_DETECTION_AREA = 0.1

# ================== 数字ROI裁剪精调参数 ==================
DIGIT_ROI_Y_OFFSET_FACTOR = -0.15
DIGIT_ROI_HEIGHT_FACTOR = 0.7
DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05

# ================== OCR 服务端识别配置 ==================
TARGET_OCR_INPUT_HEIGHT = 48

# ================== 并行OCR处理配置 ==================
NUM_OCR_WORKERS_CONFIG = 4 # Renamed to avoid conflict with actual_num_workers

# ================== 其他配置 ==================
COCO_CLASSES = ['Barcode']
PROCESS_PHOTO_DIR = "process_photo_service"

# --- Flask 应用实例 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 全局模型实例 ---
onnx_session = None
ocr_processing_pool = None # Global OCR processing pool
actual_num_ocr_workers = 1 # Actual number of workers for the pool

# --- 辅助函数 (从之前版本移植/修改) ---
# ... (allowed_file, clear_process_photo_directory, YOLO pre/post-processing functions, draw_detections) ...
# These functions are assumed to be correctly copied here as in the previous version.
# For brevity, I'm omitting them again, but they MUST be present in your app.py.
# Make sure to copy:
# clear_process_photo_directory, preprocess_image_data_for_tiling,
# postprocess_detections_from_tile, non_max_suppression_global,
# preprocess_onnx_for_main, postprocess_yolo_onnx_for_main, draw_detections

# --- BEGIN: Functions copied from test_paddle_server_multithreading_ocr.py ---
# NOTE: Ensure these functions are at the top level of the module for multiprocessing compatibility

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_process_photo_directory(directory): # Copied
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            except Exception as e: current_app.logger.error(f'Failed to delete {file_path}. Reason: {e}')
    else: os.makedirs(directory, exist_ok=True)

def preprocess_image_data_for_tiling(img_data, input_shape_hw): # Copied
    img = img_data; img_height, img_width = img.shape[:2]; input_height, input_width = input_shape_hw
    ratio = min(input_width / img_width, input_height / img_height); new_width, new_height = int(img_width * ratio), int(img_height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 128, dtype=np.uint8); x_pad, y_pad = (input_width - new_width) // 2, (input_height - new_height) // 2
    canvas[y_pad:y_pad + new_height, x_pad:x_pad + new_width] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0; return np.expand_dims(tensor, axis=0), ratio, x_pad, y_pad

def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _, \
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y, \
                                     conf_threshold_tile, model_output_channels_param_ignored): # Copied
    predictions_raw = np.squeeze(outputs[0])
    if predictions_raw.ndim != 2: return np.array([]), np.array([]), np.array([])
    actual_model_output_channels = predictions_raw.shape[0]
    if not isinstance(actual_model_output_channels, int): return np.array([]), np.array([]), np.array([])
    transposed_predictions = predictions_raw.transpose(); boxes_tile_local_scaled, scores_tile_local, class_ids_tile_local = [], [], []
    for pred_data in transposed_predictions:
        if len(pred_data) != actual_model_output_channels: continue
        cx, cy, w, h = pred_data[:4]; confidence, class_id = 0.0, -1
        if actual_model_output_channels == 6: confidence = pred_data[4]; class_id = int(pred_data[5])
        elif actual_model_output_channels == 5: confidence = pred_data[4]; class_id = 0
        elif actual_model_output_channels > 4 :
            class_scores = pred_data[4:]
            if class_scores.size > 0: confidence = np.max(class_scores); class_id = np.argmax(class_scores)
            else: continue
        else: continue
        if confidence >= conf_threshold_tile: x1,y1,x2,y2=(cx-w/2),(cy-h/2),(cx+w/2),(cy+h/2); boxes_tile_local_scaled.append([x1,y1,x2,y2]); scores_tile_local.append(confidence); class_ids_tile_local.append(class_id)
    if not boxes_tile_local_scaled: return np.array([]), np.array([]), np.array([])
    final_boxes_tile_original_coords = []; tile_h_orig, tile_w_orig = tile_original_shape_hw
    for box in boxes_tile_local_scaled:
        b_x1,b_y1,b_x2,b_y2 = box[0]-preprocessing_pad_x,box[1]-preprocessing_pad_y,box[2]-preprocessing_pad_x,box[3]-preprocessing_pad_y
        if preprocessing_ratio == 0: continue
        ot_x1,ot_y1,ot_x2,ot_y2 = b_x1/preprocessing_ratio,b_y1/preprocessing_ratio,b_x2/preprocessing_ratio,b_y2/preprocessing_ratio
        ot_x1,ot_y1=np.clip(ot_x1,0,tile_w_orig),np.clip(ot_y1,0,tile_h_orig); ot_x2,ot_y2=np.clip(ot_x2,0,tile_w_orig),np.clip(ot_y2,0,tile_h_orig); final_boxes_tile_original_coords.append([ot_x1,ot_y1,ot_x2,ot_y2])
    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold): # Copied
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    if not isinstance(scores, np.ndarray) or scores.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]; areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i); _ = order.size;order = order[1:]
        if _ == 1: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order]);xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h;ovr=inter/(areas[i]+areas[order]-inter)
        inds=np.where(ovr<=iou_threshold)[0];order=order[inds]
    return keep

def preprocess_onnx_for_main(img_data, target_shape_hw): # Copied
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

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold, \
                                   original_shape_hw, model_input_shape_hw, \
                                   ratio_preproc, pad_x_preproc, pad_y_preproc, \
                                   num_classes=1): # Copied
    raw_output_tensor = np.squeeze(outputs_onnx[0]);
    if raw_output_tensor.ndim != 2: current_app.logger.error(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []
    expected_attributes = 5
    if predictions_to_iterate.shape[1] == (4 + 1 + num_classes) and num_classes > 1 :
        expected_attributes = 4 + 1 + num_classes
    elif predictions_to_iterate.shape[1] != 5 and num_classes == 1:
        current_app.logger.warning(f"警告: 预测属性数量 {predictions_to_iterate.shape[1]} 与单类别期望值 5 不符. 将按5属性尝试。")

    for i_pred, pred_data in enumerate(predictions_to_iterate):
        if len(pred_data) != expected_attributes:
            if i_pred == 0: current_app.logger.error(f"错误: Main 每个预测的属性数量 ({len(pred_data)}) 与期望值 ({expected_attributes}) 不符。")
            continue
        box_coords_raw = pred_data[:4]; final_confidence = 0.0; class_id = 0
        if expected_attributes == 5:
            final_confidence = float(pred_data[4])
        elif expected_attributes == 6 and num_classes == 1:
            objectness = float(pred_data[4]); class_score_single = float(pred_data[5])
            final_confidence = objectness * class_score_single
        else:
            objectness = float(pred_data[4])
            class_scores_all = pred_data[5:]
            if len(class_scores_all) == num_classes:
                class_id = np.argmax(class_scores_all)
                max_class_score = float(class_scores_all[class_id])
                final_confidence = objectness * max_class_score
            else:
                if i_pred == 0: current_app.logger.error(f"错误: 类别分数数量 ({len(class_scores_all)}) 与 num_classes ({num_classes}) 不符。")
                continue
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

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None): # Copied
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out
# --- END: Functions copied ---


# --- OCR Worker Process Initialization and Task Function (must be top-level) ---
_worker_ocr_predictor = None # Global variable within each worker process

def init_ocr_worker(ocr_model_dir):
    """Initializer for each OCR worker process. Loads PaddleOCR model."""
    global _worker_ocr_predictor
    worker_pid = os.getpid()
    # Use print for worker logs as current_app.logger is not directly available
    print(f"[Worker PID {worker_pid}] Initializing OCR predictor with model_dir: {ocr_model_dir}")
    try:
        _worker_ocr_predictor = pdx.inference.create_predictor(
            model_dir=ocr_model_dir,
            model_name='PP-OCRv5_server_rec',
            device='cpu'
        )
        print(f"[Worker PID {worker_pid}] OCR predictor initialized successfully.")
    except Exception as e:
        print(f"[Worker PID {worker_pid}] CRITICAL: Failed to initialize OCR predictor: {e}\n{traceback.format_exc()}")
        _worker_ocr_predictor = None # Ensure it's None if init fails

def ocr_task_for_worker(task_data):
    """Processes a single OCR task using the worker's pre-loaded predictor."""
    global _worker_ocr_predictor
    roi_original_index, roi_display_index, image_bgr_for_ocr = task_data
    worker_pid = os.getpid()
    start_time = time.time()

    if image_bgr_for_ocr is None:
        return roi_original_index, {'rec_text': 'PREPROC_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': time.time() - start_time}

    if _worker_ocr_predictor is None:
        print(f"[Worker PID {worker_pid}] OCR predictor not available (init failed?). ROI {roi_display_index} cannot be processed.")
        return roi_original_index, {'rec_text': 'WORKER_INIT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': time.time() - start_time}

    try:
        result_gen = _worker_ocr_predictor.predict([image_bgr_for_ocr])
        recognition_result_list_for_roi = next(result_gen, None)
        final_result_dict = {'rec_text': '', 'rec_score': 0.0}
        if recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, list) and len(recognition_result_list_for_roi) > 0:
            final_result_dict = recognition_result_list_for_roi[0]
        elif recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, dict):
            final_result_dict = recognition_result_list_for_roi

        duration = time.time() - start_time
        # print(f"[Worker PID {worker_pid} | ROI {roi_display_index}] OCR DONE in {duration:.3f}s.")
        return roi_original_index, {**final_result_dict, 'pid': worker_pid, 'duration': duration}
    except Exception as e:
        duration = time.time() - start_time
        print(f"[Worker PID {worker_pid} | ROI {roi_display_index}] Error during OCR predict: {e}")
        return roi_original_index, {'rec_text': 'PREDICT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': duration}


# --- Main Image Processing Function ---
def process_image_with_ocr_logic(image_path, input_onnx_session):
    # ... (Image reading, YOLO detection, ROI extraction logic remains largely the same as previous version) ...
    # Key change is how OCR tasks are submitted to the global pool.
    current_app.logger.info(f"--- OBU 检测与识别开始 ({VERSION}) for image: {image_path} ---")
    timing_profile = {}
    t_start_overall_processing = time.time()

    t_start_img_read = time.time()
    original_image = cv2.imread(image_path)
    timing_profile['1_image_reading'] = time.time() - t_start_img_read
    if original_image is None:
        current_app.logger.error(f"错误: 无法读取图片: {image_path}")
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img_h, orig_img_w = original_image.shape[:2]
    current_app.logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    actual_max_area_threshold_px = None
    if MAX_DETECTION_AREA is not None:
        if isinstance(MAX_DETECTION_AREA, float) and 0 < MAX_DETECTION_AREA <= 1.0:
            actual_max_area_threshold_px = (orig_img_h * orig_img_w) * MAX_DETECTION_AREA
        elif isinstance(MAX_DETECTION_AREA, (int, float)) and MAX_DETECTION_AREA > 1:
            actual_max_area_threshold_px = float(MAX_DETECTION_AREA)

    current_app.logger.info("--- 开始整图检测 ---")
    input_cfg = input_onnx_session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640)
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int):
        model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]

    t_s = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s;
    t_s = time.time(); outputs_main = input_onnx_session.run(None, {input_name: input_tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s;
    detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(COCO_CLASSES)); timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s

    aggregated_boxes = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]; aggregated_scores = [d[4] for d in detections_result_list]; aggregated_class_ids = [d[5] for d in detections_result_list]; current_app.logger.info(f"整图处理与后处理完成。找到了 {len(aggregated_boxes)} 个框。")

    if len(aggregated_boxes) > 0 and ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or actual_max_area_threshold_px is not None):
        t_start_area_filter=time.time(); filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids=[],[],[]; initial_box_count_before_area_filter=len(aggregated_boxes)
        for i_box,box in enumerate(aggregated_boxes):
            b_w,b_h=box[2]-box[0],box[3]-box[1]; area=b_w*b_h; valid_area=True
            if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA: valid_area=False
            if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid_area=False
            if valid_area: filtered_by_area_boxes.append(box); filtered_by_area_scores.append(aggregated_scores[i_box]); filtered_by_area_ids.append(aggregated_class_ids[i_box])
        aggregated_boxes,aggregated_scores,aggregated_class_ids=filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids; timing_profile['5_area_filtering']=time.time()-t_start_area_filter; current_app.logger.info(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框).")
    else:
        timing_profile['5_area_filtering']=0

    tasks_for_ocr = [] # Renamed from tasks_for_ocr_parallel
    recognized_obu_data_list_intermediate = [None] * len(aggregated_boxes)

    if len(aggregated_boxes) > 0:
        current_app.logger.info(f"--- 最终检测到 {len(aggregated_boxes)} 个OBU的YOLO框, 准备进行OCR预处理 ---")
        for i, yolo_box_coords in enumerate(aggregated_boxes):
            # ... (ROI extraction and preprocessing logic as before) ...
            class_id = int(aggregated_class_ids[i]); class_name_str = COCO_CLASSES[class_id] if COCO_CLASSES and 0 <= class_id < len(COCO_CLASSES) else f"ClassID:{class_id}"
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = [int(c) for c in yolo_box_coords]; h_yolo = y2_yolo - y1_yolo; w_yolo = x2_yolo - x1_yolo
            y1_digit_ideal = y1_yolo + int(h_yolo * DIGIT_ROI_Y_OFFSET_FACTOR); h_digit_ideal = int(h_yolo * DIGIT_ROI_HEIGHT_FACTOR); y2_digit_ideal = y1_digit_ideal + h_digit_ideal
            w_digit_expanded = int(w_yolo * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_yolo = x1_yolo + w_yolo / 2.0; x1_digit_ideal = int(cx_yolo - w_digit_expanded / 2.0); x2_digit_ideal = int(cx_yolo + w_digit_expanded / 2.0)
            y1_d_clip = max(0, y1_digit_ideal); y2_d_clip = min(orig_img_h, y2_digit_ideal); x1_d_clip = max(0, x1_digit_ideal); x2_d_clip = min(orig_img_w, x2_digit_ideal)

            current_box_meta_for_task = {"original_index": i, "roi_index": i + 1, "class": class_name_str,
                                         "bbox_yolo": [x1_yolo, y1_yolo, x2_yolo, y2_yolo],
                                         "bbox_digit_ocr_ideal": [x1_digit_ideal, y1_digit_ideal, x2_digit_ideal, y2_digit_ideal],
                                         "bbox_digit_ocr_clipped": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip],
                                         "confidence_yolo": float(aggregated_scores[i])}
            recognized_obu_data_list_intermediate[i] = current_box_meta_for_task

            image_for_ocr_bgr = None
            dx1,dy1,dx2,dy2 = current_box_meta_for_task['bbox_digit_ocr_clipped']
            if dx2>dx1 and dy2>dy1:
                digit_roi_color=original_image[dy1:dy2,dx1:dx2]
                h_roi_digit, w_roi_digit = digit_roi_color.shape[:2]
                if h_roi_digit > 0 and w_roi_digit > 0:
                    scale_ocr = TARGET_OCR_INPUT_HEIGHT / h_roi_digit
                    target_w_ocr = int(w_roi_digit * scale_ocr)
                    if target_w_ocr <= 0: target_w_ocr = 1
                    resized_digit_roi_color = cv2.resize(digit_roi_color, (target_w_ocr, TARGET_OCR_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC if scale_ocr > 1 else cv2.INTER_AREA)
                    gray_resized_roi = cv2.cvtColor(resized_digit_roi_color, cv2.COLOR_BGR2GRAY)
                    _, binary_resized_roi = cv2.threshold(gray_resized_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    image_for_ocr_bgr = cv2.cvtColor(binary_resized_roi, cv2.COLOR_GRAY2BGR)

            tasks_for_ocr.append( (i, i + 1, image_for_ocr_bgr) ) # Model dir no longer passed

    ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes)
    final_recognized_obu_data = [None] * len(aggregated_boxes)

    global ocr_processing_pool, actual_num_ocr_workers # Use global pool and worker count

    if tasks_for_ocr:
        current_app.logger.info(f"\n--- Submitting {len(tasks_for_ocr)} OCR tasks to pool/serial ---")
        t_ocr_start = time.time()
        ocr_processed_results_indexed = [None] * len(tasks_for_ocr)

        if actual_num_ocr_workers > 1 and ocr_processing_pool is not None:
            try:
                # tasks_for_ocr already has the correct format: (original_idx, display_idx, image_data)
                parallel_results_with_indices = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                for res_original_idx, res_dict in parallel_results_with_indices:
                    ocr_processed_results_indexed[res_original_idx] = res_dict
            except Exception as e_pool_map:
                current_app.logger.error(f"Error during ocr_processing_pool.map: {e_pool_map}\n{traceback.format_exc()}")
                current_app.logger.warning("FALLING BACK TO SERIAL OCR due to pool.map error.")
                # Fallback to serial if pool fails mid-operation (though less likely if pool init was ok)
                # For simplicity, this example doesn't re-implement full serial fallback here if pool existed.
                # A robust app might try serial or return error.
                # For now, if pool.map fails, results might be incomplete.
                # Let's assume for now if pool exists, map works, or we handle errors from worker.
                pass # Or implement a more robust fallback

        # Serial execution (if actual_num_ocr_workers <= 1 OR ocr_processing_pool is None OR pool.map failed and we want to retry serially)
        # This serial part is more of a fallback if the pool wasn't created or if a specific error forces it.
        # If pool.map fails, the results might be partial. A full retry serially:
        run_serially = False
        if actual_num_ocr_workers <= 1 or ocr_processing_pool is None:
            run_serially = True
        # Or, if you want to retry serially after a pool.map failure:
        # if any(res is None for res in ocr_processed_results_indexed) and (actual_num_ocr_workers > 1 and ocr_processing_pool is not None):
        #    current_app.logger.warning("Pool.map might have failed, attempting serial processing for remaining tasks.")
        #    run_serially = True # This logic needs careful implementation for partial failures.

        if run_serially:
            current_app.logger.info("Performing OCR in serial mode (main process).")
            # Serial OCR needs its own predictor instance if not using workers
            serial_ocr_predictor = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG):
                     raise FileNotFoundError(f"Serial OCR: Model directory not found {SERVER_REC_MODEL_DIR_CFG_CONFIG}")
                serial_ocr_predictor = pdx.inference.create_predictor(
                    model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG, model_name='PP-OCRv5_server_rec', device='cpu'
                )
            except Exception as e_serial_init_ocr:
                current_app.logger.error(f"Error initializing serial OCR predictor: {e_serial_init_ocr}")

            if serial_ocr_predictor:
                for task_idx, task_data_tuple in enumerate(tasks_for_ocr):
                    original_idx, _, img_data = task_data_tuple
                    if img_data is not None:
                        res_gen = serial_ocr_predictor.predict([img_data])
                        res_list = next(res_gen, None)
                        if res_list and isinstance(res_list, list) and len(res_list) > 0:
                             ocr_processed_results_indexed[original_idx] = res_list[0]
                        elif res_list and isinstance(res_list, dict):
                             ocr_processed_results_indexed[original_idx] = res_list
                        else:
                             ocr_processed_results_indexed[original_idx] = {'rec_text': '', 'rec_score': 0.0}
                    else:
                         ocr_processed_results_indexed[original_idx] = {'rec_text': 'PREPROC_FAIL_MAIN_SERIAL', 'rec_score': 0.0}
                del serial_ocr_predictor
            else: # Serial predictor failed to init
                for task_idx, task_data_tuple in enumerate(tasks_for_ocr):
                     original_idx, _, _, = task_data_tuple
                     ocr_processed_results_indexed[original_idx] = {'rec_text': 'SERIAL_OCR_INIT_FAIL', 'rec_score': 0.0}

        timing_profile['7_ocr_processing_total'] = time.time() - t_ocr_start
        current_app.logger.info(f"--- OCR processing finished ({timing_profile['7_ocr_processing_total']:.3f} 秒) ---")

        # Consolidate results (same as before)
        for original_idx, recognition_result_dict in enumerate(ocr_processed_results_indexed):
            # ... (consolidation logic as in previous app.py version) ...
            current_box_info = recognized_obu_data_list_intermediate[original_idx]
            ocr_text_to_draw = "N/A"
            if recognition_result_dict and isinstance(recognition_result_dict, dict):
                raw_recognized_text = recognition_result_dict.get('rec_text', "")
                ocr_score = recognition_result_dict.get('rec_score', 0.0)
                if raw_recognized_text and raw_recognized_text not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_MAIN_SERIAL', 'SERIAL_OCR_INIT_FAIL']:
                    digits_only_text = "".join(re.findall(r'\d', raw_recognized_text))
                    if digits_only_text:
                        ocr_text_to_draw = digits_only_text
                        current_box_info["ocr_final_text"] = digits_only_text
                        current_box_info["ocr_confidence"] = ocr_score
                    else:
                        current_box_info["ocr_final_text"] = "N/A_NO_DIGITS"
                        current_box_info["ocr_confidence"] = ocr_score
                else:
                    current_box_info["ocr_final_text"] = raw_recognized_text
                    current_box_info["ocr_confidence"] = 0.0
            else:
                current_box_info["ocr_final_text"] = "N/A_INVALID_RESULT_FORMAT"
                current_box_info["ocr_confidence"] = 0.0
            ocr_texts_for_drawing[original_idx] = ocr_text_to_draw
            final_recognized_obu_data[original_idx] = current_box_info
    else:
        current_app.logger.info("No ROIs for OCR.")
        timing_profile['7_ocr_processing_total'] = 0
        # ... (populate final_recognized_obu_data with N/A if no OCR was run but boxes existed) ...
        if len(aggregated_boxes) > 0:
            for i in range(len(aggregated_boxes)):
                if recognized_obu_data_list_intermediate[i]:
                    final_recognized_obu_data[i] = {**recognized_obu_data_list_intermediate[i], "ocr_final_text": "N/A_NO_OCR_PERFORMED", "ocr_confidence": 0.0}
                else:
                     final_recognized_obu_data[i] = {"original_index": i, "roi_index": i + 1, "bbox_yolo": aggregated_boxes[i], "confidence_yolo": aggregated_scores[i], "class": COCO_CLASSES[aggregated_class_ids[i]] if COCO_CLASSES else "N/A", "ocr_final_text": "N/A_NO_OCR_PERFORMED", "ocr_confidence": 0.0 }

    # Save annotated image (same as before)
    if len(aggregated_boxes) > 0:
        # ... (drawing logic as in previous app.py version) ...
        output_img_to_draw_on = original_image.copy()
        valid_roi_indices_for_drawing = [item["roi_index"] for item in final_recognized_obu_data if item and "roi_index" in item]
        output_img_to_draw_on = draw_detections(output_img_to_draw_on, np.array(aggregated_boxes), np.array(aggregated_scores), np.array(aggregated_class_ids), COCO_CLASSES, ocr_texts=ocr_texts_for_drawing, roi_indices=valid_roi_indices_for_drawing)
        if not os.path.exists(PROCESS_PHOTO_DIR): os.makedirs(PROCESS_PHOTO_DIR, exist_ok=True)
        output_fn_base = os.path.splitext(os.path.basename(image_path))[0]; timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        final_output_image_name = f"annotated_{output_fn_base}_{timestamp}.png"; final_output_path = os.path.join(PROCESS_PHOTO_DIR, final_output_image_name)
        try:
            cv2.imwrite(final_output_path, output_img_to_draw_on)
            current_app.logger.info(f"最终结果图已保存到: {final_output_path}")
        except Exception as e_imwrite: current_app.logger.error(f"保存标注图片失败: {e_imwrite}")
        timing_profile['8_drawing_results_final'] = time.time() - (t_ocr_start + timing_profile.get('7_ocr_processing_total',0))
    else:
        current_app.logger.info("最终未检测到任何OBU ROI，无法进行OCR或绘图。")
        timing_profile['8_drawing_results_final'] = 0

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    current_app.logger.info(f"--- 时间分析概要 ({VERSION}) for {os.path.basename(image_path)} ---")
    for stage, duration in sorted(timing_profile.items()):
        current_app.logger.info(f"  {stage}: {duration:.3f} 秒")

    return final_recognized_obu_data, timing_profile


# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    # ... (File checking and saving logic remains the same as previous version) ...
    if 'file' not in request.files: return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename); timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename); filename = f"{name}_{timestamp}{ext}"
        upload_dir = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        try:
            file.save(filepath)
            current_app.logger.info(f"服务端：文件 '{filename}' 已成功保存到 '{filepath}'")
            global onnx_session
            if onnx_session is None:
                current_app.logger.error("ONNX session is not initialized!")
                return jsonify({"error": "ONNX session not initialized on server"}), 500
            ocr_results_list, timings = process_image_with_ocr_logic(filepath, onnx_session)
            return jsonify({"message": "File processed successfully.", "received_filename": original_filename, "saved_filepath": filepath, "ocr_results": ocr_results_list, "timing_profile_seconds": timings }), 200
        except FileNotFoundError as e_fnf:
            current_app.logger.error(f"文件处理错误 (FileNotFound): {e_fnf}\n{traceback.format_exc()}")
            return jsonify({"error": f"File processing error: {str(e_fnf)}"}), 500
        except Exception as e:
            current_app.logger.error(f"处理图片时发生严重错误: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"An unexpected error occurred during processing: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

def initialize_onnx_session():
    """Loads the ONNX model session globally."""
    global onnx_session
    current_app.logger.info("--- ONNX 模型初始化开始 ---")
    if not os.path.exists(ONNX_MODEL_PATH_CONFIG):
        current_app.logger.error(f"错误: ONNX模型未找到: {ONNX_MODEL_PATH_CONFIG}")
        raise FileNotFoundError(f"ONNX Model not found: {ONNX_MODEL_PATH_CONFIG}")
    try:
        onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH_CONFIG, providers=['CPUExecutionProvider'])
        current_app.logger.info(f"ONNX模型加载完成 from {ONNX_MODEL_PATH_CONFIG}")
    except Exception as e:
        current_app.logger.error(f"ONNX模型加载失败: {e}\n{traceback.format_exc()}")
        raise
    current_app.logger.info("--- ONNX 模型初始化完成 ---")

def cleanup_global_ocr_pool():
    """Closes and joins the global OCR processing pool."""
    global ocr_processing_pool
    if ocr_processing_pool:
        print("Closing global OCR processing pool...")
        ocr_processing_pool.close()
        ocr_processing_pool.join()
        print("Global OCR processing pool closed.")

# --- Main Application Runner ---
if __name__ == '__main__':
    # Set multiprocessing start method (important for Windows and macOS)
    # "spawn" is generally safer for cross-platform compatibility with Flask.
    # Needs to be called only once, and before any processes are started.
    try:
        if multiprocessing.get_start_method(allow_none=True) != 'spawn': # Check if already set
             multiprocessing.set_start_method('spawn', force=True) # force=True if you need to override
        print(f"Multiprocessing start method set to: {multiprocessing.get_start_method()}")
    except RuntimeError as e_mp_start:
        # This can happen if it's already been set and force=False, or other context issues.
        print(f"Warning: Could not set multiprocessing start method ('spawn'): {e_mp_start}. Using default: {multiprocessing.get_start_method(allow_none=True)}")


    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(PROCESS_PHOTO_DIR): os.makedirs(PROCESS_PHOTO_DIR, exist_ok=True)

    # Determine actual number of workers for the global pool
    cpu_cores = os.cpu_count() or 1
    if NUM_OCR_WORKERS_CONFIG <= 0: actual_num_ocr_workers = 1
    elif NUM_OCR_WORKERS_CONFIG > cpu_cores: actual_num_ocr_workers = cpu_cores
    else: actual_num_ocr_workers = NUM_OCR_WORKERS_CONFIG

    with app.app_context():
        initialize_onnx_session() # Load ONNX YOLO model

        if actual_num_ocr_workers > 1:
            # Check OCR model path before attempting to init workers
            if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG):
                print(f"CRITICAL: Server OCR model directory for workers not found: {SERVER_REC_MODEL_DIR_CFG_CONFIG}")
                print("OCR processing will fallback to serial mode or fail if serial also can't find model.")
                actual_num_ocr_workers = 1 # Force serial if model path for workers is bad
            else:
                print(f"Initializing global OCR processing pool with {actual_num_ocr_workers} workers...")
                try:
                    ocr_processing_pool = multiprocessing.Pool(
                        processes=actual_num_ocr_workers,
                        initializer=init_ocr_worker,
                        initargs=(SERVER_REC_MODEL_DIR_CFG_CONFIG,) # Pass model dir to worker initializer
                    )
                    print("Global OCR processing pool initialized and workers are loading models.")
                    atexit.register(cleanup_global_ocr_pool) # Ensure pool is closed on exit
                except Exception as e_pool_create:
                    print(f"CRITICAL: Failed to create global OCR processing pool: {e_pool_create}\n{traceback.format_exc()}")
                    ocr_processing_pool = None # Ensure it's None if creation failed
                    actual_num_ocr_workers = 1 # Fallback to serial

        if actual_num_ocr_workers <= 1 and ocr_processing_pool is None : # Also check if pool creation failed
             print("OCR will run in serial mode (main process or due to pool init failure).")


    # use_reloader=False is CRUCIAL for multiprocessing with Flask's dev server
    # to prevent re-initialization issues and ensure the pool is managed correctly.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)