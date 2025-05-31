# coding: utf-8
"""
OBU (On-Board Unit) 镭标码区域目标检测与数字识别脚本 (Server OCR - 并行处理版 - 详细日志)
版本: v2.5.2_server_parallel_ocr_detailed_log
功能:
- 与 v2.5.2_server_parallel_ocr 类似，采用多进程并行识别。
- 核心改动：在工作函数和主进程中增加详细的日志打印，用于追踪：
    - 工作进程PID。
    - 每个ROI由哪个进程处理。
    - 每个ROI在工作进程中的处理耗时。
    - 任务开始和结束时间点，以观察并行情况。
"""
import cv2
import numpy as np
import onnxruntime
import os
import time
import traceback
import paddlex as pdx
import re
import multiprocessing # For parallel processing
from datetime import datetime # For timestamped logs

# --- V2.5.2 配置参数 (Server OCR - 并行处理版 - 详细日志) ---
# ================== 版本信息 ==================
VERSION = "v2.5.2_server_parallel_ocr_detailed_log"

# ================== 模型路径配置 ==================
ONNX_MODEL_PATH = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
SERVER_REC_MODEL_DIR_CFG = r"D:\WorkSpaces\Python\WorkSpaces\Demos\ObuPrintNo_OCR\model\model\PaddleOCR\PP-OCRv5_server_rec_infer"

# ================== 输入图像配置 ==================
IMAGE_NAME = r"../../DATA/PIC/3.jpg"

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
OCR_PREPROCESS_TO_USE = "binary_otsu_digit"

# ================== 并行OCR处理配置 ==================
NUM_OCR_WORKERS = 4 # 设置为您期望的并行数，例如4

# ================== 其他配置 ==================
COCO_CLASSES = ['Barcode']
timing_profile = {}
process_photo_dir = "process_photo"


# --- 工作函数：用于并行处理单个ROI的OCR ---
def ocr_worker_process_roi_with_logging(task_data):
    """
    接收包含图像数据和任务信息的元组，执行OCR识别，并记录日志。
    task_data: (roi_original_index, roi_display_index, image_bgr_for_ocr)
    """
    roi_original_index, roi_display_index, image_bgr_for_ocr = task_data
    worker_pid = os.getpid()
    start_time = time.time()

    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] START at {timestamp_start}") # 太过详细，可以注释掉

    if image_bgr_for_ocr is None:
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Received None image.")
        return roi_original_index, {'rec_text': 'PREPROC_FAIL', 'rec_score': 0.0}

    try:
        # 每个工作进程独立初始化OCR引擎
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Initializing OCR predictor...")
        local_ocr_predictor = pdx.inference.create_predictor(
            model_dir=SERVER_REC_MODEL_DIR_CFG,
            model_name='PP-OCRv5_server_rec',
            device='cpu'
        )
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] OCR predictor initialized.")
    except Exception as e:
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Error initializing OCR predictor: {e}")
        return roi_original_index, {'rec_text': 'INIT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': time.time() - start_time}

    try:
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Predicting...")
        result_gen = local_ocr_predictor.predict([image_bgr_for_ocr])
        recognition_result_list_for_roi = next(result_gen, None)
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Prediction done.")

        final_result_dict = {'rec_text': '', 'rec_score': 0.0}
        if recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, list) and len(recognition_result_list_for_roi) > 0:
            final_result_dict = recognition_result_list_for_roi[0]
        elif recognition_result_list_for_roi and isinstance(recognition_result_list_for_roi, dict):
            final_result_dict = recognition_result_list_for_roi

        duration = time.time() - start_time
        timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] DONE in {duration:.3f}s at {timestamp_end}. Text: '{final_result_dict.get('rec_text', '')[:20]}...'")
        return roi_original_index, {**final_result_dict, 'pid': worker_pid, 'duration': duration}

    except Exception as e:
        duration = time.time() - start_time
        # print(f"  [Worker PID {worker_pid} | ROI {roi_display_index}] Error during OCR predict: {e}")
        return roi_original_index, {'rec_text': 'PREDICT_FAIL', 'rec_score': 0.0, 'pid': worker_pid, 'duration': duration}

# --- 辅助函数定义 (与之前版本一致) ---
# ... (clear_process_photo_directory, preprocess_image_data_for_tiling, etc. 保持不变, 此处省略)
def clear_process_photo_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): os.unlink(file_path)
            except Exception as e: print(f'Failed to delete {file_path}. Reason: {e}')
    else: os.makedirs(directory, exist_ok=True)

def preprocess_image_data_for_tiling(img_data, input_shape_hw):
    img = img_data; img_height, img_width = img.shape[:2]; input_height, input_width = input_shape_hw
    ratio = min(input_width / img_width, input_height / img_height); new_width, new_height = int(img_width * ratio), int(img_height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 128, dtype=np.uint8); x_pad, y_pad = (input_width - new_width) // 2, (input_height - new_height) // 2
    canvas[y_pad:y_pad + new_height, x_pad:x_pad + new_width] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0; return np.expand_dims(tensor, axis=0), ratio, x_pad, y_pad

def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold_tile, model_output_channels_param_ignored):
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

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold):
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

def preprocess_onnx_for_main(img_data, target_shape_hw):
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

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold,
                                   original_shape_hw, model_input_shape_hw,
                                   ratio_preproc, pad_x_preproc, pad_y_preproc,
                                   num_classes=1):
    raw_output_tensor = np.squeeze(outputs_onnx[0]);
    if raw_output_tensor.ndim != 2: print(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []
    expected_attributes = 5
    if predictions_to_iterate.shape[1] == (4 + 1 + num_classes) and num_classes > 1 :
        expected_attributes = 4 + 1 + num_classes
    elif predictions_to_iterate.shape[1] != 5 and num_classes == 1:
        print(f"警告: 预测属性数量 {predictions_to_iterate.shape[1]} 与单类别期望值 5 不符. 将按5属性尝试。")

    for i_pred, pred_data in enumerate(predictions_to_iterate):
        if len(pred_data) != expected_attributes:
            if i_pred == 0: print(f"错误: Main 每个预测的属性数量 ({len(pred_data)}) 与期望值 ({expected_attributes}) 不符。")
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
                if i_pred == 0: print(f"错误: 类别分数数量 ({len(class_scores_all)}) 与 num_classes ({num_classes}) 不符。")
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

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out

# --- 主程序 ---
if __name__ == "__main__":
    # --- 控制并行进程数 ---
    cpu_cores = os.cpu_count() if hasattr(os, 'cpu_count') else 1 # Fallback for older Python or restricted envs
    if NUM_OCR_WORKERS <= 0:
        actual_num_workers = 1
        print("并行OCR处理已禁用 (NUM_OCR_WORKERS <= 0)，将串行执行OCR。")
    elif NUM_OCR_WORKERS > cpu_cores:
        actual_num_workers = cpu_cores
        print(f"警告: 配置的并行OCR进程数 ({NUM_OCR_WORKERS}) 超出CPU核心数 ({cpu_cores})。将使用 {cpu_cores} 个进程。")
    else:
        actual_num_workers = NUM_OCR_WORKERS
    if actual_num_workers > 1 : # Only print if truly parallel
        print(f"将使用 {actual_num_workers} 个并行进程进行OCR。")


    t_start_overall = time.time(); timing_profile['0_total_script_execution'] = 0
    print(f"--- OBU 检测与识别工具 {VERSION} ---")
    clear_process_photo_directory(process_photo_dir); print(f"'{process_photo_dir}' 文件夹已清理。")
    if not os.path.exists(ONNX_MODEL_PATH): print(f"错误: 模型未找到: {ONNX_MODEL_PATH}"); exit()
    if not os.path.exists(IMAGE_NAME): print(f"错误: 图片未找到: {IMAGE_NAME}"); exit()
    if not os.path.exists(SERVER_REC_MODEL_DIR_CFG):
        print(f"错误: Server OCR模型目录未找到: {SERVER_REC_MODEL_DIR_CFG}"); exit()

    actual_max_area_threshold_px = None
    try:
        print(f"--- 初始化与模型加载 (ONNX YOLO) ---"); t_start = time.time(); session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider']); timing_profile['1_model_loading'] = time.time() - t_start; print(f"ONNX模型加载完成 ({timing_profile['1_model_loading']:.2f} 秒)")
        input_cfg = session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
        model_input_h_ref, model_input_w_ref = 640, 640
        if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int): model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]
        else: print(f"警告: 模型输入维度包含符号名称: {input_shape_onnx}. 使用参考尺寸 H={model_input_h_ref}, W={model_input_w_ref}")
        class_names = COCO_CLASSES
        print(f"模型输入: {input_name} {input_shape_onnx}. 类别设置为: {class_names}")
        t_start_img_read = time.time(); original_image = cv2.imread(IMAGE_NAME); timing_profile['2_image_reading'] = time.time() - t_start_img_read
        if original_image is None: print(f"错误: 无法读取图片: {IMAGE_NAME} (耗时: {timing_profile['2_image_reading']:.2f} 秒)"); raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")
        orig_img_h, orig_img_w = original_image.shape[:2]; print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['2_image_reading']:.2f} 秒读取)")
        if MAX_DETECTION_AREA is not None:
            if isinstance(MAX_DETECTION_AREA, float) and 0<MAX_DETECTION_AREA<=1.0: actual_max_area_threshold_px = (orig_img_h*orig_img_w)*MAX_DETECTION_AREA; print(f"MAX_DETECTION_AREA设为总面积{MAX_DETECTION_AREA*100:.1f}%, 阈值: {actual_max_area_threshold_px:.0f} px².")
            elif isinstance(MAX_DETECTION_AREA, (int,float)) and MAX_DETECTION_AREA > 1: actual_max_area_threshold_px = float(MAX_DETECTION_AREA); print(f"MAX_DETECTION_AREA设为绝对值: {actual_max_area_threshold_px:.0f} px².")
            else: actual_max_area_threshold_px = None

        apply_tiling = ENABLE_TILING
        use_fixed_grid_tiling = False
        if apply_tiling and FIXED_TILING_GRID is not None and \
           isinstance(FIXED_TILING_GRID, tuple) and len(FIXED_TILING_GRID) == 2 and \
           all(isinstance(n, int) and n > 0 for n in FIXED_TILING_GRID):
            use_fixed_grid_tiling = True; print(f"切块处理: 固定网格 {FIXED_TILING_GRID}. 重叠率: {TILE_OVERLAP_RATIO*100}%")
        elif apply_tiling:
            apply_tiling = (orig_img_w > model_input_w_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING or \
                            orig_img_h > model_input_h_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING)
            print(f"切块处理: {'动态切块' if apply_tiling else '禁用 (尺寸未达动态切块阈值)'}。"
                  f"参考模型输入: {model_input_h_ref}x{model_input_w_ref}, 重叠率: {TILE_OVERLAP_RATIO*100}%")
        else: print(f"切块处理已禁用，将执行整图推理。")

        aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []
        if apply_tiling:
            # (切块逻辑完整代码)
            t_start_tiling_loop = time.time()
            total_inference_time, total_tile_preprocessing_time, total_tile_postprocessing_time, num_tiles_processed = 0,0,0,0
            if use_fixed_grid_tiling:
                num_cols, num_rows = FIXED_TILING_GRID
                nominal_tile_w = orig_img_w / num_cols; nominal_tile_h = orig_img_h / num_rows
                overlap_w_px = int(nominal_tile_w * TILE_OVERLAP_RATIO); overlap_h_px = int(nominal_tile_h * TILE_OVERLAP_RATIO)
                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        num_tiles_processed += 1
                        stride_x = nominal_tile_w if num_cols == 1 else (nominal_tile_w - overlap_w_px); stride_y = nominal_tile_h if num_rows == 1 else (nominal_tile_h - overlap_h_px)
                        current_tile_x0 = int(c_idx * stride_x); current_tile_y0 = int(r_idx * stride_y)
                        current_tile_x1 = int(current_tile_x0 + nominal_tile_w); current_tile_y1 = int(current_tile_y0 + nominal_tile_h)
                        tile_crop_x0 = max(0, current_tile_x0); tile_crop_y0 = max(0, current_tile_y0)
                        tile_crop_x1 = min(orig_img_w, current_tile_x1); tile_crop_y1 = min(orig_img_h, current_tile_y1)
                        tile_data = original_image[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1: continue
                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data_for_tiling(tile_data, (model_input_h_ref, model_input_w_ref)); total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s
                        t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref),ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0); total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box_tile in range(boxes_np.shape[0]): b = boxes_np[i_box_tile]; aggregated_boxes.append([b[0] + tile_crop_x0, b[1] + tile_crop_y0, b[2] + tile_crop_x0, b[3] + tile_crop_y0]); aggregated_scores.append(scores_np[i_box_tile]); aggregated_class_ids.append(c_ids_np[i_box_tile])
            else: # Dynamic Tiling
                tile_w_dyn, tile_h_dyn = model_input_w_ref, model_input_h_ref
                overlap_w_dyn, overlap_h_dyn = int(tile_w_dyn * TILE_OVERLAP_RATIO), int(tile_h_dyn * TILE_OVERLAP_RATIO)
                stride_w_dyn, stride_h_dyn = tile_w_dyn - overlap_w_dyn, tile_h_dyn - overlap_h_dyn
                for y0_dyn in range(0, orig_img_h, stride_h_dyn):
                    for x0_dyn in range(0, orig_img_w, stride_w_dyn):
                        num_tiles_processed += 1
                        x1_dyn, y1_dyn = min(x0_dyn + tile_w_dyn, orig_img_w), min(y0_dyn + tile_h_dyn, orig_img_h)
                        tile_data = original_image[y0_dyn:y1_dyn, x0_dyn:x1_dyn]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1: continue
                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data_for_tiling(tile_data, (model_input_h_ref, model_input_w_ref)); total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s
                        t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref),ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0); total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box_tile in range(boxes_np.shape[0]): b = boxes_np[i_box_tile]; aggregated_boxes.append([b[0] + x0_dyn, b[1] + y0_dyn, b[2] + x0_dyn, b[3] + y0_dyn]); aggregated_scores.append(scores_np[i_box_tile]); aggregated_class_ids.append(c_ids_np[i_box_tile])
            timing_profile['3a_tiling_loop_total (incl_all_tiles_pre_inf_post)'] = time.time() - t_start_tiling_loop
            timing_profile['3b_tiling_total_tile_preprocessing'] = total_tile_preprocessing_time
            timing_profile['3c_tiling_total_tile_inference'] = total_inference_time
            timing_profile['3d_tiling_total_tile_postprocessing'] = total_tile_postprocessing_time
            print(f"切块检测完成 (处理 {num_tiles_processed} 个图块)。")
            if len(aggregated_boxes) > 0:
                t_start_nms = time.time(); keep_indices = non_max_suppression_global(np.array(aggregated_boxes), np.array(aggregated_scores), IOU_THRESHOLD); timing_profile['4a_global_nms'] = time.time() - t_start_nms
                aggregated_boxes = [aggregated_boxes[i] for i in keep_indices]; aggregated_scores = [aggregated_scores[i] for i in keep_indices]; aggregated_class_ids = [aggregated_class_ids[i] for i in keep_indices]
                print(f"全局NMS完成 ({timing_profile['4a_global_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else:
                timing_profile['4a_global_nms'] = 0; aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []; print("切块后未检测到聚合对象，或NMS后无剩余对象。")
        else:
            print("--- 开始整图检测 (使用新ONNX模型适配的预处理和后处理) ---"); t_s = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s; t_s = time.time(); outputs_main = session.run(None, {input_name: input_tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s;
            detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(class_names)); timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s
            aggregated_boxes = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]; aggregated_scores = [d[4] for d in detections_result_list]; aggregated_class_ids = [d[5] for d in detections_result_list]; print(f"整图处理与后处理完成。找到了 {len(aggregated_boxes)} 个框。")

        if len(aggregated_boxes) > 0 and ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or actual_max_area_threshold_px is not None):
            t_start_area_filter=time.time(); filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids=[],[],[]; initial_box_count_before_area_filter=len(aggregated_boxes)
            for i_box,box in enumerate(aggregated_boxes):
                b_w,b_h=box[2]-box[0],box[3]-box[1]; area=b_w*b_h; valid_area=True
                if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA: valid_area=False
                if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid_area=False
                if valid_area: filtered_by_area_boxes.append(box); filtered_by_area_scores.append(aggregated_scores[i_box]); filtered_by_area_ids.append(aggregated_class_ids[i_box])
            aggregated_boxes,aggregated_scores,aggregated_class_ids=filtered_by_area_boxes,filtered_by_area_scores,filtered_by_area_ids; timing_profile['5_area_filtering']=time.time()-t_start_area_filter; print(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框).")
        else:
            timing_profile['5_area_filtering']=0
            if len(aggregated_boxes)>0: print("面积筛选未启用或不适用。")

        ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes)
        recognized_obu_data_list = [None] * len(aggregated_boxes) # Initialize with None

        tasks_for_ocr_parallel = [] # List to hold (original_index, roi_display_index, image_data)

        if len(aggregated_boxes) > 0:
            print(f"--- 最终检测到 {len(aggregated_boxes)} 个OBU的YOLO框, 准备进行OCR预处理 ---")
            for i, yolo_box_coords in enumerate(aggregated_boxes):
                class_id = int(aggregated_class_ids[i]); class_name_str = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"ClassID:{class_id}"
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = [int(c) for c in yolo_box_coords]; h_yolo = y2_yolo - y1_yolo; w_yolo = x2_yolo - x1_yolo
                y1_digit_ideal = y1_yolo + int(h_yolo * DIGIT_ROI_Y_OFFSET_FACTOR); h_digit_ideal = int(h_yolo * DIGIT_ROI_HEIGHT_FACTOR); y2_digit_ideal = y1_digit_ideal + h_digit_ideal
                w_digit_expanded = int(w_yolo * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_yolo = x1_yolo + w_yolo / 2.0; x1_digit_ideal = int(cx_yolo - w_digit_expanded / 2.0); x2_digit_ideal = int(cx_yolo + w_digit_expanded / 2.0)
                y1_d_clip = max(0, y1_digit_ideal); y2_d_clip = min(orig_img_h, y2_digit_ideal); x1_d_clip = max(0, x1_digit_ideal); x2_d_clip = min(orig_img_w, x2_digit_ideal)

                current_box_meta_for_task = {"original_index": i, "roi_index": i + 1, "class": class_name_str, "bbox_yolo": [x1_yolo, y1_yolo, x2_yolo, y2_yolo], "bbox_digit_ocr": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip], "confidence_yolo": float(aggregated_scores[i])}

                image_for_ocr_bgr = None
                dx1,dy1,dx2,dy2=current_box_meta_for_task['bbox_digit_ocr']
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

                tasks_for_ocr_parallel.append( (i, i + 1, image_for_ocr_bgr, current_box_meta_for_task) ) # Add meta here

            if tasks_for_ocr_parallel:
                print(f"\n--- Attempting PARALLEL OCR with Server model for {len(tasks_for_ocr_parallel)} ROIs using {actual_num_workers} worker(s) ---")
                t_ocr_parallel_start = time.time()

                processed_results = [None] * len(tasks_for_ocr_parallel)

                if actual_num_workers > 1:
                    try:
                        # Prepare data for pool.map: only pass what's needed by the worker
                        worker_tasks = [(task[0], task[1], task[2]) for task in tasks_for_ocr_parallel]
                        with multiprocessing.Pool(processes=actual_num_workers) as pool:
                            # pool.map returns results in the same order as input tasks
                            # Each result from worker is (original_index, recognition_dict)
                            parallel_results_with_indices = pool.map(ocr_worker_process_roi_with_logging, worker_tasks)

                        # Reconstruct processed_results based on original_index
                        for res_original_idx, res_dict in parallel_results_with_indices:
                            processed_results[res_original_idx] = res_dict

                    except Exception as e_pool:
                        print(f"      Error during multiprocessing Pool execution: {e_pool}"); traceback.print_exc()
                        print("      FALLING BACK TO SERIAL OCR due to Pool error.")
                        actual_num_workers = 1 # Force serial for fallback

                if actual_num_workers <= 1: # Serial execution (either by config or fallback)
                    ocr_server_predictor_main = None # Initialize for serial
                    try:
                        ocr_server_predictor_main = pdx.inference.create_predictor(
                            model_dir=SERVER_REC_MODEL_DIR_CFG, model_name='PP-OCRv5_server_rec', device='cpu'
                        )
                        print("    Main OCR predictor for serial execution initialized.")
                    except Exception as e_serial_init:
                        print(f"    Error initializing main OCR predictor for serial execution: {e_serial_init}")

                    if ocr_server_predictor_main:
                        for task_idx, task_data_tuple in enumerate(tasks_for_ocr_parallel):
                            _, roi_display_idx, img_data, _ = task_data_tuple
                            if img_data is not None:
                                res_gen = ocr_server_predictor_main.predict([img_data])
                                res_list = next(res_gen, None)
                                if res_list and isinstance(res_list, list) and len(res_list) > 0:
                                    processed_results[task_idx] = res_list[0]
                                elif res_list and isinstance(res_list, dict):
                                    processed_results[task_idx] = res_list
                                else:
                                    processed_results[task_idx] = {'rec_text': '', 'rec_score': 0.0}
                            else: # img_data was None from preprocessing
                                processed_results[task_idx] = {'rec_text': 'PREPROC_FAIL_MAIN', 'rec_score': 0.0}
                    else:
                        print("    Main OCR predictor for serial execution FAILED to initialize.")
                        for task_idx in range(len(tasks_for_ocr_parallel)):
                            processed_results[task_idx] = {'rec_text': 'SERIAL_INIT_FAIL_MAIN', 'rec_score': 0.0}

                timing_profile['7_ocr_processing_total'] = time.time() - t_ocr_parallel_start
                print(f"--- PARALLEL/SERIAL OCR processing finished ({timing_profile['7_ocr_processing_total']:.3f} 秒) ---")

                for task_idx, recognition_result_dict in enumerate(processed_results):
                    _, roi_display_idx, _, current_box_info = tasks_for_ocr_parallel[task_idx] # Get meta back
                    ocr_text_to_draw = "N/A"

                    if recognition_result_dict and isinstance(recognition_result_dict, dict):
                        raw_recognized_text = recognition_result_dict.get('rec_text', "")
                        ocr_score = recognition_result_dict.get('rec_score', 0.0)
                        worker_pid = recognition_result_dict.get('pid', 'N/A_PID') # Get PID if returned by worker
                        duration = recognition_result_dict.get('duration', -1) # Get duration if returned

                        if raw_recognized_text and raw_recognized_text not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL_MAIN', 'SERIAL_INIT_FAIL_MAIN']:
                            digits_only_text = "".join(re.findall(r'\d', raw_recognized_text))
                            # Log includes PID and duration if from parallel worker
                            # print(f"  ROI {roi_display_idx} (PID:{worker_pid}, Dur:{duration:.3f}s) OCR: Raw='{raw_recognized_text}', Digits='{digits_only_text}', Score={ocr_score:.2f}")
                            if digits_only_text:
                                ocr_text_to_draw = digits_only_text
                                current_box_info["ocr_final_text"] = digits_only_text
                                current_box_info["ocr_confidence"] = ocr_score
                            # else: print(f"    OCR Server: No digits extracted from '{raw_recognized_text}'.") # Already printed by worker
                        # else: print(f"  ROI {roi_display_idx} OCR Server: No raw text or worker error ('{raw_recognized_text}').")
                    # else: print(f"  ROI {roi_display_idx} OCR Server: Invalid result format: {recognition_result_dict}")

                    ocr_texts_for_drawing[current_box_info['original_index']] = ocr_text_to_draw
                    recognized_obu_data_list[current_box_info['original_index']] = current_box_info

            else:
                print("No valid images preprocessed for OCR.")
                timing_profile['7_ocr_processing_total'] = 0
                # Populate recognized_obu_data_list with N/A if no OCR was run
                for i, yolo_box_coords in enumerate(aggregated_boxes):
                    # Simplified population for brevity, ensure all fields are present as in current_box_info
                    recognized_obu_data_list[i] = {"roi_index": i + 1, "ocr_final_text": "N/A", "ocr_confidence": 0.0,
                                                   "bbox_yolo": yolo_box_coords, "class": class_names[aggregated_class_ids[i]],
                                                   "confidence_yolo": aggregated_scores[i]}


            print("\n--- 初步识别结果列表 (未映射到矩阵) ---");
            for obu_data in recognized_obu_data_list:
                if obu_data and obu_data["ocr_final_text"]!="N/A": print(f"ROI {obu_data['roi_index']} (YOLO BBox: {obu_data['bbox_yolo']}): Digit OCR Text = {obu_data['ocr_final_text']}")
            print("---------------------------------------\n")

            t_start_drawing=time.time(); output_img_to_draw_on=original_image.copy();
            # Ensure recognized_obu_data_list is correctly populated for drawing
            valid_roi_indices_for_drawing = [item["roi_index"] for item in recognized_obu_data_list if item]
            output_img_to_draw_on=draw_detections(output_img_to_draw_on,np.array(aggregated_boxes),np.array(aggregated_scores),np.array(aggregated_class_ids),class_names,ocr_texts=ocr_texts_for_drawing,roi_indices=valid_roi_indices_for_drawing);
            timing_profile['8_drawing_results_final']=time.time()-t_start_drawing;
            output_fn_base=os.path.splitext(os.path.basename(IMAGE_NAME))[0];
            final_output_image_name = f"final_annotated_{output_fn_base}_{VERSION}.png";
            final_output_path = os.path.join(process_photo_dir, final_output_image_name);
            cv2.imwrite(final_output_path, output_img_to_draw_on);
            print(f"最终结果图已保存到: {final_output_path} ({timing_profile['8_drawing_results_final']:.2f} 秒用于绘图)")
        else:
            print("最终未检测到任何OBU ROI，无法进行OCR。"); timing_profile['7_ocr_processing_total']=0; timing_profile['8_drawing_results_final']=0
    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"发生错误: {e}"); traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 ({VERSION}) ---"); sorted_timing_profile = {k: timing_profile[k] for k in sorted(timing_profile.keys())}
        for stage, duration in sorted_timing_profile.items(): print(f"  {stage}: {duration:.3f} 秒")
        print(f"------------------------------")