import cv2
import numpy as np
import onnxruntime
import os
import time # 用于计时
import traceback # 用于打印详细错误信息
import paddle # <--- 新增导入
import paddleocr # <--- 新增导入

# --- PaddleOCR引擎初始化 ---
paddle_ocr_engine = None
try:
    print("Initializing PaddleOCR engine globally...")
    # lang='en' 适用于英文和数字。
    paddle_ocr_engine = paddleocr.PaddleOCR(lang='en')
    print("PaddleOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR engine globally: {e}")
    print("PaddleOCR features will be disabled.")
    paddle_ocr_engine = None

# --- V2.3 配置参数 (基于V2.2) ---
VERSION = "v2.3.0_paddleocr" # 版本号更新
ONNX_MODEL_PATH = r"./model/BarCode_Detect/yolov8_barcode_detection.onnx"
IMAGE_NAME = r"./PIC/1.JPG"
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

ENABLE_TILING = True
FIXED_TILING_GRID = None
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5
TILE_OVERLAP_RATIO = 0.2

MIN_DETECTION_AREA = 9000
MAX_DETECTION_AREA = 0.01

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

timing_profile = {}

def preprocess_image_data(img_data, input_shape_hw):
    img = img_data
    if img is None: raise ValueError("输入图像数据为空")
    img_height, img_width = img.shape[:2]
    input_height, input_width = input_shape_hw
    ratio = min(input_width / img_width, input_height / img_height)
    new_width, new_height = int(img_width * ratio), int(img_height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 128, dtype=np.uint8)
    x_pad, y_pad = (input_width - new_width) // 2, (input_height - new_height) // 2
    canvas[y_pad:y_pad + new_height, x_pad:x_pad + new_width] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), ratio, x_pad, y_pad

def non_max_suppression(boxes, scores, iou_threshold):
    if not isinstance(boxes, np.ndarray): boxes = np.array(boxes)
    if not isinstance(scores, np.ndarray): scores = np.array(scores)
    if boxes.shape[0] == 0: return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1: break
        remaining_indices = indices[1:]
        x1 = np.maximum(boxes[i, 0], boxes[remaining_indices, 0])
        y1 = np.maximum(boxes[i, 1], boxes[remaining_indices, 1])
        x2 = np.minimum(boxes[i, 2], boxes[remaining_indices, 2])
        y2 = np.minimum(boxes[i, 3], boxes[remaining_indices, 3])
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rem = (boxes[remaining_indices, 2] - boxes[remaining_indices, 0]) * \
                   (boxes[remaining_indices, 3] - boxes[remaining_indices, 1])
        union_area = area_i + area_rem - inter_area
        iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area, dtype=float), where=union_area!=0)
        indices = remaining_indices[np.where(iou <= iou_threshold)[0]]
    return keep

# postprocess_detections_from_tile 函数保持与您V2.2版本一致
def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold, model_output_channels_param):
    predictions_raw = np.squeeze(outputs[0])

    if predictions_raw.ndim != 2:
        print(f"错误: Postprocess Squeezed predictions_raw 维度不是2，实际为 {predictions_raw.ndim}, shape: {predictions_raw.shape}")
        return np.array([]), np.array([]), np.array([])

    actual_model_output_channels = predictions_raw.shape[0]
    if not isinstance(actual_model_output_channels, int):
         print(f"错误: Postprocess actual_model_output_channels 不是整数: {actual_model_output_channels}")
         return np.array([]), np.array([]), np.array([])

    transposed_predictions = predictions_raw.transpose()
    boxes_tile_local_scaled, scores_tile_local, class_ids_tile_local = [], [], []

    for pred_data in transposed_predictions:
        if len(pred_data) != actual_model_output_channels: continue

        cx, cy, w, h = pred_data[:4]
        confidence, class_id = 0.0, -1

        # 使用从实际输出推断的 actual_model_output_channels
        if actual_model_output_channels == 6:
            confidence = pred_data[4]
            class_id = int(pred_data[5])
        elif actual_model_output_channels == 5:
            confidence = pred_data[4]
            class_id = 0
        elif actual_model_output_channels > 4 :
            class_scores = pred_data[4:]
            if class_scores.size > 0:
                confidence = np.max(class_scores)
                class_id = np.argmax(class_scores)
            else: continue
        else: continue

        if confidence >= conf_threshold:
            x1, y1 = (cx - w / 2), (cy - h / 2)
            x2, y2 = (cx + w / 2), (cy + h / 2)
            boxes_tile_local_scaled.append([x1, y1, x2, y2])
            scores_tile_local.append(confidence)
            class_ids_tile_local.append(class_id)

    if not boxes_tile_local_scaled: return np.array([]), np.array([]), np.array([])
    final_boxes_tile_original_coords = []
    tile_h_orig, tile_w_orig = tile_original_shape_hw
    for box in boxes_tile_local_scaled:
        b_x1, b_y1, b_x2, b_y2 = box[0] - preprocessing_pad_x, box[1] - preprocessing_pad_y, \
                                 box[2] - preprocessing_pad_x, box[3] - preprocessing_pad_y
        if preprocessing_ratio == 0: continue # 避免除以0
        ot_x1, ot_y1 = b_x1 / preprocessing_ratio, b_y1 / preprocessing_ratio
        ot_x2, ot_y2 = b_x2 / preprocessing_ratio, b_y2 / preprocessing_ratio
        ot_x1, ot_y1 = np.clip(ot_x1, 0, tile_w_orig), np.clip(ot_y1, 0, tile_h_orig)
        ot_x2, ot_y2 = np.clip(ot_x2, 0, tile_w_orig), np.clip(ot_y2, 0, tile_h_orig)
        final_boxes_tile_original_coords.append([ot_x1, ot_y1, ot_x2, ot_y2])
    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        score = scores[i]
        class_id = int(class_ids[i])

        label_name = f"ClassID:{class_id}"
        if class_names and 0 <= class_id < len(class_names):
            label_name = class_names[class_id]

        yolo_label_text = f"{label_name}: {score:.2f}"

        # 绘制YOLO ROI框 (绿色)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制YOLO标签文本 (绿色)
        cv2.putText(img_out, yolo_label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制ROI索引 (红色)
        if roi_indices and i < len(roi_indices):
            cv2.putText(img_out, f"ROI:{roi_indices[i]}", (x1 + 5, y1 + 20), # 调整y位置避免与YOLO标签重叠
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 绘制OCR文本 (蓝色)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A":
            ocr_text_to_display = ocr_texts[i]
            # 将OCR文本绘制在ROI框的更上方，避免与YOLO标签和ROI索引重叠
            cv2.putText(img_out, ocr_text_to_display, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img_out

# --- 新增：清理过程图片文件夹的函数 ---
def clear_process_photo_directory(directory="process_photo"):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): # 如果需要递归删除子文件夹
                #     shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory, exist_ok=True)

# --- V2.3.2 主程序 (尝试多种预处理送OCR，清理过程图片) ---
if __name__ == "__main__":
    t_start_overall = time.time()
    timing_profile['0_total_script_execution'] = 0
    print(f"--- OBU 检测与识别工具 {VERSION} (尝试多种预处理) ---")

    # 在开始时清理过程图片文件夹
    process_photo_dir = "process_photo"
    clear_process_photo_directory(process_photo_dir)
    print(f"'{process_photo_dir}' 文件夹已清理。")


    if not os.path.exists(ONNX_MODEL_PATH): print(f"错误: 模型未找到: {ONNX_MODEL_PATH}"); exit()
    if not os.path.exists(IMAGE_NAME): print(f"错误: 图片未找到: {IMAGE_NAME}"); exit()

    actual_max_area_threshold_px = None
    # process_photo_dir 已在上面定义和创建

    try:
        # ... (模型加载，图像读取，参数设置，切块逻辑判断 - 与V2.3一致) ...
        # ... (确保这里的 model_input_h_ref, model_input_w_ref, class_names 正确设置) ...
        # ... (确保这里的 MAX_DETECTION_AREA, apply_tiling, use_fixed_grid_tiling 正确设置) ...

        # --- START OF ACTUAL TILING OR FULL IMAGE PROCESSING LOGIC ---
        # !!! 【重要】请从您能工作的V2.3版本中，将从这里开始的完整切块/整图处理逻辑复制过来 !!!
        # !!! 直到 "# --- END OF ACTUAL TILING OR FULL IMAGE PROCESSING LOGIC ---" 标记处 !!!
        # !!! 这段代码负责填充 aggregated_boxes, aggregated_scores, aggregated_class_ids !!!
        print(f"--- 初始化与模型加载 ---") # 这部分日志也应在您的代码中
        t_start = time.time(); session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider']); timing_profile['1_model_loading'] = time.time() - t_start; print(f"ONNX模型加载完成 ({timing_profile['1_model_loading']:.2f} 秒)")
        input_cfg = session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
        model_input_h_ref, model_input_w_ref = 640, 640
        if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int): model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]
        else: print(f"警告: 模型输入维度包含符号名称: {input_shape_onnx}. 使用参考尺寸 H={model_input_h_ref}, W={model_input_w_ref}")
        class_names = ["Barcode"]; print(f"模型输入: {input_name} {input_shape_onnx}. 类别设置为: {class_names}")
        t_start = time.time(); original_image = cv2.imread(IMAGE_NAME); timing_profile['2_image_reading'] = time.time() - t_start; orig_img_h, orig_img_w = original_image.shape[:2]; print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['2_image_reading']:.2f} 秒读取)")
        if MAX_DETECTION_AREA is not None:
            if isinstance(MAX_DETECTION_AREA, float) and 0 < MAX_DETECTION_AREA <= 1.0: actual_max_area_threshold_px = (orig_img_h * orig_img_w) * MAX_DETECTION_AREA; print(f"MAX_DETECTION_AREA 设置为图像总面积的 {MAX_DETECTION_AREA*100:.1f}%, 最大像素面积阈值: {actual_max_area_threshold_px:.0f} px².")
            elif isinstance(MAX_DETECTION_AREA, (int, float)) and MAX_DETECTION_AREA > 1: actual_max_area_threshold_px = float(MAX_DETECTION_AREA); print(f"MAX_DETECTION_AREA 设置为绝对像素面积阈值: {actual_max_area_threshold_px:.0f} px².")
            else: actual_max_area_threshold_px = None
        apply_tiling = ENABLE_TILING; use_fixed_grid_tiling = False
        if apply_tiling and FIXED_TILING_GRID is not None and isinstance(FIXED_TILING_GRID, tuple) and len(FIXED_TILING_GRID) == 2 and all(isinstance(n, int) and n > 0 for n in FIXED_TILING_GRID): use_fixed_grid_tiling = True; print(f"切块处理: 固定网格 {FIXED_TILING_GRID} (列,行)。重叠率: {TILE_OVERLAP_RATIO*100}%")
        elif apply_tiling: apply_tiling = (orig_img_w > model_input_w_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING or orig_img_h > model_input_h_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING); print(f"切块处理: {'动态切块' if apply_tiling else '禁用 (尺寸未达动态切块阈值)'}。参考模型输入: {model_input_h_ref}x{model_input_w_ref}, 重叠率: {TILE_OVERLAP_RATIO*100}%")
        else: print(f"切块处理禁用 (全局配置)。")
        aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []; total_inference_time, total_tile_preprocessing_time, total_tile_postprocessing_time = 0,0,0; num_tiles_processed = 0
        if apply_tiling:
            t_start_tiling_loop = time.time()
            if use_fixed_grid_tiling:
                num_cols, num_rows = FIXED_TILING_GRID; nominal_tile_w = orig_img_w / num_cols; nominal_tile_h = orig_img_h / num_rows; overlap_w_px = int(nominal_tile_w * TILE_OVERLAP_RATIO); overlap_h_px = int(nominal_tile_h * TILE_OVERLAP_RATIO)
                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        num_tiles_processed += 1; stride_x = nominal_tile_w if num_cols == 1 else (nominal_tile_w - overlap_w_px); stride_y = nominal_tile_h if num_rows == 1 else (nominal_tile_h - overlap_h_px); current_tile_x0 = int(c_idx * stride_x); current_tile_y0 = int(r_idx * stride_y); current_tile_x1 = int(current_tile_x0 + nominal_tile_w); current_tile_y1 = int(current_tile_y0 + nominal_tile_h); tile_crop_x0 = max(0, current_tile_x0); tile_crop_y0 = max(0, current_tile_y0); tile_crop_x1 = min(orig_img_w, current_tile_x1); tile_crop_y1 = min(orig_img_h, current_tile_y1); tile_data = original_image[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]; tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1: continue
                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(tile_data, (model_input_h_ref, model_input_w_ref)); total_tile_preprocessing_time += time.time() - t_s; t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s; t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0); total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box in range(boxes_np.shape[0]): b = boxes_np[i_box]; aggregated_boxes.append([b[0] + tile_crop_x0, b[1] + tile_crop_y0, b[2] + tile_crop_x0, b[3] + tile_crop_y0]); aggregated_scores.append(scores_np[i_box]); aggregated_class_ids.append(c_ids_np[i_box])
            else: # Dynamic Tiling
                tile_w_dyn, tile_h_dyn = model_input_w_ref, model_input_h_ref; overlap_w_dyn, overlap_h_dyn = int(tile_w_dyn * TILE_OVERLAP_RATIO), int(tile_h_dyn * TILE_OVERLAP_RATIO); stride_w_dyn, stride_h_dyn = tile_w_dyn - overlap_w_dyn, tile_h_dyn - overlap_h_dyn
                for y0_dyn in range(0, orig_img_h, stride_h_dyn):
                    for x0_dyn in range(0, orig_img_w, stride_w_dyn):
                        num_tiles_processed += 1; x1_dyn, y1_dyn = min(x0_dyn + tile_w_dyn, orig_img_w), min(y0_dyn + tile_h_dyn, orig_img_h); tile_data = original_image[y0_dyn:y1_dyn, x0_dyn:x1_dyn]; tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1: continue
                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(tile_data, (model_input_h_ref, model_input_w_ref)); total_tile_preprocessing_time += time.time() - t_s; t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s; t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0); total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box in range(boxes_np.shape[0]): b = boxes_np[i_box]; aggregated_boxes.append([b[0] + x0_dyn, b[1] + y0_dyn, b[2] + x0_dyn, b[3] + y0_dyn]); aggregated_scores.append(scores_np[i_box]); aggregated_class_ids.append(c_ids_np[i_box])
            timing_profile['3a_tiling_loop_total (incl_all_tiles_pre_inf_post)'] = time.time() - t_start_tiling_loop; timing_profile['3b_tiling_total_tile_preprocessing'] = total_tile_preprocessing_time; timing_profile['3c_tiling_total_tile_inference'] = total_inference_time; timing_profile['3d_tiling_total_tile_postprocessing'] = total_tile_postprocessing_time; print(f"切块检测完成 (处理 {num_tiles_processed} 个图块)。")
            if len(aggregated_boxes) > 0: t_start_nms = time.time(); keep_indices = non_max_suppression(np.array(aggregated_boxes), np.array(aggregated_scores), IOU_THRESHOLD); timing_profile['4a_global_nms'] = time.time() - t_start_nms; aggregated_boxes = [aggregated_boxes[i] for i in keep_indices]; aggregated_scores = [aggregated_scores[i] for i in keep_indices]; aggregated_class_ids = [aggregated_class_ids[i] for i in keep_indices]; print(f"全局NMS完成 ({timing_profile['4a_global_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else: timing_profile['4a_global_nms'] = 0; aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []; print("切块后未检测到聚合对象，或NMS后无剩余对象。")
        else:
            print("--- 开始整图检测 (切块未启用或不适用) ---"); t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s; t_s = time.time(); outputs = session.run(None, {input_name: tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s; t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (orig_img_h, orig_img_w), (model_input_h_ref, model_input_w_ref), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0); timing_profile['3c_fullimg_postprocessing_no_nms'] = time.time() - t_s
            if boxes_np.shape[0] > 0: t_s_nms = time.time(); keep_indices = non_max_suppression(boxes_np, scores_np, IOU_THRESHOLD); timing_profile['4a_fullimg_nms'] = time.time() - t_s_nms; aggregated_boxes = [boxes_np[i] for i in keep_indices]; aggregated_scores = [scores_np[i] for i in keep_indices]; aggregated_class_ids = [c_ids_np[i] for i in keep_indices]
            num_tiles_processed = 1; print(f"整图NMS完成。找到了 {len(aggregated_boxes)} 个框。")
        # --- END OF ACTUAL TILING OR FULL IMAGE PROCESSING LOGIC ---

        # --- 面积筛选 ---
        if len(aggregated_boxes) > 0 and ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or actual_max_area_threshold_px is not None):
            t_start_area_filter = time.time(); filtered_by_area_boxes, filtered_by_area_scores, filtered_by_area_ids = [], [], []; initial_box_count_before_area_filter = len(aggregated_boxes)
            for i, box in enumerate(aggregated_boxes):
                b_w, b_h = box[2] - box[0], box[3] - box[1]; area = b_w * b_h; valid_area = True
                if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA: valid_area = False
                if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid_area = False
                if valid_area: filtered_by_area_boxes.append(box); filtered_by_area_scores.append(aggregated_scores[i]); filtered_by_area_ids.append(aggregated_class_ids[i])
            aggregated_boxes, aggregated_scores, aggregated_class_ids = filtered_by_area_boxes, filtered_by_area_scores, filtered_by_area_ids; timing_profile['5_area_filtering'] = time.time() - t_start_area_filter
            print(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框).")
        else:
            timing_profile['5_area_filtering'] = 0
            if len(aggregated_boxes) > 0: print("面积筛选未启用或不适用。")

        # --- 可视化与输出 (包含多种预处理的OCR尝试) ---
        ocr_texts_for_drawing = []
        recognized_obu_data_list = []

        if len(aggregated_boxes) > 0:
            print(f"--- 最终检测到 {len(aggregated_boxes)} 个OBU ROI, 开始OCR识别 ---")
            t_ocr_total_start = time.time()

            for i, box_coords in enumerate(aggregated_boxes):
                class_id = int(aggregated_class_ids[i])
                class_name_str = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"ClassID:{class_id}"

                current_box_info = {
                    "roi_index": i + 1, "class": class_name_str,
                    "bbox": [int(c) for c in box_coords],
                    "confidence_yolo": float(aggregated_scores[i]),
                    "ocr_text_color": "N/A", "ocr_text_gray": "N/A", "ocr_text_binary": "N/A",
                    "ocr_final_text": "N/A", # 用于最终选择的文本
                    "ocr_confidence": 0.0
                }
                print(f"  OBU ROI {current_box_info['roi_index']}: 类别='{current_box_info['class']}', "
                      f"边界框={current_box_info['bbox']}, YOLO置信度={current_box_info['confidence_yolo']:.2f}")

                ocr_text_to_draw = "N/A"
                if paddle_ocr_engine:
                    x1_roi, y1_roi, x2_roi, y2_roi = current_box_info['bbox']
                    crop_x1, crop_y1 = max(0, x1_roi), max(0, y1_roi)
                    crop_x2, crop_y2 = min(original_image.shape[1], x2_roi), min(original_image.shape[0], y2_roi)

                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        current_roi_image_color = original_image[crop_y1:crop_y2, crop_x1:crop_x2]
                        cv2.imwrite(os.path.join(process_photo_dir, f"roi_{current_box_info['roi_index']:03d}_color.png"), current_roi_image_color)

                        current_roi_gray = cv2.cvtColor(current_roi_image_color, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(os.path.join(process_photo_dir, f"roi_{current_box_info['roi_index']:03d}_gray.png"), current_roi_gray)

                        # OTSU 二值化 (INV是因为镭射字通常比背景亮，目标是字白底黑)
                        _, current_roi_binary_otsu = cv2.threshold(current_roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        cv2.imwrite(os.path.join(process_photo_dir, f"roi_{current_box_info['roi_index']:03d}_binary_otsu.png"), current_roi_binary_otsu)

                        # 自适应二值化 (也尝试INV)
                        current_roi_binary_adaptive = cv2.adaptiveThreshold(current_roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                          cv2.THRESH_BINARY_INV, 11, 2) # block_size=11, C=2 是常用参数
                        cv2.imwrite(os.path.join(process_photo_dir, f"roi_{current_box_info['roi_index']:03d}_binary_adaptive.png"), current_roi_binary_adaptive)

                        images_to_ocr = {
                            "color": current_roi_image_color,
                            "gray": current_roi_gray,
                            "binary_otsu": current_roi_binary_otsu,
                            "binary_adaptive": current_roi_binary_adaptive
                        }

                        best_ocr_text_for_this_roi = "N/A"
                        highest_ocr_confidence = 0.0

                        for preprocess_type, image_to_process in images_to_ocr.items():
                            print(f"    Attempting OCR with {preprocess_type} preprocessed image for ROI {current_box_info['roi_index']}...")
                            try:
                                ocr_result_list = paddle_ocr_engine.ocr(image_to_process)
                                if ocr_result_list and isinstance(ocr_result_list, list) and \
                                   len(ocr_result_list) > 0 and isinstance(ocr_result_list[0], dict):
                                    image_result_dict = ocr_result_list[0]
                                    extracted_texts_list = image_result_dict.get('rec_texts', [])
                                    rec_scores_list = image_result_dict.get('rec_scores', [])
                                    if extracted_texts_list:
                                        full_recognized_text = "".join(extracted_texts_list).replace(" ", "")
                                        ocr_confidence = rec_scores_list[0] if rec_scores_list else 0.0
                                        print(f"      PaddleOCR ({preprocess_type}) Result: '{full_recognized_text}' (Conf: {ocr_confidence:.2f})")
                                        current_box_info[f"ocr_text_{preprocess_type}"] = full_recognized_text
                                        # 简单决策：取第一个成功识别且置信度最高的，或者最长的数字串等
                                        # 这里我们先简单取第一个成功识别的作为最终的，后续可以优化决策逻辑
                                        if best_ocr_text_for_this_roi == "N/A": # 或者根据置信度选择
                                            best_ocr_text_for_this_roi = full_recognized_text
                                            highest_ocr_confidence = ocr_confidence
                                        elif ocr_confidence > highest_ocr_confidence: # 如果后续有更高置信度的
                                            best_ocr_text_for_this_roi = full_recognized_text
                                            highest_ocr_confidence = ocr_confidence

                                    else: print(f"      PaddleOCR ({preprocess_type}): No text recognized.")
                                else: print(f"      PaddleOCR ({preprocess_type}): No text or unexpected result.")
                            except Exception as ocr_e: print(f"      Error during PaddleOCR ({preprocess_type}): {ocr_e}")

                        current_box_info["ocr_final_text"] = best_ocr_text_for_this_roi
                        current_box_info["ocr_confidence"] = highest_ocr_confidence
                        ocr_text_to_draw = best_ocr_text_for_this_roi

                    else: print(f"    Skipping OCR for invalid ROI (zero size).")
                else: print("    PaddleOCR engine not available.")
                recognized_obu_data_list.append(current_box_info)
                ocr_texts_for_drawing.append(ocr_text_to_draw)
                print("-" * 30)

            timing_profile['7_ocr_processing_total'] = time.time() - t_ocr_total_start
            print(f"--- 所有ROI的OCR处理完成 ({timing_profile['7_ocr_processing_total']:.3f} 秒) ---")

            # --- 矩阵反馈 (初步实现) ---
            # 假设我们知道目标的行列数，例如 TARGET_ROWS, TARGET_COLS
            # 这里需要一个逻辑将 recognized_obu_data_list 中的结果映射到矩阵
            # 为了演示，我们先简单打印识别到的文本列表
            print("\n--- 初步识别结果列表 (未映射到矩阵) ---")
            for obu_data in recognized_obu_data_list:
                if obu_data["ocr_final_text"] != "N/A":
                    print(f"ROI {obu_data['roi_index']} (BBox: {obu_data['bbox']}): {obu_data['ocr_final_text']}")
            print("---------------------------------------\n")


            t_start_drawing = time.time()
            output_img_to_draw_on = original_image.copy()
            output_img_to_draw_on = draw_detections(
                output_img_to_draw_on,
                np.array(aggregated_boxes), np.array(aggregated_scores),
                np.array(aggregated_class_ids), class_names,
                ocr_texts=ocr_texts_for_drawing, # 使用最终选择的OCR文本进行绘制
                roi_indices=[item["roi_index"] for item in recognized_obu_data_list]
            )
            timing_profile['8_drawing_results_final'] = time.time() - t_start_drawing
            output_fn_base = os.path.splitext(os.path.basename(IMAGE_NAME))[0]
            output_fn = f"output_{output_fn_base}_{VERSION}_ocr.png"
            cv2.imwrite(output_fn, output_img_to_draw_on)
            print(f"最终结果图已保存到: {output_fn} ({timing_profile['8_drawing_results_final']:.2f} 秒用于绘图)")
        else:
            print("最终未检测到任何OBU ROI，无法进行OCR。")
            timing_profile['7_ocr_processing_total'] = 0
            timing_profile['8_drawing_results_final'] = 0

    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"发生错误: {e}"); traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 ({VERSION}) ---")
        sorted_timing_profile = {k: timing_profile[k] for k in sorted(timing_profile.keys())}
        for stage, duration in sorted_timing_profile.items(): print(f"  {stage}: {duration:.3f} 秒")
        print(f"------------------------------")
