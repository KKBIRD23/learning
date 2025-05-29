import cv2
import numpy as np
import onnxruntime
import os
import time # 用于计时
import traceback # 用于打印详细错误信息

# --- V2.2 配置参数 ---
ONNX_MODEL_PATH = r"./model/yolov8/yolov8_barcode_detection.onnx"
IMAGE_NAME = r"./all_photo/IMG_0593.jpg" # 请确保这张图片存在
CONFIDENCE_THRESHOLD = 0.15 # 置信度阈值
IOU_THRESHOLD = 0.45 # 用于NMS的IoU阈值

# --- 切块逻辑配置 (V2.2 更新) ---
ENABLE_TILING = True
# 新增：固定网格切块配置。如果设置此项 (例如 (2,1) 表示2列1行)，则优先使用固定网格切块。
# 设置为 None 则使用下面的动态切块逻辑。
FIXED_TILING_GRID = None  # 示例: (2, 1) 切成2块; (3, 2) 切成6块; (1,1) 等同于不切但仍走切块流程

# 动态切块逻辑参数 (仅当 FIXED_TILING_GRID 为 None 时生效)
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5 # 图像任一维度 > 模型输入对应维度 * 此因子时，启用动态切块
# 切块重叠比例 (对动态切块和固定网格切块均有效)
TILE_OVERLAP_RATIO = 0.2

# --- V2.1 新增：检测结果面积筛选配置 ---
MIN_DETECTION_AREA = 9000
MAX_DETECTION_AREA = 0.01 # 示例值：过滤掉面积大于图像总面积 25% 的检测

# COCO 数据集80个类别名称
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

def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold, model_output_channels):
    predictions_raw = np.squeeze(outputs[0])
    num_features_per_pred = predictions_raw.shape[0]
    transposed_predictions = predictions_raw.transpose()
    boxes_tile_local_scaled, scores_tile_local, class_ids_tile_local = [], [], []

    for pred_data in transposed_predictions:
        cx, cy, w, h = pred_data[:4]
        confidence, class_id = 0.0, -1
        if model_output_channels == 6:
            confidence = pred_data[4]
            class_id = int(pred_data[5])
        elif model_output_channels == 5:
            confidence = pred_data[4]
            class_id = 0
        elif model_output_channels > 4 :
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
        ot_x1, ot_y1 = b_x1 / preprocessing_ratio, b_y1 / preprocessing_ratio
        ot_x2, ot_y2 = b_x2 / preprocessing_ratio, b_y2 / preprocessing_ratio
        ot_x1, ot_y1 = np.clip(ot_x1, 0, tile_w_orig), np.clip(ot_y1, 0, tile_h_orig)
        ot_x2, ot_y2 = np.clip(ot_x2, 0, tile_w_orig), np.clip(ot_y2, 0, tile_h_orig)
        final_boxes_tile_original_coords.append([ot_x1, ot_y1, ot_x2, ot_y2])
    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)

def draw_detections(image, boxes, scores, class_ids, class_names=None):
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        score, class_id = scores[i], int(class_ids[i])
        label_name = f"ClassID:{class_id}"
        if class_names and 0 <= class_id < len(class_names):
            label_name = class_names[class_id]
        elif not class_names:
             label_name = f"Obj:{class_id}"
        label_text = f"{label_name}: {score:.2f}"
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_out

# --- V2.2 主程序 ---
if __name__ == "__main__":
    t_start_overall = time.time()
    timing_profile['0_total_script_execution'] = 0

    if not os.path.exists(ONNX_MODEL_PATH): print(f"错误: 模型未找到: {ONNX_MODEL_PATH}"); exit()
    if not os.path.exists(IMAGE_NAME): print(f"错误: 图片未找到: {IMAGE_NAME}"); exit()

    actual_max_area_threshold_px = None

    try:
        print(f"--- 初始化与模型加载 (V2.2) ---")
        t_start = time.time()
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        timing_profile['1_model_loading'] = time.time() - t_start
        print(f"模型加载完成 ({timing_profile['1_model_loading']:.2f} 秒)")

        input_cfg = session.get_inputs()[0]
        input_name, input_shape_onnx = input_cfg.name, input_cfg.shape
        model_input_h, model_input_w = input_shape_onnx[2], input_shape_onnx[3]
        model_num_output_features = session.get_outputs()[0].shape[1]
        print(f"模型输入: {input_name} {input_shape_onnx}, 模型原始输出特征数: {model_num_output_features}")

        if model_num_output_features == 6: class_names = ["Barcode"]
        elif model_num_output_features == 5: class_names = ["Barcode"]
        elif model_num_output_features == (4 + len(COCO_CLASSES)): class_names = COCO_CLASSES
        elif model_num_output_features > 4:
            num_classes_derived = model_num_output_features - 4
            class_names = [f"Class_{i}" for i in range(num_classes_derived)]
        else: class_names = []; print(f"警告: 未能推断类别名称。")

        effective_model_output_channels_for_postproc = model_num_output_features
        model_type_msg = "特定条形码模型" if class_names == ["Barcode"] else \
                         f"COCO ({len(COCO_CLASSES)}类)" if class_names == COCO_CLASSES and len(COCO_CLASSES)>0 else \
                         f"自定义 ({len(class_names)}类)" if class_names else "未知模型类型"
        print(f"推断模型类型: {model_type_msg}, 类别数量: {len(class_names)}")

        t_start = time.time()
        original_image = cv2.imread(IMAGE_NAME)
        if original_image is None: raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")
        timing_profile['2_image_reading'] = time.time() - t_start
        orig_img_h, orig_img_w = original_image.shape[:2]
        print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['2_image_reading']:.2f} 秒读取)")

        if MAX_DETECTION_AREA is not None:
            if isinstance(MAX_DETECTION_AREA, float) and 0 < MAX_DETECTION_AREA <= 1.0:
                actual_max_area_threshold_px = (orig_img_h * orig_img_w) * MAX_DETECTION_AREA
                print(f"MAX_DETECTION_AREA 设置为图像总面积的 {MAX_DETECTION_AREA*100:.1f}%, 最大像素面积阈值: {actual_max_area_threshold_px:.0f} px².")
            elif isinstance(MAX_DETECTION_AREA, (int, float)) and MAX_DETECTION_AREA > 1:
                actual_max_area_threshold_px = float(MAX_DETECTION_AREA)
                print(f"MAX_DETECTION_AREA 设置为绝对像素面积阈值: {actual_max_area_threshold_px:.0f} px².")
            else: actual_max_area_threshold_px = None; print(f"警告: MAX_DETECTION_AREA 值无效。")
        else: print("MAX_DETECTION_AREA 未设置，不进行最大面积筛选。")

        apply_tiling = ENABLE_TILING
        use_fixed_grid_tiling = False
        if apply_tiling and FIXED_TILING_GRID is not None and \
           isinstance(FIXED_TILING_GRID, tuple) and len(FIXED_TILING_GRID) == 2 and \
           all(isinstance(n, int) and n > 0 for n in FIXED_TILING_GRID):
            use_fixed_grid_tiling = True
            print(f"切块处理已启用: 固定网格切块 {FIXED_TILING_GRID} (列,行)。重叠率: {TILE_OVERLAP_RATIO*100}%")
        elif apply_tiling: # Dynamic tiling
            apply_tiling = (orig_img_w > model_input_w * MIN_IMAGE_DIM_FACTOR_FOR_TILING or \
                            orig_img_h > model_input_h * MIN_IMAGE_DIM_FACTOR_FOR_TILING)
            if apply_tiling:
                print(f"切块处理已启用: 动态切块。模型输入尺寸: {model_input_h}x{model_input_w}, 重叠率: {TILE_OVERLAP_RATIO*100}%")
            else:
                print(f"切块处理已禁用 (图像尺寸未达动态切块阈值或配置禁用)。")
        else:
            print(f"切块处理已禁用 (全局配置禁用)。")


        aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []
        total_inference_time, total_tile_preprocessing_time, total_tile_postprocessing_time = 0, 0, 0
        num_tiles_processed = 0

        if apply_tiling:
            t_start_tiling_loop = time.time()

            if use_fixed_grid_tiling:
                num_cols, num_rows = FIXED_TILING_GRID

                # 计算每个名义图块的尺寸 (这些图块在应用重叠并裁剪后，会被缩放到模型输入尺寸)
                nominal_tile_w = orig_img_w / num_cols
                nominal_tile_h = orig_img_h / num_rows

                # 计算基于名义图块尺寸的重叠像素 (也可以选择基于模型输入尺寸计算，这里用名义尺寸更符合固定分割的直觉)
                overlap_w_px = int(nominal_tile_w * TILE_OVERLAP_RATIO)
                overlap_h_px = int(nominal_tile_h * TILE_OVERLAP_RATIO)

                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        num_tiles_processed += 1
                        # 计算当前图块的起始点 (考虑前一个图块的重叠部分)
                        # x_start 是当前名义图块的开始，减去与前一个图块的重叠部分
                        # tile_x0 = c_idx * nominal_tile_w - (overlap_w_px if c_idx > 0 else 0) # 不对
                        # tile_y0 = r_idx * nominal_tile_h - (overlap_h_px if r_idx > 0 else 0) # 不对

                        # 正确的起始点计算，基于步长概念
                        # 步长 = 名义尺寸 - 重叠量
                        # 如果只有一个块，则没有重叠，步长等于名义尺寸
                        stride_x = nominal_tile_w if num_cols == 1 else (nominal_tile_w - overlap_w_px)
                        stride_y = nominal_tile_h if num_rows == 1 else (nominal_tile_h - overlap_h_px)

                        current_tile_x0 = int(c_idx * stride_x)
                        current_tile_y0 = int(r_idx * stride_y)

                        # 实际提取的图块尺寸是名义尺寸
                        current_tile_x1 = int(current_tile_x0 + nominal_tile_w)
                        current_tile_y1 = int(current_tile_y0 + nominal_tile_h)

                        # 裁剪到图像边界
                        tile_crop_x0 = max(0, current_tile_x0)
                        tile_crop_y0 = max(0, current_tile_y0)
                        tile_crop_x1 = min(orig_img_w, current_tile_x1)
                        tile_crop_y1 = min(orig_img_h, current_tile_y1)

                        tile_data = original_image[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]

                        if tile_h_curr == 0 or tile_w_curr == 0: continue
                        # 可选：跳过过小的图块 (例如由于裁剪导致)
                        if tile_h_curr < model_input_h * 0.1 or tile_w_curr < model_input_w * 0.1 : continue

                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(tile_data, (model_input_h, model_input_w)); total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s
                        t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h, model_input_w), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, effective_model_output_channels_for_postproc); total_tile_postprocessing_time += time.time() - t_s

                        if boxes_np.shape[0] > 0:
                            for i in range(boxes_np.shape[0]):
                                b = boxes_np[i]
                                aggregated_boxes.append([b[0] + tile_crop_x0, b[1] + tile_crop_y0, b[2] + tile_crop_x0, b[3] + tile_crop_y0])
                                aggregated_scores.append(scores_np[i])
                                aggregated_class_ids.append(c_ids_np[i])

            else: # Dynamic Tiling (original V2.1 logic)
                tile_w_dyn, tile_h_dyn = model_input_w, model_input_h # Dynamic tiles are model input size
                overlap_w_dyn, overlap_h_dyn = int(tile_w_dyn * TILE_OVERLAP_RATIO), int(tile_h_dyn * TILE_OVERLAP_RATIO)
                stride_w_dyn, stride_h_dyn = tile_w_dyn - overlap_w_dyn, tile_h_dyn - overlap_h_dyn

                for y0_dyn in range(0, orig_img_h, stride_h_dyn):
                    for x0_dyn in range(0, orig_img_w, stride_w_dyn):
                        num_tiles_processed += 1
                        x1_dyn, y1_dyn = min(x0_dyn + tile_w_dyn, orig_img_w), min(y0_dyn + tile_h_dyn, orig_img_h)
                        tile_data = original_image[y0_dyn:y1_dyn, x0_dyn:x1_dyn]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0: continue
                        if tile_h_curr < model_input_h * 0.1 or tile_w_curr < model_input_w * 0.1 : continue

                        t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(tile_data, (model_input_h, model_input_w)); total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time(); outputs = session.run(None, {input_name: tensor}); total_inference_time += time.time() - t_s
                        t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (tile_h_curr, tile_w_curr), (model_input_h, model_input_w), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, effective_model_output_channels_for_postproc); total_tile_postprocessing_time += time.time() - t_s

                        if boxes_np.shape[0] > 0:
                            for i in range(boxes_np.shape[0]):
                                b = boxes_np[i]
                                aggregated_boxes.append([b[0] + x0_dyn, b[1] + y0_dyn, b[2] + x0_dyn, b[3] + y0_dyn])
                                aggregated_scores.append(scores_np[i])
                                aggregated_class_ids.append(c_ids_np[i])

            timing_profile['3a_tiling_loop_total (incl_all_tiles_pre_inf_post)'] = time.time() - t_start_tiling_loop
            timing_profile['3b_tiling_total_tile_preprocessing'] = total_tile_preprocessing_time
            timing_profile['3c_tiling_total_tile_inference'] = total_inference_time
            timing_profile['3d_tiling_total_tile_postprocessing'] = total_tile_postprocessing_time
            print(f"切块检测完成 (处理 {num_tiles_processed} 个图块)。")

            if len(aggregated_boxes) > 0:
                t_start_nms = time.time(); keep_indices = non_max_suppression(np.array(aggregated_boxes), np.array(aggregated_scores), IOU_THRESHOLD); timing_profile['4a_global_nms'] = time.time() - t_start_nms
                aggregated_boxes = [aggregated_boxes[i] for i in keep_indices]; aggregated_scores = [aggregated_scores[i] for i in keep_indices]; aggregated_class_ids = [aggregated_class_ids[i] for i in keep_indices]
                print(f"全局NMS完成 ({timing_profile['4a_global_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else:
                timing_profile['4a_global_nms'] = 0; aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []; print("切块后未检测到聚合对象，或NMS后无剩余对象。")

        else: # 整图处理 (如果 apply_tiling 为 False)
            print("--- 开始整图检测 (切块未启用) ---")
            t_s = time.time(); tensor, ratio, pad_x, pad_y = preprocess_image_data(original_image, (model_input_h, model_input_w)); timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s
            t_s = time.time(); outputs = session.run(None, {input_name: tensor}); timing_profile['3b_fullimg_inference'] = time.time() - t_s
            t_s = time.time(); boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(outputs, (orig_img_h, orig_img_w), (model_input_h, model_input_w), ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, effective_model_output_channels_for_postproc); timing_profile['3c_fullimg_postprocessing_no_nms'] = time.time() - t_s

            if boxes_np.shape[0] > 0:
                t_s_nms = time.time(); keep_indices = non_max_suppression(boxes_np, scores_np, IOU_THRESHOLD); timing_profile['4a_fullimg_nms'] = time.time() - t_s_nms
                aggregated_boxes = [boxes_np[i] for i in keep_indices]; aggregated_scores = [scores_np[i] for i in keep_indices]; aggregated_class_ids = [c_ids_np[i] for i in keep_indices]
                print(f"整图NMS完成 ({timing_profile['4a_fullimg_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else:
                timing_profile['4a_fullimg_nms'] = 0; aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []; print("整图检测未找到对象，或NMS后无剩余对象。")

        if len(aggregated_boxes) > 0 and ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or actual_max_area_threshold_px is not None):
            t_start_area_filter = time.time(); filtered_by_area_boxes, filtered_by_area_scores, filtered_by_area_ids = [], [], []; initial_box_count_before_area_filter = len(aggregated_boxes)
            for i, box in enumerate(aggregated_boxes):
                b_w, b_h = box[2] - box[0], box[3] - box[1]; area = b_w * b_h; valid_area = True
                if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA: valid_area = False
                if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid_area = False
                if valid_area: filtered_by_area_boxes.append(box); filtered_by_area_scores.append(aggregated_scores[i]); filtered_by_area_ids.append(aggregated_class_ids[i])
            aggregated_boxes, aggregated_scores, aggregated_class_ids = filtered_by_area_boxes, filtered_by_area_scores, filtered_by_area_ids; timing_profile['5_area_filtering'] = time.time() - t_start_area_filter
            print(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框, {timing_profile['5_area_filtering']:.3f} 秒).")
            if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0: print(f"  使用最小面积阈值: {MIN_DETECTION_AREA} px²")
            if actual_max_area_threshold_px is not None: print(f"  使用最大面积阈值: {actual_max_area_threshold_px:.0f} px²")
        else:
            timing_profile['5_area_filtering'] = 0
            if len(aggregated_boxes) > 0: print("面积筛选未启用或不适用。")

        if len(aggregated_boxes) > 0:
            print(f"--- 最终检测结果 (共 {len(aggregated_boxes)} 个对象) ---")
            for i, box_coords in enumerate(aggregated_boxes):
                class_id = int(aggregated_class_ids[i]); class_name_str = f"ClassID:{class_id}"
                if class_names and 0 <= class_id < len(class_names): class_name_str = class_names[class_id]
                elif not class_names and class_id == 0 : class_name_str = "Object"
                print(f"  对象 {i+1}: 类别='{class_name_str}', 边界框={np.array(box_coords).astype(int).tolist()}, 置信度={aggregated_scores[i]:.2f}")
            t_start_drawing = time.time(); output_img = draw_detections(original_image, np.array(aggregated_boxes), np.array(aggregated_scores), np.array(aggregated_class_ids), class_names); timing_profile['6_drawing_results'] = time.time() - t_start_drawing
            output_fn_base = os.path.splitext(os.path.basename(IMAGE_NAME))[0]; output_fn = f"output_{output_fn_base}_v2.2_processed.png"; cv2.imwrite(output_fn, output_img)
            print(f"结果已保存到: {output_fn} ({timing_profile['6_drawing_results']:.2f} 秒用于绘图)")
        else:
            print("最终未检测到任何对象。"); timing_profile['6_drawing_results'] = 0

    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"发生错误: {e}"); traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 (V2.2) ---"); sorted_timing_profile = {k: timing_profile[k] for k in sorted(timing_profile.keys())};
        for stage, duration in sorted_timing_profile.items(): print(f"  {stage}: {duration:.3f} 秒")
        print(f"------------------------------")