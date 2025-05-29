import cv2
import numpy as np
import onnxruntime
import os
import time # 用于计时

# --- V2.0 配置参数 ---
# !! 修改这里为你想要测试的ONNX模型路径 !!
ONNX_MODEL_PATH = r"./model/yolov8/yolov8_barcode_detection.onnx" # 或 "yolov8n.onnx"

IMAGE_NAME = "3.jpg"  # 或者你想要测试的其他图片
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45

# --- 切块逻辑配置 (Tiling Configuration) ---
ENABLE_TILING = True  # 是否启用切块逻辑
# 当图像的宽度或高度大于模型输入尺寸的 N 倍时，才启用切块
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5
TILE_OVERLAP_RATIO = 0.2  # 图块之间的重叠比例 (例如 0.2 表示 20% 的重叠)

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

# --- 辅助函数 ---

def preprocess_image_data(img_data, input_shape_hw):
    """
    预处理图像数据 (NumPy array)
    input_shape_hw: (height, width) for model input
    """
    img = img_data
    if img is None:
        raise ValueError("输入图像数据为空")

    img_height, img_width = img.shape[:2]
    input_height, input_width = input_shape_hw

    ratio_w = input_width / img_width
    ratio_h = input_height / img_height
    ratio = min(ratio_w, ratio_h)

    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 128, dtype=np.uint8)
    x_offset_padding = (input_width - new_width) // 2
    y_offset_padding = (input_height - new_height) // 2
    canvas[y_offset_padding:y_offset_padding + new_height, x_offset_padding:x_offset_padding + new_width] = resized_img

    input_img_tensor = canvas.transpose(2, 0, 1)
    input_img_tensor = np.ascontiguousarray(input_img_tensor, dtype=np.float32)
    input_img_tensor /= 255.0
    input_img_tensor = np.expand_dims(input_img_tensor, axis=0)

    return input_img_tensor, ratio, x_offset_padding, y_offset_padding

def non_max_suppression(boxes, scores, iou_threshold):
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    if boxes.shape[0] == 0:
        return []

    indices = np.argsort(scores)[::-1]
    keep_boxes_indices = []
    while len(indices) > 0:
        current_index = indices[0]
        keep_boxes_indices.append(current_index)
        current_box = boxes[current_index]
        remaining_indices = indices[1:]
        if len(remaining_indices) == 0:
            break
        remaining_boxes_for_iou = boxes[remaining_indices]

        x1_max = np.maximum(current_box[0], remaining_boxes_for_iou[:, 0])
        y1_max = np.maximum(current_box[1], remaining_boxes_for_iou[:, 1])
        x2_min = np.minimum(current_box[2], remaining_boxes_for_iou[:, 2])
        y2_min = np.minimum(current_box[3], remaining_boxes_for_iou[:, 3])

        inter_area = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)

        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes_for_iou[:, 2] - remaining_boxes_for_iou[:, 0]) * \
                          (remaining_boxes_for_iou[:, 3] - remaining_boxes_for_iou[:, 1])

        union_area = current_area + remaining_areas - inter_area
        iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area, dtype=float), where=union_area!=0)

        indices_to_keep_from_remaining = np.where(iou <= iou_threshold)[0]
        indices = remaining_indices[indices_to_keep_from_remaining]

    return keep_boxes_indices


def postprocess_detections_from_tile(outputs, tile_original_shape_hw, model_input_shape_hw,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold, model_output_channels):
    """
    对单个图块的推理结果进行后处理。
    返回的边界框坐标是相对于 *原始未缩放的图块* 的。
    返回: boxes (np.array), scores (np.array), class_ids (np.array)
    """
    predictions_raw = np.squeeze(outputs[0])
    num_channels = predictions_raw.shape[0]

    boxes_tile_local_scaled = []
    scores_tile_local = []
    class_ids_tile_local = []

    for pred_data in predictions_raw.transpose():
        cx, cy, w, h = pred_data[:4]
        current_confidence = 0.0
        current_class_id = -1

        if num_channels == 6:
            current_confidence = pred_data[4]
            current_class_id = 0
        elif num_channels > 4:
            class_scores = pred_data[4:]
            current_confidence = np.max(class_scores)
            current_class_id = np.argmax(class_scores)
        else:
            continue

        if current_confidence >= conf_threshold:
            x1 = (cx - w / 2)
            y1 = (cy - h / 2)
            x2 = (cx + w / 2)
            y2 = (cy + h / 2)
            boxes_tile_local_scaled.append([x1, y1, x2, y2])
            scores_tile_local.append(current_confidence)
            class_ids_tile_local.append(current_class_id)

    if not boxes_tile_local_scaled:
        return np.array([]), np.array([]), np.array([])

    final_boxes_tile_original_coords = []
    tile_h_original, tile_w_original = tile_original_shape_hw

    for box in boxes_tile_local_scaled:
        box_no_pad_x1 = box[0] - preprocessing_pad_x
        box_no_pad_y1 = box[1] - preprocessing_pad_y
        box_no_pad_x2 = box[2] - preprocessing_pad_x
        box_no_pad_y2 = box[3] - preprocessing_pad_y

        orig_tile_x1 = box_no_pad_x1 / preprocessing_ratio
        orig_tile_y1 = box_no_pad_y1 / preprocessing_ratio
        orig_tile_x2 = box_no_pad_x2 / preprocessing_ratio
        orig_tile_y2 = box_no_pad_y2 / preprocessing_ratio

        orig_tile_x1 = np.clip(orig_tile_x1, 0, tile_w_original)
        orig_tile_y1 = np.clip(orig_tile_y1, 0, tile_h_original)
        orig_tile_x2 = np.clip(orig_tile_x2, 0, tile_w_original)
        orig_tile_y2 = np.clip(orig_tile_y2, 0, tile_h_original)

        final_boxes_tile_original_coords.append([orig_tile_x1, orig_tile_y1, orig_tile_x2, orig_tile_y2])

    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)


def draw_detections(image, boxes, scores, class_ids, class_names=None):
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int) # 确保是整数类型用于绘图
        score = scores[i]
        class_id = int(class_ids[i]) # 确保 class_id 是整数

        label_name = f"ClassID:{class_id}"
        if class_names and 0 <= class_id < len(class_names):
            label_name = class_names[class_id]

        label_text = f"{label_name}: {score:.2f}"
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_out

# --- V2.0 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误: ONNX 模型文件未找到: {ONNX_MODEL_PATH}")
        exit()
    if not os.path.exists(IMAGE_NAME):
        print(f"错误: 测试图片未找到: {IMAGE_NAME}")
        exit()

    start_time_total = time.time()

    try:
        print(f"--- 初始化与模型加载 (V2.0) ---")
        print(f"正在加载模型: {ONNX_MODEL_PATH}")
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

        input_cfg = session.get_inputs()[0]
        output_cfg = session.get_outputs()[0]
        input_name = input_cfg.name
        input_shape_onnx = input_cfg.shape
        model_input_height = input_shape_onnx[2]
        model_input_width = input_shape_onnx[3]

        output_shape_onnx = output_cfg.shape
        model_output_channels = output_shape_onnx[1]

        print(f"模型输入名称: {input_name}, 输入形状 (ONNX): {input_shape_onnx}")
        print(f"模型输出形状 (ONNX): {output_shape_onnx}, 输出通道数: {model_output_channels}")

        class_names_to_use = None
        if model_output_channels == 6:
            class_names_to_use = ["Barcode"]
            print("模型类型: 特定条形码模型 (6 输出通道)。")
        elif model_output_channels == (4 + len(COCO_CLASSES)):
            class_names_to_use = COCO_CLASSES
            print(f"模型类型: COCO 通用模型 ({len(COCO_CLASSES)} 类, {model_output_channels} 输出通道)。")
        else:
            num_classes = model_output_channels - 4
            if num_classes > 0:
                class_names_to_use = [f"Class_{i}" for i in range(num_classes)]
                print(f"模型类型: 自定义类别模型 ({num_classes} 类, {model_output_channels} 输出通道)。")
            else:
                class_names_to_use = []
                print(f"警告: 未知模型类型，输出通道 {model_output_channels}。")

        original_image = cv2.imread(IMAGE_NAME)
        if original_image is None:
            raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")

        orig_img_h, orig_img_w = original_image.shape[:2]
        print(f"原始图片尺寸: (H={orig_img_h}, W={orig_img_w})")

        apply_tiling = False
        if ENABLE_TILING:
            if orig_img_w > model_input_width * MIN_IMAGE_DIM_FACTOR_FOR_TILING or \
               orig_img_h > model_input_height * MIN_IMAGE_DIM_FACTOR_FOR_TILING:
                apply_tiling = True
                print("启用切块处理，因为图像尺寸较大。")
            else:
                print("图像尺寸未达到切块阈值，将进行整图处理。")
        else:
            print("切块处理被禁用，将进行整图处理。")

        # 初始化为列表，方便在切块逻辑中 append
        aggregated_boxes_global = []
        aggregated_scores_global = []
        aggregated_class_ids_global = []

        if apply_tiling:
            print(f"--- 开始切块检测 ---")
            tile_size_w, tile_size_h = model_input_width, model_input_height
            overlap_w = int(tile_size_w * TILE_OVERLAP_RATIO)
            overlap_h = int(tile_size_h * TILE_OVERLAP_RATIO)
            stride_w = tile_size_w - overlap_w
            stride_h = tile_size_h - overlap_h

            num_tiles_processed = 0
            for y_start in range(0, orig_img_h, stride_h):
                for x_start in range(0, orig_img_w, stride_w):
                    num_tiles_processed += 1
                    x_end = min(x_start + tile_size_w, orig_img_w)
                    y_end = min(y_start + tile_size_h, orig_img_h)

                    current_tile_img_data = original_image[y_start:y_end, x_start:x_end]
                    current_tile_h, current_tile_w = current_tile_img_data.shape[:2]

                    if current_tile_h == 0 or current_tile_w == 0:
                        continue

                    preprocessed_tile_tensor, tile_ratio, tile_pad_x, tile_pad_y = \
                        preprocess_image_data(current_tile_img_data, (model_input_height, model_input_width))

                    tile_outputs = session.run(None, {input_name: preprocessed_tile_tensor})

                    tile_boxes_local_np, tile_scores_np, tile_class_ids_np = postprocess_detections_from_tile(
                        tile_outputs,
                        (current_tile_h, current_tile_w),
                        (model_input_height, model_input_width),
                        tile_ratio, tile_pad_x, tile_pad_y,
                        CONFIDENCE_THRESHOLD,
                        model_output_channels
                    )

                    if tile_boxes_local_np.shape[0] > 0:
                        for i in range(tile_boxes_local_np.shape[0]):
                            box_local = tile_boxes_local_np[i]
                            global_x1 = box_local[0] + x_start
                            global_y1 = box_local[1] + y_start
                            global_x2 = box_local[2] + x_start
                            global_y2 = box_local[3] + y_start

                            aggregated_boxes_global.append([global_x1, global_y1, global_x2, global_y2])
                            aggregated_scores_global.append(tile_scores_np[i])
                            aggregated_class_ids_global.append(tile_class_ids_np[i])

            print(f"--- 切块检测完成，共处理 {num_tiles_processed} 个图块 ---")

            if len(aggregated_boxes_global) > 0:
                print(f"进行全局非极大值抑制 (处理 {len(aggregated_boxes_global)} 个初始框)...")
                # non_max_suppression 输入需要是 NumPy 数组
                keep_indices_global = non_max_suppression(
                    np.array(aggregated_boxes_global),
                    np.array(aggregated_scores_global),
                    IOU_THRESHOLD
                )

                # 更新为 NMS 后的结果 (保持为列表)
                final_boxes_after_nms = [aggregated_boxes_global[i] for i in keep_indices_global]
                final_scores_after_nms = [aggregated_scores_global[i] for i in keep_indices_global]
                final_class_ids_after_nms = [aggregated_class_ids_global[i] for i in keep_indices_global]

                aggregated_boxes_global = final_boxes_after_nms
                aggregated_scores_global = final_scores_after_nms
                aggregated_class_ids_global = final_class_ids_after_nms
            else:
                print("在所有图块中均未检测到高于置信度阈值的对象。")
                # 确保在无检测时列表为空
                aggregated_boxes_global, aggregated_scores_global, aggregated_class_ids_global = [], [], []


        else: # --- 整图处理 ---
            print("--- 开始整图检测 (无切块) ---")
            preprocessed_full_img_tensor, full_img_ratio, full_img_pad_x, full_img_pad_y = \
                preprocess_image_data(original_image, (model_input_height, model_input_width))

            full_img_outputs = session.run(None, {input_name: preprocessed_full_img_tensor})

            # postprocess_detections_from_tile 返回 NumPy 数组
            boxes_np, scores_np, class_ids_np = postprocess_detections_from_tile(
                full_img_outputs,
                (orig_img_h, orig_img_w),
                (model_input_height, model_input_width),
                full_img_ratio, full_img_pad_x, full_img_pad_y,
                CONFIDENCE_THRESHOLD,
                model_output_channels
            )

            if boxes_np.shape[0] > 0:
                print(f"进行整图非极大值抑制 (处理 {boxes_np.shape[0]} 个初始框)...")
                keep_indices_full_img = non_max_suppression(boxes_np, scores_np, IOU_THRESHOLD)

                # 将 NMS 结果转换为列表，以便后续统一处理
                aggregated_boxes_global = [boxes_np[i] for i in keep_indices_full_img]
                aggregated_scores_global = [scores_np[i] for i in keep_indices_full_img]
                aggregated_class_ids_global = [class_ids_np[i] for i in keep_indices_full_img]
            else:
                # 确保在无检测时列表为空
                aggregated_boxes_global, aggregated_scores_global, aggregated_class_ids_global = [], [], []


        # --- 可视化与输出 ---
        if len(aggregated_boxes_global) > 0:
            print(f"--- 检测结果 (共 {len(aggregated_boxes_global)} 个对象) ---")
            for i, box_coords in enumerate(aggregated_boxes_global):
                class_id = int(aggregated_class_ids_global[i]) # 确保是整数
                class_name = f"ClassID:{class_id}"
                if class_names_to_use and 0 <= class_id < len(class_names_to_use):
                    class_name = class_names_to_use[class_id]

                print(f"  对象 {i+1}: 类别='{class_name}', 边界框={np.array(box_coords).astype(int).tolist()}, 置信度={aggregated_scores_global[i]:.2f}")

            # 确保传递给 draw_detections 的是 NumPy 数组
            output_image_display = draw_detections(
                original_image,
                np.array(aggregated_boxes_global),
                np.array(aggregated_scores_global),
                np.array(aggregated_class_ids_global),
                class_names=class_names_to_use
            )
            output_image_filename = f"output_{os.path.splitext(os.path.basename(IMAGE_NAME))[0]}_v2.png"
            cv2.imwrite(output_image_filename, output_image_display)
            print(f"结果已保存到: {output_image_filename}")

            cv2.imshow("Detected Objects (V2.0)", output_image_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("在当前设置下，最终未检测到任何对象。")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        print(f"--- 总处理时间: {end_time_total - start_time_total:.2f} 秒 ---")