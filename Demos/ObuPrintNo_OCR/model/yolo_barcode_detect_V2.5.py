# coding: utf-8
"""
OBU (On-Board Unit) 镭标码区域目标检测与识别脚本
版本: v2.5.0_ocr_barcode_integration
功能:
- 使用 ONNX YOLO 模型检测 OBU 镭标码区域。
- 对检测到的区域分割出数字子区域和条形码子区域。
- 使用 PaddleOCR 对数字子区域进行 OCR 识别。
- 使用 pyzbar 对条形码子区域进行条形码解码。
- 整合 OCR 和条形码识别结果。
- 支持整图推理和可选的切块推理。
- 保存处理过程中的 ROI 图像和最终结果图。
- 提供详细的计时分析。

依赖安装提示:
1. 安装 zbar 系统库 (具体命令取决于您的操作系统)
   - Ubuntu/Debian: sudo apt-get install libzbar0
   - Fedora/CentOS: sudo yum install zbar
   - macOS: brew install zbar
2. 安装 Python 包:
   pip install opencv-python numpy onnxruntime paddleocr paddlepaddle pyzbar -i https://pypi.tuna.tsinghua.edu.cn/simple
   (如果您有GPU并希望PaddlePaddle使用GPU，请安装对应GPU版本的paddlepaddle)
"""
import cv2
import numpy as np
import onnxruntime
import os
import time
import traceback
import paddle
import paddleocr
from pyzbar.pyzbar import decode as pyzbar_decode
from pyzbar.pyzbar import ZBarSymbol # 用于可能的类型指定

# --- PaddleOCR引擎初始化 ---
paddle_ocr_engine = None
try:
    print("Initializing PaddleOCR engine globally...")
    # 如需使用GPU，取消注释下一行并确保 paddlepaddle-gpu 已正确安装
    # paddle_ocr_engine = paddleocr.PaddleOCR(lang='en', use_gpu=True)
    # 针对参数变化进行调整：
    # 1. use_angle_cls -> use_textline_orientation
    # 2. show_log 参数移除，PaddleOCR默认日志级别可以通过logging模块控制，或者其内部已有调整
    #    通常我们不需要在这里显式关闭，除非日志非常多。如果需要减少日志，可以尝试其推荐的参数。
    #    对于识别任务，`det=False` (如果只做识别不做检测) 或者 `rec=True` 是核心。
    #    我们这里是传入已经裁剪好的图块进行识别，所以检测部分(det)可以认为是关闭的。
    paddle_ocr_engine = paddleocr.PaddleOCR(lang='en', use_textline_orientation=False, det=False, rec=True)
    print("PaddleOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR engine globally: {e}")
    # traceback.print_exc() # 调试时可以打开，查看详细错误
    print("PaddleOCR features will be disabled.")
    paddle_ocr_engine = None

# --- V2.5 配置参数 ---
VERSION = "v2.5.0_ocr_barcode_integration"
ONNX_MODEL_PATH = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
IMAGE_NAME = r"../../DATA/PIC/3.jpg" # 示例图片路径
CONFIDENCE_THRESHOLD = 0.25  # YOLO检测置信度阈值
IOU_THRESHOLD = 0.45         # NMS的IOU阈值

# --- 切块逻辑配置 ---
ENABLE_TILING = False  # 默认禁用切块，优先使用新模型的整图推理
FIXED_TILING_GRID = None # 例如 (2, 2) 表示固定 2x2 网格切块
MIN_IMAGE_DIM_FACTOR_FOR_TILING = 1.5 # 图像边长大于模型输入边长此倍数时才考虑动态切块
TILE_OVERLAP_RATIO = 0.2 # 切块重叠率

# --- 检测结果面积筛选配置 ---
MIN_DETECTION_AREA = 2000  # 最小有效检测框面积 (像素平方)
MAX_DETECTION_AREA = 0.1   # 最大有效检测框面积占图像总面积的比例

# --- V2.5 ROI 精确分割参数 (基于YOLO检测框bbox_yolo) ---
# 假设 YOLO 检测框完整包含了数字和条形码区域。
# 这些比例值需要根据实际镭标码中数字区和条码区的布局进行调优。
# 数字区域在YOLO框内的相对位置和大小
DIGIT_SUBAREA_Y_START_RATIO = 0.05  # 数字区域上边缘距离YOLO框上边缘的距离占YOLO框高度的比例
DIGIT_SUBAREA_HEIGHT_RATIO = 0.40 # 数字区域高度占YOLO框总高度的比例
DIGIT_SUBAREA_WIDTH_EXPAND_FACTOR = 1.05 # 数字区域宽度扩展因子

# 条形码区域在YOLO框内的相对位置和大小
BARCODE_SUBAREA_Y_START_RATIO = 0.45 # 条形码区域上边缘距离YOLO框上边缘的距离占YOLO框高度的比例
BARCODE_SUBAREA_HEIGHT_RATIO = 0.50 # 条形码区域高度占YOLO框总高度的比例
BARCODE_SUBAREA_WIDTH_EXPAND_FACTOR = 1.0 # 条形码区域宽度扩展因子 (通常条码对宽度较敏感)

# 选择送给PaddleOCR的预处理类型: "color_digit", "gray_digit", "binary_otsu_digit", "binary_adaptive_digit"
# 或 "all" 尝试所有, 或列表 ["binary_otsu_digit", "gray_digit"]
OCR_PREPROCESS_TYPE_TO_USE = "binary_otsu_digit"

COCO_CLASSES = ['Barcode']  # 模型检测的类别名称 (镭标码区域)

timing_profile = {}
process_photo_dir = "process_photo"  # 过程图片保存目录

# --- 清理过程图片文件夹的函数 ---
def clear_process_photo_directory(directory="process_photo"):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory, exist_ok=True)

# --- 原V2.2/V2.3的切块相关函数 (保留但不一定调用) ---
def preprocess_image_data_for_tiling(img_data, input_shape_hw):
    img = img_data
    if img is None:
        raise ValueError("输入图像数据为空 (tiling)")
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

def postprocess_detections_from_tile(outputs, tile_original_shape_hw, _,
                                     preprocessing_ratio, preprocessing_pad_x, preprocessing_pad_y,
                                     conf_threshold_tile, model_output_channels_param_ignored):
    predictions_raw = np.squeeze(outputs[0])
    if predictions_raw.ndim != 2:
        return np.array([]), np.array([]), np.array([])
    actual_model_output_channels = predictions_raw.shape[0]
    if not isinstance(actual_model_output_channels, int):
        return np.array([]), np.array([]), np.array([])
    transposed_predictions = predictions_raw.transpose()
    boxes_tile_local_scaled, scores_tile_local, class_ids_tile_local = [], [], []
    for pred_data in transposed_predictions:
        if len(pred_data) != actual_model_output_channels:
            continue
        cx, cy, w, h = pred_data[:4]
        confidence, class_id = 0.0, -1
        if actual_model_output_channels == 6: # (cx,cy,w,h,conf,class_id)
            confidence = pred_data[4]
            class_id = int(pred_data[5])
        elif actual_model_output_channels == 5: # (cx,cy,w,h,conf) -> single class
            confidence = pred_data[4]
            class_id = 0 # Assume single class if only 5 channels
        elif actual_model_output_channels > 4 : # (cx,cy,w,h,conf_class1,conf_class2,...)
            class_scores = pred_data[4:]
            if class_scores.size > 0:
                confidence = np.max(class_scores)
                class_id = np.argmax(class_scores)
            else:
                continue
        else:
            continue
        if confidence >= conf_threshold_tile:
            x1, y1, x2, y2 = (cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)
            boxes_tile_local_scaled.append([x1, y1, x2, y2])
            scores_tile_local.append(confidence)
            class_ids_tile_local.append(class_id)
    if not boxes_tile_local_scaled:
        return np.array([]), np.array([]), np.array([])
    final_boxes_tile_original_coords = []
    tile_h_orig, tile_w_orig = tile_original_shape_hw
    for box in boxes_tile_local_scaled:
        b_x1, b_y1, b_x2, b_y2 = box[0] - preprocessing_pad_x, box[1] - preprocessing_pad_y, \
                                 box[2] - preprocessing_pad_x, box[3] - preprocessing_pad_y
        if preprocessing_ratio == 0:
            continue
        ot_x1, ot_y1, ot_x2, ot_y2 = b_x1 / preprocessing_ratio, b_y1 / preprocessing_ratio, \
                                     b_x2 / preprocessing_ratio, b_y2 / preprocessing_ratio
        ot_x1, ot_y1 = np.clip(ot_x1, 0, tile_w_orig), np.clip(ot_y1, 0, tile_h_orig)
        ot_x2, ot_y2 = np.clip(ot_x2, 0, tile_w_orig), np.clip(ot_y2, 0, tile_h_orig)
        final_boxes_tile_original_coords.append([ot_x1, ot_y1, ot_x2, ot_y2])
    return np.array(final_boxes_tile_original_coords), np.array(scores_tile_local), np.array(class_ids_tile_local)

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold):
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0:
        return []
    if not isinstance(scores, np.ndarray) or scores.size == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        current_order_size = order.size
        order = order[1:]
        if current_order_size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds]
    return keep

# --- 新的预处理和后处理函数 (适配新ONNX模型) ---
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
                                   num_classes=1): # num_classes (unused but kept for signature)
    raw_output_tensor = np.squeeze(outputs_onnx[0])
    if raw_output_tensor.ndim != 2:
        print(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}")
        return []
    # YOLOv8 output format: [batch, num_predictions, 4_coords + num_classes_conf] OR [batch, 4_coords + num_classes_conf, num_predictions]
    # For single class, it's [batch, num_predictions, 4_coords + 1_conf] or transposed
    # Assuming model output is [batch, 5, num_predictions] for single class (x,y,w,h,conf)
    # Or [batch, num_predictions, 5]
    # We need to ensure predictions are iterated correctly.
    # If shape is (5, N), transpose it to (N, 5). If (N, 5), use as is.
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor

    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []
    # For single class YOLOv8, output per detection is (cx, cy, w, h, confidence)
    expected_attributes = 4 + 1 # 4 for bbox, 1 for confidence for single class

    for i_pred, pred_data in enumerate(predictions_to_iterate):
        if len(pred_data) != expected_attributes:
            if i_pred == 0: # Print error only once per processing
                print(f"错误: Main每个预测的属性数量 ({len(pred_data)}) 与期望值 ({expected_attributes}) 不符。")
            continue

        box_coords_raw = pred_data[:4]  # cx, cy, w, h
        final_confidence = float(pred_data[4]) # Confidence for the single class
        class_id = 0 # Since it's a single class model for 'Barcode'

        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = box_coords_raw
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2])
            scores_candidate.append(final_confidence)
            class_ids_candidate.append(class_id)

    if not boxes_candidate:
        return []

    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)
    final_detections = []
    orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx)
        box_model_coords = boxes_candidate[idx]
        score = scores_candidate[idx]
        class_id_val = class_ids_candidate[idx]

        # Scale back to original image coordinates
        box_no_pad_x1, box_no_pad_y1 = box_model_coords[0] - pad_x_preproc, box_model_coords[1] - pad_y_preproc
        box_no_pad_x2, box_no_pad_y2 = box_model_coords[2] - pad_x_preproc, box_model_coords[3] - pad_y_preproc

        if ratio_preproc == 0: continue # Avoid division by zero

        orig_x1, orig_y1 = box_no_pad_x1 / ratio_preproc, box_no_pad_y1 / ratio_preproc
        orig_x2, orig_y2 = box_no_pad_x2 / ratio_preproc, box_no_pad_y2 / ratio_preproc

        # Clip to original image boundaries
        final_x1, final_y1 = np.clip(orig_x1, 0, orig_w), np.clip(orig_y1, 0, orig_h)
        final_x2, final_y2 = np.clip(orig_x2, 0, orig_w), np.clip(orig_y2, 0, orig_h)

        final_detections.append([int(final_x1), int(final_y1), int(final_x2), int(final_y2), score, class_id_val])
    return final_detections

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    img_out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        score = scores[i]
        class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"ClassID:{class_id}"

        # YOLO Box and label
        yolo_label_text = f"{label_name}: {score:.2f}"
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for YOLO box
        cv2.putText(img_out, yolo_label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ROI Index
        if roi_indices and i < len(roi_indices):
            cv2.putText(img_out, f"ROI:{roi_indices[i]}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1) # Red for ROI index

        # OCR/Barcode Text (final recognized value)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] not in ["N/A", "FAIL"]:
            # Position OCR text above the YOLO label
            text_to_display = ocr_texts[i]
            (text_width, text_height), _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y_pos = y1 - 30
            if text_y_pos < text_height : # If text goes off screen top, put it inside box
                 text_y_pos = y1 + text_height + 5
            cv2.putText(img_out, text_to_display, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue for recognized text
        elif ocr_texts and i < len(ocr_texts) and ocr_texts[i] == "FAIL":
             cv2.putText(img_out, "FAIL", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)


    return img_out

# --- V2.5 新增：条形码解码函数 (带预处理尝试) ---
def decode_barcode_from_image_roi(image_roi, roi_index_for_debug=""):
    """
    使用 pyzbar 从给定的图像ROI中解码条形码。
    包含一些预处理尝试。

    :param image_roi: 裁剪出的条形码区域图像 (NumPy array, BGR格式)
    :param roi_index_for_debug: 用于保存调试图像的ROI索引
    :return: (decoded_text, type) 或 (None, None)
    """
    if image_roi is None or image_roi.size == 0:
        return None, None

    # 原始灰度图
    if image_roi.ndim == 3 and image_roi.shape[2] == 3:
        gray_orig = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    elif image_roi.ndim == 2:
        gray_orig = image_roi # Already grayscale
    else:
        print(f"警告: 条形码ROI图像通道数异常: {image_roi.shape}")
        return None, None

    # 图像列表，尝试不同的预处理
    images_to_try = {
        "original_gray": gray_orig,
    }

    # 1. 对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast_gray = clahe.apply(gray_orig.copy())
    images_to_try["clahe_gray"] = enhanced_contrast_gray
    if roi_index_for_debug: # 保存调试图像
        cv2.imwrite(os.path.join(process_photo_dir, f"barcode_roi_{roi_index_for_debug}_debug_clahe.png"), enhanced_contrast_gray)


    # 2. 尝试锐化 (Unsharp Masking)
    # blurred_for_usm = cv2.GaussianBlur(gray_orig, (5,5), 0) # Sigma 0 means it's computed from kernel size
    # sharpened_usm = cv2.addWeighted(gray_orig, 1.5, blurred_for_usm, -0.5, 0)
    # images_to_try["sharpened_usm_gray"] = sharpened_usm
    # if roi_index_for_debug:
    #     cv2.imwrite(os.path.join(process_photo_dir, f"barcode_roi_{roi_index_for_debug}_debug_usm.png"), sharpened_usm)

    # 3. 尝试自适应二值化 (pyzbar内部也会做二值化，但有时外部提供效果更好)
    # adaptive_thresh_block_size = int(min(gray_orig.shape[0], gray_orig.shape[1]) * 0.15) // 2 * 2 + 1 # 约15%边长，奇数
    # if adaptive_thresh_block_size < 3: adaptive_thresh_block_size = 11 # 最小块大小
    # adaptive_binary = cv2.adaptiveThreshold(gray_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                       cv2.THRESH_BINARY, adaptive_thresh_block_size, 5) # C=5
    # images_to_try["adaptive_binary"] = adaptive_binary
    # if roi_index_for_debug:
    #     cv2.imwrite(os.path.join(process_photo_dir, f"barcode_roi_{roi_index_for_debug}_debug_adaptive_binary.png"), adaptive_binary)


    for preprocess_name, processed_gray_roi in images_to_try.items():
        # print(f"    Trying pyzbar with: {preprocess_name} for ROI {roi_index_for_debug}") # 调试时打开
        try:
            # 明确指定尝试解码的条码类型，如果已知是CODE128
            # decoded_objects = pyzbar_decode(processed_gray_roi, symbols=[ZBarSymbol.CODE128])
            decoded_objects = pyzbar_decode(processed_gray_roi)

            if decoded_objects:
                # 优先选择 CODE128 类型的结果
                for obj in decoded_objects:
                    if obj.type == 'CODE128':
                        try:
                            barcode_data = obj.data.decode('utf-8')
                            barcode_data_cleaned = ''.join(filter(str.isalnum, barcode_data)).upper() # 清洗并转大写
                            if barcode_data_cleaned:
                                # print(f"      SUCCESS ({preprocess_name}): {barcode_data_cleaned}") # 调试时打开
                                return barcode_data_cleaned, obj.type
                        except UnicodeDecodeError:
                            try:
                                barcode_data = obj.data.decode('latin-1')
                                barcode_data_cleaned = ''.join(filter(str.isalnum, barcode_data)).upper()
                                if barcode_data_cleaned:
                                    # print(f"      SUCCESS ({preprocess_name}, latin-1): {barcode_data_cleaned}") # 调试时打开
                                    return barcode_data_cleaned, obj.type
                            except UnicodeDecodeError:
                                 print(f"警告: 条形码数据解码失败 (UTF-8 and latin-1) for ROI {roi_index_for_debug} with {preprocess_name}: {obj.data}")
                                 continue

                # 如果没有CODE128，返回找到的第一个可解码的其他类型条码 (作为备选)
                # for obj in decoded_objects:
                #     try:
                #         barcode_data = obj.data.decode('utf-8')
                #         barcode_data_cleaned = ''.join(filter(str.isalnum, barcode_data)).upper()
                #         if barcode_data_cleaned:
                #             # print(f"      SUCCESS (Non-CODE128, {preprocess_name}): {barcode_data_cleaned}, Type: {obj.type}") # 调试时打开
                #             return barcode_data_cleaned, obj.type
                #     except UnicodeDecodeError:
                #         # ... (latin-1 fallback for non-CODE128) ...
                #         continue
        except Exception as e:
            print(f"条形码解码时发生错误 ({preprocess_name}) for ROI {roi_index_for_debug}: {e}")
            # traceback.print_exc()

    return None, None # 如果所有尝试都失败

# --- V2.5 主程序 ---
if __name__ == "__main__":
    t_start_overall = time.time()
    timing_profile['0_total_script_execution'] = 0
    print(f"--- OBU 检测与识别工具 {VERSION} ---")

    clear_process_photo_directory(process_photo_dir)
    print(f"'{process_photo_dir}' 文件夹已清理。")

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"错误: 模型未找到: {ONNX_MODEL_PATH}")
        exit()
    if not os.path.exists(IMAGE_NAME):
        print(f"错误: 图片未找到: {IMAGE_NAME}")
        exit()

    actual_max_area_threshold_px = None
    try:
        print(f"--- 初始化与模型加载 ---")
        t_start = time.time()
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # 如果有GPU并配置好ONNXRuntime-GPU
        timing_profile['1_model_loading'] = time.time() - t_start
        print(f"ONNX模型加载完成 ({timing_profile['1_model_loading']:.2f} 秒)")

        input_cfg = session.get_inputs()[0]
        input_name = input_cfg.name
        input_shape_onnx = input_cfg.shape

        model_input_h_ref, model_input_w_ref = 640, 640 # 默认参考值
        # 动态获取模型输入尺寸，如果模型定义了固定尺寸
        if len(input_shape_onnx) == 4 and \
           isinstance(input_shape_onnx[2], int) and input_shape_onnx[2] > 0 and \
           isinstance(input_shape_onnx[3], int) and input_shape_onnx[3] > 0:
            model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]
        else:
            print(f"警告: 模型输入维度包含符号名称或无效: {input_shape_onnx}. 使用参考尺寸 H={model_input_h_ref}, W={model_input_w_ref}")

        class_names = COCO_CLASSES
        print(f"模型输入: {input_name} {input_shape_onnx}. 类别设置为: {class_names}")

        t_start_img_read = time.time()
        original_image = cv2.imread(IMAGE_NAME)
        timing_profile['2_image_reading'] = time.time() - t_start_img_read

        if original_image is None:
            print(f"错误: 无法读取图片: {IMAGE_NAME} (耗时: {timing_profile['2_image_reading']:.2f} 秒)")
            raise FileNotFoundError(f"无法读取图片: {IMAGE_NAME}")

        orig_img_h, orig_img_w = original_image.shape[:2]
        print(f"原始图片: {IMAGE_NAME} (H={orig_img_h}, W={orig_img_w}) ({timing_profile['2_image_reading']:.2f} 秒读取)")

        if MAX_DETECTION_AREA is not None:
            if isinstance(MAX_DETECTION_AREA, float) and 0 < MAX_DETECTION_AREA <= 1.0:
                actual_max_area_threshold_px = (orig_img_h * orig_img_w) * MAX_DETECTION_AREA
                print(f"MAX_DETECTION_AREA设为总面积{MAX_DETECTION_AREA*100:.1f}%, 阈值: {actual_max_area_threshold_px:.0f} px².")
            elif isinstance(MAX_DETECTION_AREA, (int, float)) and MAX_DETECTION_AREA > 1:
                actual_max_area_threshold_px = float(MAX_DETECTION_AREA)
                print(f"MAX_DETECTION_AREA设为绝对值: {actual_max_area_threshold_px:.0f} px².")
            else:
                actual_max_area_threshold_px = None # 无效设置，禁用

        apply_tiling = ENABLE_TILING
        use_fixed_grid_tiling = False
        if apply_tiling and FIXED_TILING_GRID is not None and \
           isinstance(FIXED_TILING_GRID, tuple) and len(FIXED_TILING_GRID) == 2 and \
           all(isinstance(n, int) and n > 0 for n in FIXED_TILING_GRID):
            use_fixed_grid_tiling = True
            print(f"切块处理: 固定网格 {FIXED_TILING_GRID}. 重叠率: {TILE_OVERLAP_RATIO*100}%")
        elif apply_tiling:
            apply_tiling = (orig_img_w > model_input_w_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING or \
                            orig_img_h > model_input_h_ref * MIN_IMAGE_DIM_FACTOR_FOR_TILING)
            print(f"切块处理: {'动态切块' if apply_tiling else '禁用 (尺寸未达动态切块阈值)'}。"
                  f"参考模型输入: {model_input_h_ref}x{model_input_w_ref}, 重叠率: {TILE_OVERLAP_RATIO*100}%")
        else:
            print(f"切块处理禁用 (全局配置)。")

        aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []
        if apply_tiling:
            t_start_tiling_loop = time.time()
            total_inference_time, total_tile_preprocessing_time, total_tile_postprocessing_time, num_tiles_processed = 0,0,0,0
            if use_fixed_grid_tiling:
                num_cols, num_rows = FIXED_TILING_GRID
                nominal_tile_w = orig_img_w / num_cols
                nominal_tile_h = orig_img_h / num_rows
                overlap_w_px = int(nominal_tile_w * TILE_OVERLAP_RATIO)
                overlap_h_px = int(nominal_tile_h * TILE_OVERLAP_RATIO)
                for r_idx in range(num_rows):
                    for c_idx in range(num_cols):
                        num_tiles_processed += 1
                        stride_x = nominal_tile_w if num_cols == 1 else (nominal_tile_w - overlap_w_px)
                        stride_y = nominal_tile_h if num_rows == 1 else (nominal_tile_h - overlap_h_px)
                        current_tile_x0 = int(c_idx * stride_x)
                        current_tile_y0 = int(r_idx * stride_y)
                        current_tile_x1 = int(current_tile_x0 + nominal_tile_w)
                        current_tile_y1 = int(current_tile_y0 + nominal_tile_h)
                        tile_crop_x0 = max(0, current_tile_x0)
                        tile_crop_y0 = max(0, current_tile_y0)
                        tile_crop_x1 = min(orig_img_w, current_tile_x1)
                        tile_crop_y1 = min(orig_img_h, current_tile_y1)
                        tile_data = original_image[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or \
                           tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1: # 忽略过小的图块
                            continue
                        t_s = time.time()
                        tensor, ratio, pad_x, pad_y = preprocess_image_data_for_tiling(tile_data, (model_input_h_ref, model_input_w_ref))
                        total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time()
                        outputs = session.run(None, {input_name: tensor})
                        total_inference_time += time.time() - t_s
                        t_s = time.time()
                        boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(
                            outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref),
                            ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0 # 0 for model_output_channels (unused here)
                        )
                        total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box in range(boxes_np.shape[0]):
                                b = boxes_np[i_box]
                                aggregated_boxes.append([b[0] + tile_crop_x0, b[1] + tile_crop_y0,
                                                         b[2] + tile_crop_x0, b[3] + tile_crop_y0])
                                aggregated_scores.append(scores_np[i_box])
                                aggregated_class_ids.append(c_ids_np[i_box])
            else: # Dynamic Tiling
                tile_w_dyn, tile_h_dyn = model_input_w_ref, model_input_h_ref # Use model input size as tile size
                overlap_w_dyn, overlap_h_dyn = int(tile_w_dyn * TILE_OVERLAP_RATIO), int(tile_h_dyn * TILE_OVERLAP_RATIO)
                stride_w_dyn, stride_h_dyn = tile_w_dyn - overlap_w_dyn, tile_h_dyn - overlap_h_dyn
                for y0_dyn in range(0, orig_img_h, stride_h_dyn):
                    for x0_dyn in range(0, orig_img_w, stride_w_dyn):
                        num_tiles_processed += 1
                        x1_dyn, y1_dyn = min(x0_dyn + tile_w_dyn, orig_img_w), min(y0_dyn + tile_h_dyn, orig_img_h)
                        tile_data = original_image[y0_dyn:y1_dyn, x0_dyn:x1_dyn]
                        tile_h_curr, tile_w_curr = tile_data.shape[:2]
                        if tile_h_curr == 0 or tile_w_curr == 0 or \
                           tile_h_curr < model_input_h_ref * 0.1 or tile_w_curr < model_input_w_ref * 0.1:
                            continue
                        t_s = time.time()
                        tensor, ratio, pad_x, pad_y = preprocess_image_data_for_tiling(tile_data, (model_input_h_ref, model_input_w_ref))
                        total_tile_preprocessing_time += time.time() - t_s
                        t_s = time.time()
                        outputs = session.run(None, {input_name: tensor})
                        total_inference_time += time.time() - t_s
                        t_s = time.time()
                        boxes_np, scores_np, c_ids_np = postprocess_detections_from_tile(
                            outputs, (tile_h_curr, tile_w_curr), (model_input_h_ref, model_input_w_ref),
                            ratio, pad_x, pad_y, CONFIDENCE_THRESHOLD, 0
                        )
                        total_tile_postprocessing_time += time.time() - t_s
                        if boxes_np.shape[0] > 0:
                            for i_box in range(boxes_np.shape[0]):
                                b = boxes_np[i_box]
                                aggregated_boxes.append([b[0] + x0_dyn, b[1] + y0_dyn,
                                                         b[2] + x0_dyn, b[3] + y0_dyn])
                                aggregated_scores.append(scores_np[i_box])
                                aggregated_class_ids.append(c_ids_np[i_box])
            timing_profile['3a_tiling_loop_total (incl_all_tiles_pre_inf_post)'] = time.time() - t_start_tiling_loop
            timing_profile['3b_tiling_total_tile_preprocessing'] = total_tile_preprocessing_time
            timing_profile['3c_tiling_total_tile_inference'] = total_inference_time
            timing_profile['3d_tiling_total_tile_postprocessing'] = total_tile_postprocessing_time
            print(f"切块检测完成 (处理 {num_tiles_processed} 个图块)。")
            if len(aggregated_boxes) > 0:
                t_start_nms = time.time()
                keep_indices = non_max_suppression_global(np.array(aggregated_boxes), np.array(aggregated_scores), IOU_THRESHOLD)
                timing_profile['4a_global_nms'] = time.time() - t_start_nms
                aggregated_boxes = [aggregated_boxes[i] for i in keep_indices]
                aggregated_scores = [aggregated_scores[i] for i in keep_indices]
                aggregated_class_ids = [aggregated_class_ids[i] for i in keep_indices]
                print(f"全局NMS完成 ({timing_profile['4a_global_nms']:.2f} 秒)。找到了 {len(aggregated_boxes)} 个框。")
            else:
                timing_profile['4a_global_nms'] = 0
                aggregated_boxes, aggregated_scores, aggregated_class_ids = [], [], []
                print("切块后未检测到聚合对象，或NMS后无剩余对象。")
        else: # 整图推理
            print("--- 开始整图检测 (使用新ONNX模型适配的预处理和后处理) ---")
            t_s = time.time()
            input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(
                original_image, (model_input_h_ref, model_input_w_ref)
            )
            timing_profile['3a_fullimg_preprocessing'] = time.time() - t_s
            t_s = time.time()
            outputs_main = session.run(None, {input_name: input_tensor})
            timing_profile['3b_fullimg_inference'] = time.time() - t_s
            t_s = time.time()
            detections_result_list = postprocess_yolo_onnx_for_main(
                outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
                original_image.shape[:2], (model_input_h_ref, model_input_w_ref),
                ratio_main, pad_x_main, pad_y_main, num_classes=len(class_names)
            )
            timing_profile['3c_fullimg_postprocessing'] = time.time() - t_s
            aggregated_boxes = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]
            aggregated_scores = [d[4] for d in detections_result_list]
            aggregated_class_ids = [d[5] for d in detections_result_list]
            num_tiles_processed = 1
            print(f"整图处理与后处理完成。找到了 {len(aggregated_boxes)} 个框。")

        if len(aggregated_boxes) > 0 and \
           ((MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0) or \
            actual_max_area_threshold_px is not None):
            t_start_area_filter = time.time()
            filtered_by_area_boxes, filtered_by_area_scores, filtered_by_area_ids = [], [], []
            initial_box_count_before_area_filter = len(aggregated_boxes)
            for i, box in enumerate(aggregated_boxes):
                b_w, b_h = box[2] - box[0], box[3] - box[1]
                area = b_w * b_h
                valid_area = True
                if MIN_DETECTION_AREA is not None and MIN_DETECTION_AREA > 0 and area < MIN_DETECTION_AREA:
                    valid_area = False
                if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px:
                    valid_area = False
                if valid_area:
                    filtered_by_area_boxes.append(box)
                    filtered_by_area_scores.append(aggregated_scores[i])
                    filtered_by_area_ids.append(aggregated_class_ids[i])
            aggregated_boxes = filtered_by_area_boxes
            aggregated_scores = filtered_by_area_scores
            aggregated_class_ids = filtered_by_area_ids
            timing_profile['5_area_filtering'] = time.time() - t_start_area_filter
            print(f"面积筛选完成 (从 {initial_box_count_before_area_filter} 减少到 {len(aggregated_boxes)} 个框).")
        else:
            timing_profile['5_area_filtering'] = 0
            if len(aggregated_boxes) > 0:
                print("面积筛选未启用或不适用。")

        # --- V2.5: OCR 与条形码识别循环 ---
        ocr_texts_for_drawing = [] # Stores the final recognized value for drawing
        recognized_obu_data_list = []

        if len(aggregated_boxes) > 0:
            print(f"--- 最终检测到 {len(aggregated_boxes)} 个OBU的YOLO框, 开始精确裁剪、识别数字与条形码 ---")
            t_recognition_total_start = time.time()

            for i, yolo_box_coords in enumerate(aggregated_boxes):
                class_id = int(aggregated_class_ids[i])
                class_name_str = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"ClassID:{class_id}"
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = [int(c) for c in yolo_box_coords]
                h_yolo = y2_yolo - y1_yolo
                w_yolo = x2_yolo - x1_yolo

                current_box_info = {
                    "roi_index": i + 1,
                    "class": class_name_str,
                    "bbox_yolo": [x1_yolo, y1_yolo, x2_yolo, y2_yolo],
                    "confidence_yolo": float(aggregated_scores[i]),
                    "bbox_digit_final": None,
                    "ocr_attempts": {}, # Stores {"preprocess_type": {"text": "...", "confidence": ...}}
                    "ocr_final_text": "N/A",
                    "ocr_final_confidence": 0.0,
                    "bbox_barcode_final": None,
                    "barcode_text": "N/A",
                    "barcode_type": "N/A",
                    "final_recognized_value": "N/A",
                    "recognition_source": "N/A" # 'OCR', 'BARCODE', 'OCR_AND_BARCODE_MATCH', 'CONFLICT_BC_PRIO', 'NONE'
                }
                print(f"  OBU ROI {current_box_info['roi_index']} (YOLO Box): 类别='{current_box_info['class']}', "
                      f"边界框={current_box_info['bbox_yolo']}, YOLO置信度={current_box_info['confidence_yolo']:.2f}")

                # --- 1. 提取并识别数字区域 (OCR) ---
                y1_digit_abs = y1_yolo + int(h_yolo * DIGIT_SUBAREA_Y_START_RATIO)
                h_digit_abs = int(h_yolo * DIGIT_SUBAREA_HEIGHT_RATIO)
                y2_digit_abs = y1_digit_abs + h_digit_abs
                cx_yolo_center = x1_yolo + w_yolo / 2.0 # Ensure float division for center
                w_digit_expanded = int(w_yolo * DIGIT_SUBAREA_WIDTH_EXPAND_FACTOR)
                x1_digit_abs = int(cx_yolo_center - w_digit_expanded / 2.0)
                x2_digit_abs = int(cx_yolo_center + w_digit_expanded / 2.0)

                x1_d_clip = max(0, x1_digit_abs)
                y1_d_clip = max(0, y1_digit_abs)
                x2_d_clip = min(orig_img_w, x2_digit_abs)
                y2_d_clip = min(orig_img_h, y2_digit_abs)
                current_box_info["bbox_digit_final"] = [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip]
                print(f"    Calculated Digit Sub-ROI for OCR: {current_box_info['bbox_digit_final']}")

                if paddle_ocr_engine and (x2_d_clip > x1_d_clip) and (y2_d_clip > y1_d_clip):
                    digit_roi_color_ocr = original_image[y1_d_clip:y2_d_clip, x1_d_clip:x2_d_clip]
                    if digit_roi_color_ocr.size == 0:
                        print("    Warning: Digit Sub-ROI for OCR is empty after clipping.")
                    else:
                        cv2.imwrite(os.path.join(process_photo_dir, f"digit_roi_{current_box_info['roi_index']:03d}_ocr_color.png"), digit_roi_color_ocr)
                        digit_roi_gray_ocr = cv2.cvtColor(digit_roi_color_ocr, cv2.COLOR_BGR2GRAY)
                        # Otsu's binarization - useful for well-contrasted numbers
                        # INV might be needed if numbers are light on dark background. Test this.
                        # For镭标码, numbers are often dark on light metallic background. So THRESH_BINARY_INV might be wrong.
                        # Let's try THRESH_BINARY first, or adaptive.
                        _, digit_roi_otsu_ocr = cv2.threshold(digit_roi_gray_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # Adaptive thresholding can be better for uneven illumination
                        digit_roi_adaptive_ocr = cv2.adaptiveThreshold(digit_roi_gray_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                     cv2.THRESH_BINARY, 11, 2) # Block size 11, C 2

                        # Save processed images for ocr
                        cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_ocr_gray.png"),digit_roi_gray_ocr)
                        cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_ocr_binary_otsu.png"),digit_roi_otsu_ocr)
                        cv2.imwrite(os.path.join(process_photo_dir,f"digit_roi_{current_box_info['roi_index']:03d}_ocr_binary_adaptive.png"),digit_roi_adaptive_ocr)

                        images_to_ocr_map = {
                            "color_digit": digit_roi_color_ocr,
                            "gray_digit": digit_roi_gray_ocr,
                            "binary_otsu_digit": digit_roi_otsu_ocr, # Using non-inverted Otsu
                            "binary_adaptive_digit": digit_roi_adaptive_ocr # Using non-inverted adaptive
                        }

                        ocr_inputs_to_try_cfg = []
                        if OCR_PREPROCESS_TYPE_TO_USE == "all": ocr_inputs_to_try_cfg = list(images_to_ocr_map.keys())
                        elif isinstance(OCR_PREPROCESS_TYPE_TO_USE, str) and OCR_PREPROCESS_TYPE_TO_USE in images_to_ocr_map:
                            ocr_inputs_to_try_cfg = [OCR_PREPROCESS_TYPE_TO_USE]
                        elif isinstance(OCR_PREPROCESS_TYPE_TO_USE, list):
                            ocr_inputs_to_try_cfg = [key for key in OCR_PREPROCESS_TYPE_TO_USE if key in images_to_ocr_map]
                        if not ocr_inputs_to_try_cfg:
                            ocr_inputs_to_try_cfg = ["binary_otsu_digit"]
                            print(f"警告: OCR_PREPROCESS_TYPE_TO_USE 配置无效，默认尝试 'binary_otsu_digit'")

                        best_ocr_text_this_roi = "N/A"
                        highest_ocr_conf_this_roi = 0.0
                        for preprocess_key in ocr_inputs_to_try_cfg:
                            img_for_ocr = images_to_ocr_map[preprocess_key]
                            if img_for_ocr.ndim == 2: img_for_ocr = cv2.cvtColor(img_for_ocr, cv2.COLOR_GRAY2BGR)

                            print(f"    Attempting OCR with {preprocess_key} for Digit Sub-ROI {current_box_info['roi_index']}...")
                            try:
                                # PaddleOCR expects BGR image
                                ocr_result_paddle = paddle_ocr_engine.ocr(img_for_ocr, cls=False) # cls=False for number recognition
                                if ocr_result_paddle and ocr_result_paddle[0] is not None: # New format: [[line1_info, line2_info,...]]
                                    line_results = ocr_result_paddle[0]
                                    current_text_parts = []
                                    current_conf_parts = []
                                    for line_info in line_results: # line_info is [bbox_coords, (text, confidence)]
                                        text_segment, segment_conf = line_info[1]
                                        # Filter common OCR noise for numbers if needed, e.g., remove spaces, non-digits
                                        text_segment_cleaned = ''.join(filter(str.isalnum, text_segment)).upper() # Keep alnum, to upper
                                        if text_segment_cleaned: # Only add if there's content after cleaning
                                            current_text_parts.append(text_segment_cleaned)
                                            current_conf_parts.append(segment_conf)

                                    if current_text_parts:
                                        full_rec_text = "".join(current_text_parts)
                                        avg_rec_conf = sum(current_conf_parts) / len(current_conf_parts) if current_conf_parts else 0.0
                                        print(f"      PaddleOCR ({preprocess_key}) Result: '{full_rec_text}' (Avg Conf: {avg_rec_conf:.2f})")
                                        current_box_info["ocr_attempts"][preprocess_key] = {"text": full_rec_text, "confidence": avg_rec_conf}
                                        if best_ocr_text_this_roi == "N/A" or avg_rec_conf > highest_ocr_conf_this_roi:
                                            best_ocr_text_this_roi = full_rec_text
                                            highest_ocr_conf_this_roi = avg_rec_conf
                                    else:
                                        print(f"      PaddleOCR ({preprocess_key}): No valid text segments found after cleaning.")
                                else:
                                    print(f"      PaddleOCR ({preprocess_key}): No text detected or result is None.")
                            except Exception as ocr_e:
                                print(f"      Error during PaddleOCR ({preprocess_key}): {ocr_e}")
                                # traceback.print_exc()

                        current_box_info["ocr_final_text"] = best_ocr_text_this_roi
                        current_box_info["ocr_final_confidence"] = highest_ocr_conf_this_roi
                        if best_ocr_text_this_roi != "N/A":
                            print(f"    Best OCR for Digit Sub-ROI {current_box_info['roi_index']}: '{best_ocr_text_this_roi}' (Conf: {highest_ocr_conf_this_roi:.2f})")
                        else:
                            print(f"    OCR for Digit Sub-ROI {current_box_info['roi_index']} failed or no text found.")
                else:
                    if not paddle_ocr_engine: print("    PaddleOCR engine not available. Skipping OCR.")
                    else: print(f"    Skipping OCR for invalid/zero-size or empty Digit Sub-ROI.")


                # --- 2. 提取并识别条形码区域 ---
                y1_barcode_abs = y1_yolo + int(h_yolo * BARCODE_SUBAREA_Y_START_RATIO)
                h_barcode_abs = int(h_yolo * BARCODE_SUBAREA_HEIGHT_RATIO)
                y2_barcode_abs = y1_barcode_abs + h_barcode_abs
                w_barcode_expanded = int(w_yolo * BARCODE_SUBAREA_WIDTH_EXPAND_FACTOR)
                x1_barcode_abs = int(cx_yolo_center - w_barcode_expanded / 2.0)
                x2_barcode_abs = int(cx_yolo_center + w_barcode_expanded / 2.0)

                x1_b_clip = max(0, x1_barcode_abs)
                y1_b_clip = max(0, y1_barcode_abs)
                x2_b_clip = min(orig_img_w, x2_barcode_abs)
                y2_b_clip = min(orig_img_h, y2_barcode_abs)
                current_box_info["bbox_barcode_final"] = [x1_b_clip, y1_b_clip, x2_b_clip, y2_b_clip]
                print(f"    Calculated Barcode Sub-ROI for decoding: {current_box_info['bbox_barcode_final']}")

                if (x2_b_clip > x1_b_clip) and (y2_b_clip > y1_b_clip):
                    barcode_roi_color = original_image[y1_b_clip:y2_b_clip, x1_b_clip:x2_b_clip]
                    if barcode_roi_color.size == 0:
                        print("    Warning: Barcode Sub-ROI for decoding is empty after clipping.")
                    else:
                        cv2.imwrite(os.path.join(process_photo_dir, f"barcode_roi_{current_box_info['roi_index']:03d}_decode_color.png"), barcode_roi_color)
                        print(f"    Attempting Barcode decoding for Sub-ROI {current_box_info['roi_index']}...")
                        barcode_text, barcode_type = decode_barcode_from_image_roi(barcode_roi_color, current_box_info['roi_index'])
                        if barcode_text:
                            current_box_info["barcode_text"] = barcode_text
                            current_box_info["barcode_type"] = barcode_type
                            print(f"      Barcode Result: '{barcode_text}' (Type: {barcode_type})")
                        else:
                            print(f"      Barcode decoding failed or no barcode found.")
                else:
                    print(f"    Skipping Barcode decoding for invalid/zero-size or empty Barcode Sub-ROI.")

                # --- 3. 整合识别结果 ---
                ocr_val = current_box_info["ocr_final_text"]
                barcode_val = current_box_info["barcode_text"]

                # Simple cleaning: ensure both are uppercase and alphanumeric for comparison
                ocr_val_cleaned = ''.join(filter(str.isalnum, ocr_val)).upper() if ocr_val != "N/A" else "N/A"
                barcode_val_cleaned = ''.join(filter(str.isalnum, barcode_val)).upper() if barcode_val != "N/A" else "N/A"

                if barcode_val_cleaned != "N/A":
                    current_box_info["final_recognized_value"] = barcode_val_cleaned # Use cleaned version
                    current_box_info["recognition_source"] = "BARCODE"
                    if ocr_val_cleaned != "N/A":
                        if barcode_val_cleaned == ocr_val_cleaned:
                            current_box_info["recognition_source"] = "OCR_AND_BARCODE_MATCH"
                            print(f"    整合结果: OCR与条形码匹配: '{barcode_val_cleaned}'")
                        else:
                            current_box_info["recognition_source"] = "CONFLICT_BC_PRIO" # Barcode priority in case of conflict
                            print(f"    整合结果: OCR ('{ocr_val_cleaned}') 与 条形码 ('{barcode_val_cleaned}') 不匹配. 优先使用条形码.")
                    else: # Only barcode result
                         print(f"    整合结果: 仅条形码识别: '{barcode_val_cleaned}'")
                elif ocr_val_cleaned != "N/A": # Only OCR result
                    current_box_info["final_recognized_value"] = ocr_val_cleaned
                    current_box_info["recognition_source"] = "OCR"
                    print(f"    整合结果: 仅OCR识别: '{ocr_val_cleaned}'")
                else: # Both failed
                    current_box_info["recognition_source"] = "NONE"
                    print(f"    整合结果: OCR与条形码识别均失败.")

                # Update text for drawing
                if current_box_info["final_recognized_value"] != "N/A":
                    ocr_texts_for_drawing.append(current_box_info["final_recognized_value"])
                else:
                    ocr_texts_for_drawing.append("FAIL")

                recognized_obu_data_list.append(current_box_info)
                print("-" * 30)

            timing_profile['7_recognition_total'] = time.time() - t_recognition_total_start
            print(f"--- 所有ROI的OCR与条形码识别处理完成 ({timing_profile['7_recognition_total']:.3f} 秒) ---")

            print("\n--- 识别结果概要 ---")
            for obu_data in recognized_obu_data_list:
                yolo_bbox_str = f"YOLO: {obu_data['bbox_yolo']}"
                ocr_str = f"OCR: '{obu_data['ocr_final_text']}'(C:{obu_data['ocr_final_confidence']:.2f})"
                barcode_str = f"BC: '{obu_data['barcode_text']}'(T:{obu_data['barcode_type']})"
                final_str = f"==> Final: '{obu_data['final_recognized_value']}' (Src: {obu_data['recognition_source']})"
                print(f"ROI {obu_data['roi_index']:02d} | {yolo_bbox_str} | {ocr_str} | {barcode_str} | {final_str}")
            print("---------------------------------------\n")

            t_start_drawing = time.time()
            output_img_to_draw_on = original_image.copy()
            output_img_to_draw_on = draw_detections(
                output_img_to_draw_on,
                np.array(aggregated_boxes),
                np.array(aggregated_scores),
                np.array(aggregated_class_ids),
                class_names,
                ocr_texts=ocr_texts_for_drawing,
                roi_indices=[item["roi_index"] for item in recognized_obu_data_list]
            )
            timing_profile['8_drawing_results_final'] = time.time() - t_start_drawing
            output_fn_base = os.path.splitext(os.path.basename(IMAGE_NAME))[0]

            # Generate a clean version string for filename, e.g., v2.5.0
            version_clean_for_filename = VERSION.split('_')[0] if '_' in VERSION else VERSION

            output_fn = f"output_{output_fn_base}_{version_clean_for_filename}.png"
            cv2.imwrite(output_fn, output_img_to_draw_on)
            print(f"最终结果图已保存到: {output_fn} ({timing_profile['8_drawing_results_final']:.2f} 秒用于绘图)")
        else:
            print("最终未检测到任何OBU ROI，无法进行识别。")
            timing_profile['7_recognition_total'] = 0
            timing_profile['8_drawing_results_final'] = 0

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
    except Exception as e:
        print(f"发生未处理的错误: {e}")
        traceback.print_exc()
    finally:
        timing_profile['0_total_script_execution'] = time.time() - t_start_overall
        print(f"\n--- 时间分析概要 ({VERSION}) ---")
        # Sort timing profile by key (stage number) before printing
        sorted_timing_keys = sorted(timing_profile.keys(), key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 99)
        for stage_key in sorted_timing_keys:
            duration = timing_profile[stage_key]
            print(f"  {stage_key}: {duration:.3f} 秒")
        print(f"------------------------------")