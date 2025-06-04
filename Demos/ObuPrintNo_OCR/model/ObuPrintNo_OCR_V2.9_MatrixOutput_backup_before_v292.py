# coding: utf-8
"""
OBU (车载单元) 镭标码识别与矩阵输出脚本
版本: v2.9.1_Axis_And_Layout_Inference
功能:
- 利用YOLO检测条码作为OBU锚点。
- 智能识别包含2个OBU的特殊行（起始或末尾）。
- 基于特殊行和50个OBU (12*4+2)的布局先验，推断理想网格。
- PaddleOCR识别数字，并将其填充到推断的网格中。
- 输出最终的OBU矩阵。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr
import onnxruntime # 用于YOLO
from itertools import product
import csv
from datetime import datetime
from collections import Counter

# --- V2.9.1 配置参数 ---
VERSION = "v2.9.1_Axis_And_Layout_Inference"
IMAGE_PATHS = [
    r"./PIC/1.JPG",
    r"./PIC/2.JPG",
    r"./PIC/3.JPG",
]
BASE_OUTPUT_DIR = "./output_v2.9_axis_matrix"
TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP_NOW}_{VERSION}")
LOG_FILE_PATH = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"矩阵日志_{VERSION}_{TIMESTAMP_NOW}.csv") # 可选
os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True)

# --- 策略选择 (本版本专注于YOLO辅助的轴心标定方案) ---
MATRIX_BUILD_STRATEGY = "YOLO_AXIS_ASSISTED"

# --- PaddleOCR 初始化相关参数 ---
LANG_CFG = 'en'
USE_TEXTLINE_ORIENTATION_CFG = False
USE_DOC_ORIENTATION_CLASSIFY_CFG = False
USE_DOC_UNWARPING_CFG = False
OCR_VERSION_CFG = None
TEXT_DETECTION_MODEL_DIR_CFG = None
TEXT_RECOGNITION_MODEL_DIR_CFG = None
TEXT_DETECTION_MODEL_NAME_CFG = None
TEXT_RECOGNITION_MODEL_NAME_CFG = None
PADDLE_OCR_FINE_PARAMS = {
    "text_det_limit_side_len": 960, "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.6, "text_rec_score_thresh": 0.5,
}

# --- OBU码筛选规则 (PaddleOCR后处理) ---
OBU_CODE_PREFIX_FILTER_CFG = "5001"
OBU_CODE_LENGTH_FILTER_CFG = 16

# --- YOLOv8 相关配置 ---
YOLO_ONNX_MODEL_PATH_CFG = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
YOLO_CONFIDENCE_THRESHOLD_CFG = 0.25
YOLO_IOU_THRESHOLD_CFG = 0.45
YOLO_INPUT_WIDTH_CFG = 640
YOLO_INPUT_HEIGHT_CFG = 640

# --- 矩阵与布局先验配置 ---
LAYOUT_CONFIG = {
    "total_obus": 50,       # 期望的总OBU数量
    "regular_rows": 12,     # 常规行的数量
    "regular_cols": 4,      # 常规行每行的OBU数量
    "special_row_exists": True, # <--- 新增并设置为 True
    "special_row_cols": 2,  # 特殊行（只有2个OBU）的列数
    # "special_row_is_last": True # 这个仍然由程序动态判断，所以可以不在这里预设
}

# 行分组和匹配的阈值参数
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.6 # YOLO条码框行分组时，Y坐标差异阈值 = 平均框高 * 此因子
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.8 # PaddleOCR识别的数字中心与理想格点中心的最大匹配距离因子 (乘以平均OBU宽度)
# 预期的数字区域相对于YOLO条码框的偏移和尺寸 (用于匹配)
# 这些值需要根据您的实际OBU上数字和条码的相对位置仔细调整
DIGIT_AREA_RELATIVE_Y_OFFSET = -0.8  # 数字区域中心相对于条码框中心的Y偏移因子 (乘以条码框高度，负数表示在条码框上方)
DIGIT_AREA_SEARCH_HEIGHT_FACTOR = 1.0 # 数字区域搜索高度的因子 (乘以条码框高度)
DIGIT_AREA_SEARCH_WIDTH_FACTOR = 1.2  # 数字区域搜索宽度的因子 (乘以条码框宽度)


# --- 全局变量 ---
paddle_ocr_engine_global = None
yolo_session_global = None
# (CSV_HEADER 可以根据需要调整)

# --- 函数定义 ---
def initialize_paddleocr():
    global paddle_ocr_engine_global
    init_params = {'lang': LANG_CFG, 'use_textline_orientation': USE_TEXTLINE_ORIENTATION_CFG,
                   'use_doc_orientation_classify': USE_DOC_ORIENTATION_CLASSIFY_CFG,
                   'use_doc_unwarping': USE_DOC_UNWARPING_CFG, 'ocr_version': OCR_VERSION_CFG,
                   'text_detection_model_dir': TEXT_DETECTION_MODEL_DIR_CFG,
                   'text_recognition_model_dir': TEXT_RECOGNITION_MODEL_DIR_CFG,
                   'text_detection_model_name': TEXT_DETECTION_MODEL_NAME_CFG,
                   'text_recognition_model_name': TEXT_RECOGNITION_MODEL_NAME_CFG,
                   **PADDLE_OCR_FINE_PARAMS }
    ocr_params_final_filtered = {k: v for k, v in init_params.items() if v is not None}
    print(f"\n正在使用以下参数初始化PaddleOCR: {ocr_params_final_filtered}")
    try:
        paddle_ocr_engine_global = paddleocr.PaddleOCR(**ocr_params_final_filtered); print("PaddleOCR引擎初始化成功。"); return True
    except Exception as e: print(f"PaddleOCR引擎初始化失败: {e}"); paddle_ocr_engine_global = None; return False

def load_yolo_model():
    global yolo_session_global
    if not os.path.exists(YOLO_ONNX_MODEL_PATH_CFG): print(f"错误: YOLO ONNX模型未找到: {YOLO_ONNX_MODEL_PATH_CFG}"); return False
    try:
        print(f"正在加载YOLO模型: {YOLO_ONNX_MODEL_PATH_CFG}")
        yolo_session_global = onnxruntime.InferenceSession(YOLO_ONNX_MODEL_PATH_CFG, providers=['CPUExecutionProvider'])
        print("YOLO模型加载成功。"); return True
    except Exception as e: print(f"YOLO模型加载失败: {e}"); yolo_session_global = None; return False

# --- YOLO 相关函数 (来自您V2.5.0的提供) ---
def preprocess_for_yolo(img_data, target_h, target_w): # 原名 preprocess_onnx_for_main
    img_height_orig, img_width_orig = img_data.shape[:2]
    ratio = min(target_w / img_width_orig, target_h / img_height_orig)
    new_w, new_h = int(img_width_orig * ratio), int(img_height_orig * ratio)
    resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2; pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    input_tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, ratio, pad_x, pad_y

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold): # 确保此函数存在且正确
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    if not isinstance(scores, np.ndarray) or scores.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]; areas=(x2-x1+1e-6)*(y2-y1+1e-6); order=scores.argsort()[::-1]; keep=[] #加epsilon防面积为0
    while order.size > 0:
        i = order[0]; keep.append(i);_ = order.size;order = order[1:]
        if _ == 1: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order]);xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h;ovr=inter/(areas[i]+areas[order]-inter+1e-6)
        inds=np.where(ovr<=iou_threshold)[0];order=order[inds]
    return keep

def postprocess_yolo_detections(outputs_onnx, conf_threshold, iou_threshold,
                                   original_shape_hw, model_input_shape_hw,
                                   ratio_preproc, pad_x_preproc, pad_y_preproc): # 原名 postprocess_yolo_onnx_for_main
    raw_output_tensor = np.squeeze(outputs_onnx[0])
    if raw_output_tensor.ndim != 2: print(f"错误: YOLO输出张量维度不为2. Shape: {raw_output_tensor.shape}"); return []
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor
    boxes_candidate, scores_candidate = [], [] # 只收集框和分数给NMS
    expected_attributes = 4 + 1
    for pred_data in predictions_to_iterate:
        if len(pred_data) != expected_attributes: continue
        final_confidence = float(pred_data[4])
        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = pred_data[:4]
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2]); scores_candidate.append(final_confidence)
    if not boxes_candidate: return []
    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)

    final_barcode_boxes_xyxy = [] # 只返回在原图坐标系下的 [x1,y1,x2,y2]
    orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx); box_model_coords = boxes_candidate[idx]
        # 从模型坐标空间转换回原图无填充区域的坐标
        box_no_pad_x1 = box_model_coords[0] - pad_x_preproc
        box_no_pad_y1 = box_model_coords[1] - pad_y_preproc
        box_no_pad_x2 = box_model_coords[2] - pad_x_preproc
        box_no_pad_y2 = box_model_coords[3] - pad_y_preproc
        # 反推回原始图像尺寸的坐标
        if ratio_preproc == 0: continue
        orig_x1 = box_no_pad_x1 / ratio_preproc
        orig_y1 = box_no_pad_y1 / ratio_preproc
        orig_x2 = box_no_pad_x2 / ratio_preproc
        orig_y2 = box_no_pad_y2 / ratio_preproc
        # 裁剪到图像边界并转为整数
        final_x1 = int(np.clip(orig_x1, 0, orig_w))
        final_y1 = int(np.clip(orig_y1, 0, orig_h))
        final_x2 = int(np.clip(orig_x2, 0, orig_w))
        final_y2 = int(np.clip(orig_y2, 0, orig_h))
        final_barcode_boxes_xyxy.append([final_x1, final_y1, final_x2, final_y2])
    return final_barcode_boxes_xyxy

def get_yolo_barcode_anchors(image): # 与V2.9.0框架一致
    # ... (代码与V2.9.0框架中的 get_yolo_barcode_anchors 一致) ...
    if not yolo_session_global: print("错误: YOLO会话未初始化。"); return [], 0.0
    input_tensor, ratio, pad_x, pad_y = preprocess_for_yolo(image, YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG)
    input_name = yolo_session_global.get_inputs()[0].name
    t_start = time.time(); outputs = yolo_session_global.run(None, {input_name: input_tensor}); yolo_predict_time = time.time() - t_start
    print(f"  YOLO predict() 耗时 {yolo_predict_time:.3f}s")
    detected_barcode_boxes = postprocess_yolo_detections(
        outputs_onnx=outputs,  # 明确指定参数名可以避免顺序问题，或者严格按顺序
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD_CFG,
        iou_threshold=YOLO_IOU_THRESHOLD_CFG,
        original_shape_hw=image.shape[:2],
        model_input_shape_hw=(YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG),
        ratio_preproc=ratio,
        pad_x_preproc=pad_x,
        pad_y_preproc=pad_y
    )
    print(f"  YOLO检测到 {len(detected_barcode_boxes)} 个条码框。")
    return detected_barcode_boxes, yolo_predict_time


def get_box_center_and_dims(box_xyxy_or_poly):
    """计算框的中心点 (cx, cy), 宽度 w, 高度 h"""
    # 修正对 NumPy 数组的有效性检查
    if box_xyxy_or_poly is None: # 首先检查是否为None
        return None, None, None, None
    if isinstance(box_xyxy_or_poly, (list, np.ndarray)) and len(box_xyxy_or_poly) == 0: # 检查是否为空列表或数组
        return None, None, None, None
    # 对于 NumPy 数组，如果想检查它是否“有内容”，通常是检查其 .size 或特定条件
    # 但对于我们的box数据，主要是检查它是否None或长度为0
    # 如果它是一个非空的NumPy数组，后续的类型和长度检查会处理它

    # 后续的类型和长度检查保持不变
    if len(box_xyxy_or_poly) == 4 and isinstance(box_xyxy_or_poly[0], (int, float)): # xyxy
        x1, y1, x2, y2 = box_xyxy_or_poly
        return int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2-x1), int(y2-y1)
    elif isinstance(box_xyxy_or_poly, (list, np.ndarray)) and \
         len(box_xyxy_or_poly) > 0 and \
         isinstance(box_xyxy_or_poly[0], (list, np.ndarray)): # polygon
        points = np.array(box_xyxy_or_poly, dtype=np.int32)
        if len(points) > 0: # 确保转换后的points不为空
            x, y, w, h = cv2.boundingRect(points) # 使用外接矩形来估计尺寸和中心
            return x + w // 2, y + h // 2, w, h

    print(f"警告: get_box_center_and_dims 接收到无法解析的box格式: {box_xyxy_or_poly}") # 增加一个警告
    return None, None, None, None

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.png"):
    """
    在图片上绘制PaddleOCR的原始检测结果和筛选后的OBU结果。
    Args:
        image (numpy.ndarray): 原始OpenCV图像 (BGR格式)。
        all_ocr_data (list): 包含所有原始提取OCR数据的列表。
                             每个元素是一个字典: {"text": str, "score": float, "box": list_of_points}
        potential_obu_data (list): 包含筛选后认为是OBU码的数据的列表，结构同上。
        output_path (str): 可视化结果图片的保存路径。
    """
    img_out = image.copy()
    _c = cv2 # 使用别名简化后续cv2的调用

    if img_out is None:
        print(f"错误: 用于绘制的输入图像为None。无法保存到 {output_path}")
        return

    if not all_ocr_data and not potential_obu_data :
        print(f"没有OCR数据可以绘制到 {output_path}.")
        try:
            _c.imwrite(output_path, img_out)
            print(f"无OCR数据, 底图已保存到: {output_path}")
        except Exception as e_save:
            print(f"保存底图失败 {output_path}: {e_save}")
        return

    # 1. 绘制所有原始检测框 (用细的绿色线条)
    if all_ocr_data:
        for item in all_ocr_data:
            box_polygon = item.get('box') # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            if box_polygon is not None and isinstance(box_polygon, (list, np.ndarray)) and len(box_polygon) > 0 :
                try:
                    points = np.array(box_polygon, dtype=np.int32)
                    _c.polylines(img_out, [points], isClosed=True, color=(0, 180, 0), thickness=1) # 绿色细线
                except Exception as e_draw_poly:
                    print(f"警告: 无法为检测框 {box_polygon} 绘制多边形. 错误: {e_draw_poly}")

    # 2. 绘制筛选出的Potential OBU (用粗的蓝色框，并标注识别文本)
    drawn_potential_text_count = 0
    if potential_obu_data:
        for item in potential_obu_data:
            text = item['text']
            box_polygon = item.get('box')
            if box_polygon is None or not isinstance(box_polygon, (list, np.ndarray)) or len(box_polygon) == 0:
                continue
            try:
                points = np.array(box_polygon, dtype=np.int32)
                _c.polylines(img_out, [points], isClosed=True, color=(255, 0, 0), thickness=3) # 蓝色粗线

                label = f"{text}" # 可以加上分数: f"{text} ({item['score']:.2f})"

                text_anchor_x = points[0][0]
                text_anchor_y = points[0][1] - 10
                if text_anchor_y < 15 : text_anchor_y = points[0][1] + 25

                (text_width, text_height), baseline = _c.getTextSize(label, _c.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                _c.rectangle(img_out,
                             (text_anchor_x, text_anchor_y - text_height - baseline + 1),
                             (text_anchor_x + text_width, text_anchor_y + baseline -1),
                             (220,220,220), -1)
                _c.putText(img_out, label, (text_anchor_x, text_anchor_y),
                            _c.FONT_HERSHEY_SIMPLEX, 0.8, (180, 0, 0), 2)
                drawn_potential_text_count +=1
            except Exception as e_draw_potential:
                print(f"警告: 无法为潜在OBU绘制检测框 {box_polygon}. 错误: {e_draw_potential}")

        if drawn_potential_text_count > 0:
            print(f"已在图上绘制 {drawn_potential_text_count} 个潜在OBU的文本。")

    try:
        _c.imwrite(output_path, img_out)
        print(f"OCR可视化结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存可视化图片失败 {output_path}: {e}")

# --- 核心：矩阵构建与填充 ---
def build_matrix_yolo_axis_calibrated(yolo_boxes, paddle_results, layout_config, image_shape_hw):
    """
    基于YOLO锚点、特殊行识别和布局先验来构建和填充OBU矩阵。
    """
    print("  正在使用YOLO锚点、特殊行和布局先验构建矩阵...")
    if not yolo_boxes:
        print("  YOLO未检测到条码，无法构建矩阵。")
        return [["YOLO无检测"] * layout_config["regular_cols"]], 0 # 返回一个占位矩阵

    # 1. 为YOLO框计算中心点和尺寸
    yolo_anchors = []
    for i, box in enumerate(yolo_boxes): # box is [x1,y1,x2,y2]
        cx, cy, w, h = get_box_center_and_dims(box)
        if cx is not None:
            yolo_anchors.append({'id': i, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box})

    if not yolo_anchors: print("  未能从YOLO框中计算出有效锚点。"); return [["无有效锚点"]],0

    # 2. 对YOLO锚点按Y坐标进行初步行分组
    yolo_anchors.sort(key=lambda a: a['cy']) # 主要按Y排序
    yolo_rows_grouped = []
    if not yolo_anchors: return [["YOLO锚点排序后为空"]],0

    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors if a['h'] > 0]) if any(a['h'] > 0 for a in yolo_anchors) else 30
    y_threshold = avg_h_yolo * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    current_row_for_grouping = [yolo_anchors[0]]
    for i in range(1, len(yolo_anchors)):
        if abs(yolo_anchors[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold:
            current_row_for_grouping.append(yolo_anchors[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx'])) # 行内按X排序
            current_row_for_grouping = [yolo_anchors[i]]
    yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))

    print(f"  YOLO锚点初步分为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")

    # 3. 识别特殊行 (2个OBU的行) 并确定其位置 (顶部或底部)
    #    并推断常规列数
    special_row_candidate_indices = [i for i, r in enumerate(yolo_rows_grouped) if len(r) == layout_config["special_row_cols"]]
    regular_row_col_counts = [len(r) for i, r in enumerate(yolo_rows_grouped) if i not in special_row_candidate_indices]

    inferred_regular_cols = layout_config["regular_cols"] # 默认值
    if regular_row_col_counts:
        col_mode = Counter(regular_row_col_counts).most_common(1)[0][0]
        if col_mode > 0 : inferred_regular_cols = col_mode
        print(f"  从常规行推断出的主要列数为: {inferred_regular_cols}")
    else: # 如果所有行都像特殊行，或者只有特殊行
        print(f"  警告: 未找到足够的常规行来推断列数，将使用默认常规列数: {inferred_regular_cols}")


    # 确定特殊行的位置
    # 简化逻辑：假设特殊行如果存在，只会在最开始或最末尾，并且其数量与预设的 special_row_cols 严格匹配
    # 并且总行数应该等于 regular_rows + 1 (如果 special_row_exists)
    num_expected_total_rows = layout_config["regular_rows"] + (1 if layout_config["special_row_exists"] else 0)

    # (这个特殊行识别和布局推断是核心难点，以下是非常初步的逻辑，需要大力优化)
    # 我们需要一个更鲁棒的方法来根据 yolo_rows_grouped 和 layout_config 生成50个理想坑位坐标
    # 此处暂时简化，直接使用yolo_rows_grouped的结构来构建矩阵，后续再替换为精确的50坑位推断

    matrix_rows = len(yolo_rows_grouped)
    matrix_cols = inferred_regular_cols # 暂用推断的常规列数作为矩阵宽度
    if matrix_rows == 0 : return [["无YOLO行"]],0

    # 修正：如果特殊行存在，且其列数不等于常规列数，矩阵宽度可能需要取所有行中最大的列数
    all_col_counts_in_yolo_rows = [len(r) for r in yolo_rows_grouped]
    if all_col_counts_in_yolo_rows:
        matrix_cols = max(all_col_counts_in_yolo_rows)


    final_matrix = [["未识别"] * matrix_cols for _ in range(matrix_rows)]
    matrix_filled_count = 0

    paddle_results_with_center = []
    for pr in paddle_results:
        cx, cy, pw, ph = get_box_center_and_dims(pr['box'])
        if cx is not None: paddle_results_with_center.append({**pr, 'cx': cx, 'cy': cy, 'w':pw, 'h':ph, 'used': False})

    # 填充逻辑：遍历YOLO推断出的每个格子，找最佳PaddleOCR匹配
    for r_idx, yolo_row_items in enumerate(yolo_rows_grouped):
        for c_idx, yolo_anchor in enumerate(yolo_row_items):
            if c_idx >= matrix_cols: continue

            yolo_cx, yolo_cy, yolo_w, yolo_h = yolo_anchor['cx'], yolo_anchor['cy'], yolo_anchor['w'], yolo_anchor['h']

            # 定义数字区域的预期中心 (基于YOLO条码框)
            expected_digit_cx = yolo_cx
            expected_digit_cy = yolo_cy + int(yolo_h * DIGIT_AREA_RELATIVE_Y_OFFSET) # 向上偏移

            best_match_obu = None
            min_dist_sq = (PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * yolo_w)**2 # 动态阈值

            for p_obu in paddle_results_with_center:
                if p_obu['used']: continue
                dist_sq = (p_obu['cx'] - expected_digit_cx)**2 + (p_obu['cy'] - expected_digit_cy)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_match_obu = p_obu

            if best_match_obu:
                final_matrix[r_idx][c_idx] = best_match_obu['text']
                best_match_obu['used'] = True
                matrix_filled_count += 1

    print(f"  YOLO轴心标定方案: 构建矩阵 {matrix_rows}x{matrix_cols}, 填充OBU数: {matrix_filled_count}")
    return final_matrix, matrix_rows, matrix_cols, matrix_filled_count


def print_matrix_to_console(matrix, strategy_name=""): # 与V2.9.0框架一致
    # ... (代码与V2.9.0框架中的 print_matrix_to_console 一致) ...
    if not matrix or (isinstance(matrix, list) and not matrix[0]): print(f"  {strategy_name} 生成的矩阵为空，无法打印。"); return
    print(f"\n--- {strategy_name} OBU识别矩阵 ({len(matrix)}行 x {len(matrix[0]) if matrix and matrix[0] else 0}列) ---")
    for row in matrix:
        row_display = ["  红  " if item == "未识别" else (f" {item[-4:]} " if isinstance(item, str) and item.startswith("5001") and len(item)>4 else f" {str(item)[:4]:^4} ") for item in row]
        print(" ".join(row_display))
    print("---------------------------------------------")

# --- (draw_ocr_results_refined 函数与V2.8.4一致，此处省略) ---
# ...

# --- 主程序 (V2.9.1) ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"--- OBU识别与矩阵输出工具 {VERSION} ---")
    print(f"输出目录: {os.path.abspath(CURRENT_RUN_OUTPUT_DIR)}")

    if not initialize_paddleocr(): exit()
    if not load_yolo_model():
        print("警告: YOLO模型加载失败。YOLO轴心标定策略将不可用。")
        # 如果YOLO是唯一策略，则退出
        if MATRIX_BUILD_STRATEGY == "YOLO_AXIS_ASSISTED": exit()

    for image_path_current in IMAGE_PATHS:
        print(f"\n\n========== 处理图片: {image_path_current} ==========")
        img_filename_base = os.path.splitext(os.path.basename(image_path_current))[0]
        original_image = cv2.imread(image_path_current)
        if original_image is None: print(f"错误: 无法读取图片 {image_path_current}"); continue

        # --- PaddleOCR 识别 ---
        print(f"\n--- 步骤1: PaddleOCR 文本检测与识别 ---")
        t_start_paddle = time.time()
        ocr_prediction_result = paddle_ocr_engine_global.predict(original_image)
        paddle_predict_time = time.time() - t_start_paddle
        print(f"PaddleOCR predict() 完成, 耗时: {paddle_predict_time:.3f}s")
        all_paddle_ocr_data = []; potential_obu_list_paddle = []
        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]
            dt_polys = ocr_result_object.get('dt_polys'); rec_texts = ocr_result_object.get('rec_texts'); rec_scores = ocr_result_object.get('rec_scores')
            if rec_texts and rec_scores and dt_polys:
                max_items = min(len(rec_texts), len(rec_scores), len(dt_polys))
                if len(rec_texts) != max_items or len(rec_scores) != max_items or len(dt_polys) != max_items : print(f"  警告: PaddleOCR原始输出长度不匹配: texts({len(rec_texts)}), scores({len(rec_scores)}), boxes({len(dt_polys)}). 按最短 {max_items} 处理。")
                for i in range(max_items):
                    item_data = {"text": str(rec_texts[i]), "score": float(rec_scores[i]), "box": dt_polys[i]}
                    all_paddle_ocr_data.append(item_data)
                    text_check = item_data['text'].strip()
                    # 使用PADDLE_OCR_FINE_PARAMS中的识别阈值进行筛选
                    if text_check.startswith(OBU_CODE_PREFIX_FILTER_CFG) and \
                       len(text_check) == OBU_CODE_LENGTH_FILTER_CFG and \
                       text_check.isdigit() and \
                       item_data['score'] >= PADDLE_OCR_FINE_PARAMS['text_rec_score_thresh']:
                        potential_obu_list_paddle.append(item_data)
        print(f"PaddleOCR 原始有效文本 {len(all_paddle_ocr_data)} 条, 内容筛选后潜在OBU {len(potential_obu_list_paddle)} 个。")

        # --- YOLO 检测 ---
        yolo_barcodes_list = []
        if yolo_session_global: # 只有YOLO加载成功才执行
            print(f"\n--- 步骤2: YOLO 条码锚点检测 ---")
            yolo_barcodes_list, _ = get_yolo_barcode_anchors(original_image.copy())

        # --- 构建和打印矩阵 ---
        final_matrix_to_display, rows, cols, count = [],0,0,0
        if MATRIX_BUILD_STRATEGY == "YOLO_AXIS_ASSISTED":
            if yolo_session_global and yolo_barcodes_list:
                final_matrix_to_display, rows, cols, count = build_matrix_yolo_axis_calibrated(yolo_barcodes_list, potential_obu_list_paddle, LAYOUT_CONFIG, original_image.shape[:2])
                print_matrix_to_console(final_matrix_to_display, f"策略: YOLO轴心标定 - {img_filename_base}")
            else:
                print("YOLO模型未加载或未检测到条码，无法执行YOLO轴心标定策略。")
        else: # 默认或指定PADDLE_ONLY (可以后续添加更多策略分支)
            # 这里可以调用一个纯PaddleOCR的矩阵构建方案，例如V2.9.0中的 build_matrix_with_paddle_only
            print("当前版本主要测试YOLO轴心标定，纯PaddleOCR矩阵构建待后续完善或从旧版引入。")

        # 可视化 (可以根据最终选择的矩阵构建策略来决定画什么)
        # output_viz_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"output_{img_filename_base}_{VERSION}.png")
        # draw_final_matrix_on_image(original_image, final_matrix_to_display, yolo_derived_grid_coords, output_viz_path)
        # (上面的绘图函数需要重新设计)
        # 暂时先用之前的绘图函数画出PaddleOCR的原始和筛选结果
        temp_viz_path = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"temp_paddle_viz_{img_filename_base}_{VERSION}.png")
        if original_image is not None:
             draw_ocr_results_refined(original_image, all_paddle_ocr_data, potential_obu_list_paddle, temp_viz_path)


    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。")
    print(f"-------------------------------------------------")