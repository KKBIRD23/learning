# coding: utf-8
"""
OBU (车载单元) 镭标码识别与矩阵输出脚本
版本: v2.9.2_Math_Calibration_with_YOLO_Assist
功能:
- 核心: 实现基于少数控制点进行数学标定，精确推断OBU网格，并填充识别结果。
- YOLO检测条码作为OBU锚点，辅助选取控制点。
- PaddleOCR识别数字。
- 输出最终的OBU矩阵。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr
import onnxruntime
from itertools import product
import csv
from datetime import datetime
from collections import Counter
from scipy.spatial.distance import cdist # 用于计算距离矩阵，方便匹配

# --- V2.9.2 配置参数 ---
VERSION = "v2.9.2_Math_Calibration_with_YOLO_Assist"
IMAGE_PATHS = [
    r"../../DATA/PIC/1.JPG",
    r"../../DATA/PIC/2.JPG",
    r"../../DATA/PIC/3.JPG",
]
BASE_OUTPUT_DIR = "./output_v2.9_math_calib"
TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP_NOW}_{VERSION}")
# LOG_FILE_PATH = os.path.join(CURRENT_RUN_OUTPUT_DIR, f"矩阵日志_{VERSION}_{TIMESTAMP_NOW}.csv") # 可选
os.makedirs(CURRENT_RUN_OUTPUT_DIR, exist_ok=True)

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
MIN_OBU_SCORE_FOR_CONTROL_POINT = 0.85 # 作为控制点的OBU的最低置信度

# --- YOLOv8 相关配置 ---
YOLO_ONNX_MODEL_PATH_CFG = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx"
YOLO_CONFIDENCE_THRESHOLD_CFG = 0.25
YOLO_IOU_THRESHOLD_CFG = 0.45
YOLO_INPUT_WIDTH_CFG = 640
YOLO_INPUT_HEIGHT_CFG = 640

# --- 矩阵与布局先验配置 ---
LAYOUT_CONFIG = {
    "total_obus": 50,
    "regular_rows": 12,
    "regular_cols": 4,
    "special_row_cols": 2,
    # "special_row_is_last": True # 这个将由程序动态判断或结合控制点推断
}
# 匹配和几何推断的阈值
CONTROL_POINT_MATCH_MAX_DIST = 30 # 控制点YOLO框中心与Paddle数字框中心的最大匹配距离（像素）
PADDLE_OBU_TO_IDEAL_GRID_MAX_DIST_FACTOR = 0.7 # PaddleOCR识别的数字中心与理想格点中心的最大匹配距离因子 (乘以平均OBU估算宽度)


# --- 全局变量 ---
paddle_ocr_engine_global = None
yolo_session_global = None

# --- 函数定义 ---
# initialize_paddleocr, load_yolo_model (与V2.9.1一致)
# preprocess_for_yolo, non_max_suppression_global, postprocess_yolo_detections, get_yolo_barcode_anchors (您已提供并验证)
# get_box_center_and_dims (已验证)
# draw_ocr_results_refined (用于临时可视化PaddleOCR结果)
# print_matrix_to_console (用于打印最终矩阵)
# (为保持框架清晰，这些已验证或您已提供的函数代码暂时省略，请确保它们在您的脚本中)
# ... (Ensure these functions are correctly defined in your script as per previous versions) ...
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
    """使用YOLO模型检测图片中的所有条码，返回条码框列表（锚点）。"""
    if not yolo_session_global:
        print("错误: YOLO会话未初始化。")
        return [], 0.0

    input_tensor, ratio, pad_x, pad_y = preprocess_for_yolo(image, YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG)

    input_name = yolo_session_global.get_inputs()[0].name
    t_start = time.time()
    outputs = yolo_session_global.run(None, {input_name: input_tensor})
    yolo_predict_time = time.time() - t_start
    print(f"  YOLO predict() 耗时 {yolo_predict_time:.3f}s")

    # --- 关键修正：使用关键字参数或严格按顺序传递 ---
    detected_barcode_boxes = postprocess_yolo_detections(
        outputs_onnx=outputs,
        conf_threshold=YOLO_CONFIDENCE_THRESHOLD_CFG,
        iou_threshold=YOLO_IOU_THRESHOLD_CFG,
        original_shape_hw=image.shape[:2],
        model_input_shape_hw=(YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG),
        ratio_preproc=ratio,
        pad_x_preproc=pad_x,
        pad_y_preproc=pad_y
    )
    # 或者严格按顺序（不推荐，易错）:
    # detected_barcode_boxes = postprocess_yolo_detections(
    #     outputs,
    #     YOLO_CONFIDENCE_THRESHOLD_CFG,
    #     YOLO_IOU_THRESHOLD_CFG,
    #     image.shape[:2],
    #     (YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG),
    #     ratio,
    #     pad_x,
    #     pad_y
    # )

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
    print(f"警告: get_box_center_and_dims 接收到无法解析的box格式: {box_xyxy_or_poly}"); return None, None, None, None

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


# --- 核心：数学标定与矩阵构建 ---
def build_matrix_with_math_calibration(yolo_anchors_input, paddle_results_input, layout_config, image_wh):
    """
    核心函数：通过YOLO锚点辅助选择控制点，进行数学标定，推断理想网格，并用PaddleOCR结果填充。
    Args:
        yolo_anchors_input (list): YOLO检测到的条码框列表 [{'cx', 'cy', 'w', 'h', 'box_yolo'}, ...]
        paddle_results_input (list): PaddleOCR筛选后的OBU结果列表 [{"text":..., "score":..., "box":poly, 'cx', 'cy'}, ...]
        layout_config (dict): 包含布局先验的配置
        image_wh (tuple): 原始图片的宽度和高度 (w, h)
    Returns:
        list: 二维矩阵, int: 填充的OBU数量
    """
    print("  正在执行数学标定矩阵构建...")
    if not yolo_anchors_input:
        print("  YOLO未提供锚点，数学标定法无法执行。")
        return [["YOLO无锚点"] * layout_config["regular_cols"]], 0

    # 1. 预处理PaddleOCR结果，添加中心点，并按分数筛选控制点候选
    control_point_candidates = []
    for pr in paddle_results_input:
        if pr['score'] >= MIN_OBU_SCORE_FOR_CONTROL_POINT:
            cx, cy, _, _ = get_box_center_and_dims(pr['box'])
            if cx is not None:
                control_point_candidates.append({**pr, 'cx': cx, 'cy': cy, 'used_for_calib': False})

    if len(control_point_candidates) < 3: # 至少需要3个点才能做初步的几何推断
        print(f"  高置信度OBU数量 ({len(control_point_candidates)}) 不足3个，无法进行有效数学标定。")
        # 此处可以回退到V2.9.1的 build_matrix_yolo_axis_calibrated 的简化逻辑，或者直接返回空/错误
        # 为简化，我们先返回一个提示性的空矩阵
        return [["控制点不足"] * layout_config["regular_cols"]], 0

    # 2. 选取控制点 (这是一个复杂的步骤，以下是非常简化的初步思路)
    #    目标：找到3-4个分布良好（例如，构成四边形）的、且其位置能被YOLO条码框佐证的控制点。
    #    我们先尝试找到与YOLO条码框匹配的高置信度PaddleOCR结果作为控制点。

    # 2a. 将YOLO锚点和PaddleOCR高置信度候选进行匹配
    #     (这是一个简化的匹配，实际可能需要更复杂的对应关系)
    calib_control_points_img_coords = [] # 存储控制点在图像上的 (cx, cy)
    calib_control_points_logical_coords = [] # 存储这些控制点在逻辑网格中的 (row, col) - 这步最难！

    # 排序YOLO锚点，方便后续处理
    yolo_anchors_sorted = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))

    # ==========================================================================
    # TODO: Wang, "选取控制点" 和 "从控制点推断理想网格" 是这里的核心难点。
    # 以下是一个非常非常初步的、占位性的逻辑，几乎肯定无法直接工作，需要我们一起大力填充和优化。
    # 我们需要一个鲁棒的方法来从 yolo_anchors_sorted 和 control_point_candidates 中选出几个点，
    # 并确定它们在50个OBU的逻辑网格中的准确(行,列)索引。
    #
    # 暂时策略：我们先假设能神奇地找到左上角、右上角、左下角（或特殊行的点）的3-4个控制点。
    # 在实际实现中，这可能需要复杂的几何分析、霍夫变换、RANSAC等。
    #
    # 简化占位：我们先强行取YOLO锚点中的几个作为“伪控制点”，并假设它们的逻辑坐标。
    # 这只是为了让后续的“理想网格生成”和“填充”能跑起来，实际效果会很差。
    # ==========================================================================

    if len(yolo_anchors_sorted) >= 4: # 假设我们至少有4个YOLO锚点可以作为粗略的参考
        # 这是一个非常粗糙的假设，实际需要智能选取
        # 假设我们取YOLO锚点中最左上、最右上、最左下、最右下的点（或接近的点）
        # 并强行赋予它们逻辑坐标，这在真实场景中是不准确的。

        # 示例：取YOLO锚点中的第一个、第 regular_cols-1 个（如果存在的话）
        # 第 (regular_rows-1)*regular_cols 个， 最后一个
        # 这完全是示意性的，不能直接用于生产！

        # ---- 这里需要替换为真正的控制点选取和逻辑坐标确定算法 ----
        print("  警告: 当前使用的是占位性的控制点选取逻辑，矩阵结果可能不准确！")
        # 假设我们通过某种方式（例如，YOLO的特殊行识别）找到了以下控制点：
        # control_points_for_transform = [
        #    {'img_cx': yolo_anchor1_cx, 'img_cy': yolo_anchor1_cy, 'logical_row': 0, 'logical_col': 0},
        #    {'img_cx': yolo_anchor2_cx, 'img_cy': yolo_anchor2_cy, 'logical_row': 0, 'logical_col': layout_config['regular_cols']-1},
        #    {'img_cx': yolo_anchor3_cx, 'img_cy': yolo_anchor3_cy, 'logical_row': layout_config['regular_rows']-1, 'logical_col': 0},
        #    # 如果有第四个点会更好，例如右下角
        # ]
        # 如果控制点少于3个，无法计算变换

        # ---- 占位结束 ----

        # 如果我们没有可靠的控制点和变换，就回退到类似V2.9.1的基于YOLO行分组的简单矩阵
        print("  由于精确数学标定逻辑复杂，暂时回退到基于YOLO行分组的近似矩阵构建。")
        # (这里可以调用一个简化版的，类似V2.9.1 build_matrix_yolo_axis_calibrated 的填充逻辑)
        # (为了让代码能跑通，我们先返回一个基于YOLO原始分组的矩阵)
        matrix_rows = len(yolo_rows_grouped) if 'yolo_rows_grouped' in locals() and yolo_rows_grouped else layout_config['regular_rows']
        matrix_cols = Counter([len(r) for r in yolo_rows_grouped]).most_common(1)[0][0] if 'yolo_rows_grouped' in locals() and yolo_rows_grouped and any(yolo_rows_grouped) else layout_config['regular_cols']

        temp_matrix = [["占位"] * matrix_cols for _ in range(matrix_rows)]
        temp_filled_count = 0
        # (此处省略了填充temp_matrix的逻辑，因为它依赖于精确的坑位)
        return temp_matrix, temp_filled_count


    # 3. 如果成功获取控制点并计算了变换矩阵 (例如仿射变换或透视变换)
    #    或者计算了 V_row, V_col, delta_x, delta_y

    # 4. 生成50个理想坑位的精确坐标 (expected_grid_coords)
    #    expected_grid_coords = [] # 列表，每个元素是 (cx, cy)
    #    # ... (基于变换矩阵或向量参数，从逻辑原点开始生成所有50个坑位) ...

    # 5. 将 paddle_results_input 分配到理想坑位
    # final_matrix = [["未识别"] * layout_config["regular_cols"] for _ in range(layout_config["regular_rows"])]
    # # ... (处理特殊行，所以矩阵维度可能不是固定的 regular_rows x regular_cols)
    # matrix_filled_count = 0
    # for ideal_cx, ideal_cy in expected_grid_coords:
    #     # 找到最近的、未被使用的 paddle_result
    #     # ... (匹配逻辑) ...
    #     if matched_paddle_obu:
    #         # 获取对应的逻辑行列号，填充到final_matrix
    #         matrix_filled_count +=1

    # print(f"  数学标定方案: 填充OBU数: {matrix_filled_count}")
    # return final_matrix, matrix_filled_count

    # 由于上述标定逻辑复杂，暂时返回一个提示
    return [["数学标定待实现"] * layout_config["regular_cols"]], 0


# --- 主程序 (V2.9.2) ---
if __name__ == "__main__":
    # ... (与V2.9.1的主程序结构基本一致，主要是调用新的矩阵构建函数)
    overall_start_time = time.time()
    print(f"--- OBU识别与矩阵输出工具 {VERSION} ---"); print(f"输出目录: {os.path.abspath(CURRENT_RUN_OUTPUT_DIR)}")
    if not initialize_paddleocr(): exit()
    if not load_yolo_model(): print("警告: YOLO模型加载失败。") # 不直接退出，因为可能只用PaddleOCR

    for image_path_current in IMAGE_PATHS:
        print(f"\n\n========== 处理图片: {image_path_current} ==========")
        img_filename_base = os.path.splitext(os.path.basename(image_path_current))[0]
        original_image = cv2.imread(image_path_current)
        if original_image is None: print(f"错误: 无法读取图片 {image_path_current}"); continue

        # --- PaddleOCR 识别 ---
        print(f"\n--- 步骤1: PaddleOCR 文本检测与识别 ---")
        # ... (与V2.9.1相同的PaddleOCR处理逻辑，得到 all_paddle_ocr_data 和 potential_obu_list_paddle)
        t_start_paddle = time.time(); ocr_prediction_result = paddle_ocr_engine_global.predict(original_image); paddle_predict_time = time.time() - t_start_paddle
        print(f"PaddleOCR predict() 完成, 耗时: {paddle_predict_time:.3f}s")
        all_paddle_ocr_data = []; potential_obu_list_paddle = []
        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]; dt_polys = ocr_result_object.get('dt_polys'); rec_texts = ocr_result_object.get('rec_texts'); rec_scores = ocr_result_object.get('rec_scores')
            if rec_texts and rec_scores and dt_polys:
                max_items = min(len(rec_texts), len(rec_scores), len(dt_polys))
                if len(rec_texts) != max_items or len(rec_scores) != max_items or len(dt_polys) != max_items : print(f"  警告: PaddleOCR原始输出长度不匹配: texts({len(rec_texts)}), scores({len(rec_scores)}), boxes({len(dt_polys)}). 按最短 {max_items} 处理。")
                for i in range(max_items):
                    item_data = {"text": str(rec_texts[i]), "score": float(rec_scores[i]), "box": dt_polys[i]}
                    all_paddle_ocr_data.append(item_data)
                    text_check = item_data['text'].strip()
                    if text_check.startswith(OBU_CODE_PREFIX_FILTER_CFG) and len(text_check) == OBU_CODE_LENGTH_FILTER_CFG and text_check.isdigit() and item_data['score'] >= PADDLE_OCR_FINE_PARAMS['text_rec_score_thresh']:
                        potential_obu_list_paddle.append(item_data)
        print(f"PaddleOCR 原始有效文本 {len(all_paddle_ocr_data)} 条, 内容筛选后潜在OBU {len(potential_obu_list_paddle)} 个。")

        # --- YOLO 检测 ---
        yolo_anchors = [] # 存放YOLO锚点信息 {'cx', 'cy', 'w', 'h', 'box_yolo'}
        if yolo_session_global:
            print(f"\n--- 步骤2: YOLO 条码锚点检测 ---")
            yolo_barcode_boxes_xyxy, _ = get_yolo_barcode_anchors(original_image.copy())
            for box in yolo_barcode_boxes_xyxy:
                cx, cy, w, h = get_box_center_and_dims(box)
                if cx is not None: yolo_anchors.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box})

        # --- 构建和打印矩阵 (使用数学标定方案) ---
        final_matrix, filled_count = build_matrix_with_math_calibration(yolo_anchors, potential_obu_list_paddle, LAYOUT_CONFIG, original_image.shape[:2])
        print_matrix_to_console(final_matrix, f"数学标定矩阵 - {img_filename_base}")

        # 可视化 (可以画出理想网格点、YOLO锚点、匹配上的PaddleOCR文本)
        # ... (可视化代码待充实)

    overall_end_time = time.time(); total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。"); print(f"-------------------------------------------------")