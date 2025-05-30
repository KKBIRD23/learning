# coding: utf-8
"""
OBU (车载单元) 镭标码识别与矩阵输出脚本
版本: V2.9.7_SmartGrid_Algo_Framework
功能:
- 核心: 为智能网格生成搭建算法框架，包括特殊行识别、几何参数估算、
        理想坑位生成、匹配填充等模块的初步实现或占位。
- YOLO检测条码作为OBU锚点。
- PaddleOCR识别数字。
- 输出最终的OBU矩阵。
- 过程图片输出到 ./Process_Photo/ 子目录 (每次运行前清理)，并使用JPG格式。
"""
import cv2
import numpy as np
import os
import time
import traceback
import paddleocr
import onnxruntime
from collections import Counter
from scipy.spatial.distance import cdist
from datetime import datetime
import shutil # 用于清理文件夹

# --- V2.9.7 配置参数 ---
VERSION = "V2.9.7_SmartGrid_Algo_Framework" # 您可以将此值更新到您实际的文件名中的x
IMAGE_PATHS = [
    r"../../DATA/PIC/1.JPG",
    r"../../DATA/PIC/2.JPG",
    r"../../DATA/PIC/3.JPG",
]
# 输出主目录 (脚本将在此目录下创建 Process_Photo 和 run_... 日志文件夹)
BASE_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在目录
OUTPUT_ROOT_DIR = os.path.join(BASE_SCRIPT_DIR, "output_v2.9_smartgrid_framework")

# 过程图片统一保存到相对于脚本的 ./Process_Photo/ 目录
PROCESS_PHOTO_DIR = os.path.join(BASE_SCRIPT_DIR, "Process_Photo")

TIMESTAMP_NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
# 本次运行的独立日志文件夹 (如果需要更详细的运行日志，不仅仅是CSV)
CURRENT_RUN_LOG_SUBDIR = os.path.join(OUTPUT_ROOT_DIR, f"run_logs_{TIMESTAMP_NOW}_{VERSION}")
# LOG_FILE_PATH = os.path.join(CURRENT_RUN_LOG_SUBDIR, f"矩阵日志_{VERSION}_{TIMESTAMP_NOW}.csv") # 可选

os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
os.makedirs(PROCESS_PHOTO_DIR, exist_ok=True)
# os.makedirs(CURRENT_RUN_LOG_SUBDIR, exist_ok=True) # 如果需要单独的日志运行文件夹

# --- PaddleOCR 初始化相关参数 ---
LANG_CFG = 'en'; USE_TEXTLINE_ORIENTATION_CFG = False; USE_DOC_ORIENTATION_CLASSIFY_CFG = False
USE_DOC_UNWARPING_CFG = False; OCR_VERSION_CFG = None
TEXT_DETECTION_MODEL_DIR_CFG = None; TEXT_RECOGNITION_MODEL_DIR_CFG = None
TEXT_DETECTION_MODEL_NAME_CFG = None; TEXT_RECOGNITION_MODEL_NAME_CFG = None
PADDLE_OCR_FINE_PARAMS = {"text_det_limit_side_len": 960, "text_det_thresh": 0.3,
                          "text_det_box_thresh": 0.6, "text_rec_score_thresh": 0.5}

# --- OBU码筛选规则 ---
OBU_CODE_PREFIX_FILTER_CFG = "5001"; OBU_CODE_LENGTH_FILTER_CFG = 16

# --- YOLOv8 相关配置 ---
YOLO_ONNX_MODEL_PATH_CFG = r"./model/BarCode_Detect/BarCode_Detect_dynamic.onnx" # 相对于脚本的路径
YOLO_CONFIDENCE_THRESHOLD_CFG = 0.25; YOLO_IOU_THRESHOLD_CFG = 0.45
YOLO_INPUT_WIDTH_CFG = 640; YOLO_INPUT_HEIGHT_CFG = 640

# --- 矩阵与布局先验配置 ---
LAYOUT_CONFIG = {"total_obus": 50, "regular_rows_count": 12, "regular_cols_count": 4,
                 "special_row_cols_count": 2, "expected_total_rows": 13 }

# --- 算法相关阈值 ---
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.75
MIN_YOLO_ANCHORS_FOR_LAYOUT = 10
MIN_OBUS_FOR_RELIABLE_ROW = 2
DIGIT_AREA_RELATIVE_Y_OFFSET_FROM_YOLO = -0.7

# --- 全局变量 ---
paddle_ocr_engine_global = None; yolo_session_global = None

# --- 辅助函数 (确保这些函数已从之前版本正确复制并测试) ---
# (initialize_paddleocr, load_yolo_model, preprocess_for_yolo, non_max_suppression_global,
#  postprocess_yolo_detections, get_yolo_barcode_anchors, get_box_center_and_dims,
#  draw_ocr_results_refined, print_matrix_to_console)
# ... (此处粘贴V2.9.2/V2.9.6中已验证的这些辅助函数代码) ...
def initialize_paddleocr():
    global paddle_ocr_engine_global
    init_params = {'lang': LANG_CFG, 'use_textline_orientation': USE_TEXTLINE_ORIENTATION_CFG, 'use_doc_orientation_classify': USE_DOC_ORIENTATION_CLASSIFY_CFG, 'use_doc_unwarping': USE_DOC_UNWARPING_CFG, 'ocr_version': OCR_VERSION_CFG, 'text_detection_model_dir': TEXT_DETECTION_MODEL_DIR_CFG, 'text_recognition_model_dir': TEXT_RECOGNITION_MODEL_DIR_CFG,'text_detection_model_name': TEXT_DETECTION_MODEL_NAME_CFG, 'text_recognition_model_name': TEXT_RECOGNITION_MODEL_NAME_CFG, **PADDLE_OCR_FINE_PARAMS }
    ocr_params_final_filtered = {k: v for k, v in init_params.items() if v is not None}
    print(f"\n正在使用以下参数初始化PaddleOCR: {ocr_params_final_filtered}")
    try: paddle_ocr_engine_global = paddleocr.PaddleOCR(**ocr_params_final_filtered); print("PaddleOCR引擎初始化成功。"); return True
    except Exception as e: print(f"PaddleOCR引擎初始化失败: {e}"); paddle_ocr_engine_global = None; return False

def load_yolo_model():
    global yolo_session_global
    yolo_model_path_abs = os.path.join(BASE_SCRIPT_DIR, YOLO_ONNX_MODEL_PATH_CFG)
    if not os.path.exists(yolo_model_path_abs): print(f"错误: YOLO ONNX模型未找到: {yolo_model_path_abs}"); return False
    try: print(f"正在加载YOLO模型: {yolo_model_path_abs}"); yolo_session_global = onnxruntime.InferenceSession(yolo_model_path_abs, providers=['CPUExecutionProvider']); print("YOLO模型加载成功。"); return True
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
    if not yolo_session_global: print("错误: YOLO会话未初始化。"); return [], 0.0
    input_tensor, ratio, pad_x, pad_y = preprocess_for_yolo(image, YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG)
    input_name = yolo_session_global.get_inputs()[0].name
    t_start = time.time(); outputs = yolo_session_global.run(None, {input_name: input_tensor}); yolo_predict_time = time.time() - t_start
    print(f"  YOLO predict() 耗时 {yolo_predict_time:.3f}s")
    detected_barcode_boxes = postprocess_yolo_detections(outputs_onnx=outputs, conf_threshold=YOLO_CONFIDENCE_THRESHOLD_CFG, iou_threshold=YOLO_IOU_THRESHOLD_CFG, original_shape_hw=image.shape[:2], model_input_shape_hw=(YOLO_INPUT_HEIGHT_CFG, YOLO_INPUT_WIDTH_CFG), ratio_preproc=ratio, pad_x_preproc=pad_x, pad_y_preproc=pad_y)
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
    return None, None, None, None

def draw_ocr_results_refined(image, all_ocr_data, potential_obu_data, output_path="output_ocr_visualization.jpg"):
    img_out = image.copy();_c = cv2
    if img_out is None: print(f"错误: 用于绘制的输入图像为None。无法保存到 {output_path}"); return
    if not all_ocr_data and not potential_obu_data :
        print(f"没有OCR数据可以绘制到 {output_path}.")
        try: _c.imwrite(output_path, img_out, [_c.IMWRITE_JPEG_QUALITY, 85]); print(f"无OCR数据, 底图已保存到: {output_path}")
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
    try: _c.imwrite(output_path, img_out, [_c.IMWRITE_JPEG_QUALITY, 85]); print(f"OCR可视化结果已保存到: {output_path}")
    except Exception as e: print(f"保存可视化图片失败 {output_path}: {e}")

def print_matrix_to_console(matrix, strategy_name=""):
    if not matrix or (isinstance(matrix, list) and not matrix[0]): print(f"  {strategy_name} 生成的矩阵为空，无法打印。"); return
    print(f"\n--- {strategy_name} OBU识别矩阵 ({len(matrix)}行 x {len(matrix[0]) if matrix and matrix[0] else 0}列) ---")
    for row in matrix:
        row_display = ["  红  " if item == "未识别" else (f" {item[-4:]} " if isinstance(item, str) and item.startswith("5001") and len(item)>4 else f" {str(item)[:4]:^4} ") for item in row]
        print(" ".join(row_display))
    print("---------------------------------------------")

def clear_directory(directory_path):
    """清空指定目录下的所有文件和子文件夹"""
    if not os.path.isdir(directory_path): return
    print(f"正在清空目录: {directory_path}")
    for item_name in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) # 递归删除子文件夹
        except Exception as e:
            print(f'无法删除 {item_path}. 原因: {e}')

# --- 核心算法模块 (V2.9.7 初步框架) ---
def cluster_yolo_anchors_into_rows(yolo_anchors_with_info, y_threshold_factor):
    """将YOLO锚点按Y坐标聚类成行，并进行行内X排序。"""
    if not yolo_anchors_with_info: return []
    # 1. 按Y坐标主序，X坐标次序排序 (确保即使Y相近，X小的也在前面)
    yolo_anchors_sorted = sorted(yolo_anchors_with_info, key=lambda a: (a['cy'], a['cx']))

    # 2. 行分组
    yolo_rows_grouped = []
    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors_sorted if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted) else 30
    y_coord_diff_threshold = avg_h_yolo * y_threshold_factor

    current_row = []
    if yolo_anchors_sorted: # 确保列表不为空
        current_row.append(yolo_anchors_sorted[0])
        for i in range(1, len(yolo_anchors_sorted)):
            # 使用当前行第一个元素的Y坐标作为基准，或行内平均Y坐标
            # 这里简化为与前一个元素的Y坐标比较
            if abs(yolo_anchors_sorted[i]['cy'] - current_row[-1]['cy']) < y_coord_diff_threshold:
                current_row.append(yolo_anchors_sorted[i])
            else:
                yolo_rows_grouped.append(sorted(current_row, key=lambda a: a['cx'])) # 行内按X排序
                current_row = [yolo_anchors_sorted[i]]
        if current_row: # 添加最后一行
            yolo_rows_grouped.append(sorted(current_row, key=lambda a: a['cx']))

    print(f"  YOLO锚点精确行分组为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")
    return yolo_rows_grouped

def identify_special_rows_and_orientation(yolo_rows_grouped, layout_config):
    """
    智能识别特殊行（2个OBU）及其位置（顶部或底部）。
    返回: (special_row_top_idx, special_row_bottom_idx, inferred_regular_cols)
           idx为None表示未在该位置找到明确的特殊行。
    """
    print("  正在识别特殊行...")
    special_row_top_idx = None
    special_row_bottom_idx = None
    num_detected_yolo_rows = len(yolo_rows_grouped)

    # 从YOLO行中推断常规列数 (取众数，但排除掉数量为2的行进行统计，如果它们可能是特殊行)
    regular_col_counts_options = [len(r) for r in yolo_rows_grouped if len(r) != layout_config["special_row_cols_count"]]
    if not regular_col_counts_options: # 如果所有行都是2个或者没有其他数量的行
        regular_col_counts_options = [len(r) for r in yolo_rows_grouped] # 就用所有行的数量

    inferred_regular_cols = layout_config["regular_cols_count"] # 默认
    if regular_col_counts_options:
        mode_res = Counter(regular_col_counts_options).most_common(1)
        if mode_res and mode_res[0][0] > 0 : inferred_regular_cols = mode_res[0][0]
    print(f"  从YOLO行推断出的常规列数 (暂定): {inferred_regular_cols}")

    if num_detected_yolo_rows == 0: return None, None, inferred_regular_cols

    # 检查第一行是否为特殊行
    if len(yolo_rows_grouped[0]) == layout_config["special_row_cols_count"]:
        # 如果只有一行且是2，也认为是特殊行
        if num_detected_yolo_rows == 1:
            special_row_top_idx = 0
        # 如果有多行，检查下一行是否像常规行
        elif num_detected_yolo_rows > 1 and abs(len(yolo_rows_grouped[1]) - inferred_regular_cols) <= 1: # 允许常规行YOLO漏检1个
            special_row_top_idx = 0
            print("  初步判断: 特殊行在顶部。")

    # 检查最后一行是否为特殊行 (如果总行数大于1且顶部不是特殊行)
    if num_detected_yolo_rows > 1 and special_row_top_idx is None: # 避免重复判断或冲突
        if len(yolo_rows_grouped[-1]) == layout_config["special_row_cols_count"] and \
           abs(len(yolo_rows_grouped[-2]) - inferred_regular_cols) <= 1:
            special_row_bottom_idx = num_detected_yolo_rows - 1
            print("  初步判断: 特殊行在底部。")

    # 如果顶部和底部都像特殊行（例如，总共只有两行，都是2个OBU），需要更复杂的逻辑，暂时先不处理这种情况
    if special_row_top_idx is not None and special_row_bottom_idx is not None and special_row_top_idx == special_row_bottom_idx:
        # 这种情况通常是总行数很少，例如只有一行是2个OBU
        if num_detected_yolo_rows == 1 : special_row_bottom_idx = None # 只有一行，不算底部特殊
        else: # 如果有多行，但首尾都是2，这不符合12*4+2的结构，可能需要人工检查或调整参数
             print("  警告: 首尾行都像特殊行，布局可能与预期不符。优先判断顶部。")
             special_row_bottom_idx = None


    if special_row_top_idx is None and special_row_bottom_idx is None:
        print("  警告: 未能明确识别出顶部或底部的特殊行。将基于YOLO行数和推断列数构建网格。")
        # 这种情况下，我们可能需要一个更鲁棒的全局网格拟合，或者默认特殊行在底部（更常见）
        # 暂时先不改变 special_row_top/bottom_idx 的 None 状态

    return special_row_top_idx, special_row_bottom_idx, inferred_regular_cols

def estimate_grid_geometric_parameters(yolo_rows_grouped, special_row_top_idx, special_row_bottom_idx, inferred_regular_cols, layout_config):
    """
    估算网格的几何参数（行高、列宽、起始点、透视参数等）。
    这是核心算法，当前为高度简化版。
    返回: 一个包含几何参数的字典, 例如:
           {'avg_row_h': float, 'avg_col_w': float,
            'row_y_coords': list_of_floats, # 每逻辑行的中心Y
            'col_x_coords_at_row_y': func, # 一个函数，输入y，返回该y高度下各逻辑列的中心X列表
            'grid_start_xy': (x,y) } # 逻辑 (0,0) 的图像坐标
    """
    print("  正在估算网格几何参数...")
    if not yolo_rows_grouped: return None

    # 1. 估算平均OBU尺寸
    all_widths = [a['w'] for row in yolo_rows_grouped for a in row if a.get('w',0)>0]
    all_heights = [a['h'] for row in yolo_rows_grouped for a in row if a.get('h',0)>0]
    avg_w = np.mean(all_widths) if all_widths else 100
    avg_h = np.mean(all_heights) if all_heights else 40
    print(f"  估算YOLO锚点平均尺寸: W={avg_w:.1f}, H={avg_h:.1f}")

    # 2. 估算每逻辑行的中心Y坐标
    #    理想情况：有13个逻辑行。我们尝试将YOLO检测到的行映射到这13个逻辑行上。
    #    这是一个简化的映射，假设YOLO检测到的行顺序与逻辑行顺序一致。
    logical_row_y_coords = [0.0] * layout_config["expected_total_rows"]

    # 使用YOLO行的平均Y作为该逻辑行的Y (如果YOLO行数与预期接近)
    # TODO: 需要更鲁棒的映射，处理YOLO漏检整行的情况
    if len(yolo_rows_grouped) >= layout_config["expected_total_rows"] - 2 and len(yolo_rows_grouped) <= layout_config["expected_total_rows"] + 2 : # 允许一些误差
        for i in range(min(len(yolo_rows_grouped), layout_config["expected_total_rows"])):
            logical_row_y_coords[i] = np.mean([a['cy'] for a in yolo_rows_grouped[i]])
        # 如果YOLO行数不足，用平均行高插值/外推
        if len(yolo_rows_grouped) < layout_config["expected_total_rows"] and len(yolo_rows_grouped) > 0:
            last_known_y = logical_row_y_coords[len(yolo_rows_grouped)-1]
            for i in range(len(yolo_rows_grouped), layout_config["expected_total_rows"]):
                logical_row_y_coords[i] = last_known_y + (avg_h + 10) * (i - (len(yolo_rows_grouped) - 1)) # 简单线性外推
    else: # YOLO行数与预期差异太大，使用基于首尾的等分 (非常粗略)
        print("  警告: YOLO行数与预期差异较大，行Y坐标估算可能不准。")
        if yolo_anchors_sorted:
            min_y_overall = yolo_anchors_sorted[0]['cy']
            max_y_overall = yolo_anchors_sorted[-1]['cy']
            if layout_config["expected_total_rows"] > 1:
                step_y = (max_y_overall - min_y_overall) / (layout_config["expected_total_rows"] -1)
                for i in range(layout_config["expected_total_rows"]): logical_row_y_coords[i] = min_y_overall + i * step_y
            else: logical_row_y_coords[0] = (min_y_overall + max_y_overall) / 2
        else: # 实在没办法了
            for i in range(layout_config["expected_total_rows"]): logical_row_y_coords[i] = 100 + i * (avg_h + 10)


    # 3. 估算每逻辑行中，各逻辑列的中心X坐标 (处理透视的关键)
    #    这是一个非常复杂的任务，需要拟合每列的X坐标随Y坐标变化的趋势。
    #    初步简化：对于每一目标逻辑行，我们尝试从yolo_rows_grouped中找到Y坐标最接近的实际YOLO行，
    #              然后基于该实际YOLO行的X分布（第一个元素的X，和行内平均X间距）来推算该逻辑行的X坐标。
    #    返回一个函数 col_x_func(logical_row_idx, logical_col_idx) -> cx

    # 存储每条YOLO物理行的X坐标信息：{'min_x', 'max_x', 'avg_x_spacing', 'anchors_in_row'}
    yolo_row_x_info = []
    for r_group in yolo_rows_grouped:
        if not r_group: continue
        xs = [a['cx'] for a in r_group]
        min_x_in_row = min(xs)
        max_x_in_row = max(xs)
        avg_spacing_in_row = 0
        if len(xs) > 1:
            spacings = [xs[j+1] - xs[j] for j in range(len(xs)-1)]
            avg_spacing_in_row = np.mean(spacings) if spacings else avg_w * 1.1 # 默认间距
        yolo_row_x_info.append({
            'cy_of_row': np.mean([a['cy'] for a in r_group]),
            'min_x': min_x_in_row,
            'max_x': max_x_in_row,
            'avg_x_spacing': avg_spacing_in_row,
            'num_anchors': len(r_group),
            'anchors': r_group # 保存原始锚点
        })

    def get_ideal_col_x_for_row(target_logical_row_idx, target_logical_col_idx,
                                num_cols_for_this_target_row, is_target_row_special):
        # 找到Y坐标最接近 target_logical_row_idx 对应 ideal_y 的那条 yolo_row_x_info
        target_y = logical_row_y_coords[target_logical_row_idx]
        if not yolo_row_x_info: # 没有YOLO行信息，只能用全局平均
            overall_center_x = image_wh[1] / 2 # 图像中心X
            start_x_for_row = overall_center_x - (num_cols_for_this_target_row / 2.0 - 0.5) * (avg_w * 1.1)
            return start_x_for_row + target_logical_col_idx * (avg_w*1.1)

        closest_yolo_row_info = min(yolo_row_x_info, key=lambda r_info: abs(r_info['cy_of_row'] - target_y))

        # 使用closest_yolo_row_info的X分布来推算
        # 如果目标行是特殊行（2列），X坐标需要相对于常规行（4列）的中心进行对齐
        effective_start_x = closest_yolo_row_info['min_x']
        effective_col_spacing = closest_yolo_row_info['avg_x_spacing'] if closest_yolo_row_info['avg_x_spacing'] > 0 else avg_w * 1.1

        # 如果是特殊行，它的X坐标应该是相对于常规4列的中间位置
        if is_target_row_special:
            # 假设常规4列的中心X与closest_yolo_row_info的中心X近似
            center_x_of_regular_row_approx = (closest_yolo_row_info['min_x'] + closest_yolo_row_info['max_x']) / 2 \
                                             if closest_yolo_row_info['num_anchors'] >= inferred_regular_cols -1 \
                                             else image_wh[1] / 2 # Fallback to image center

            # 特殊行的两个OBU应该对称分布在这个中心附近
            if num_cols_for_this_target_row == 2:
                # 这里的间距可以用常规列的平均间距，或者特殊行自身的（如果YOLO检测到了2个）
                special_row_col_spacing = avg_w * 1.1 # 默认
                if closest_yolo_row_info['num_anchors'] == 2:
                    special_row_col_spacing = abs(closest_yolo_row_info['anchors'][1]['cx'] - closest_yolo_row_info['anchors'][0]['cx'])

                if target_logical_col_idx == 0:
                    return center_x_of_regular_row_approx - special_row_col_spacing / 2.0
                elif target_logical_col_idx == 1:
                    return center_x_of_regular_row_approx + special_row_col_spacing / 2.0

        # 对于常规行
        return effective_start_x + target_logical_col_idx * effective_col_spacing

    print("  网格几何参数估算完成（初步）。")
    return {
        'avg_row_h': avg_h, 'avg_col_w': avg_w,
        'logical_row_y_coords': logical_row_y_coords,
        'get_ideal_col_x_for_row_func': get_ideal_col_x_for_row,
        'inferred_regular_cols_from_yolo': inferred_regular_cols, # 把这个也传出去
        'special_row_is_at_top_from_yolo': special_row_is_at_top
    }


def generate_all_ideal_grid_slots(grid_geometric_params, layout_config):
    """根据估算出的几何参数和布局先验，生成所有50个理想坑位的坐标。"""
    print("  正在生成所有理想坑位坐标...")
    if not grid_geometric_params: return []

    ideal_slots = []
    logical_row_ys = grid_geometric_params['logical_row_y_coords']
    col_x_func = grid_geometric_params['get_ideal_col_x_for_row_func']
    special_is_top = grid_geometric_params['special_row_is_at_top_from_yolo']

    current_obu_count = 0
    for r_logic in range(layout_config["expected_total_rows"]):
        cols_this_logic_row = layout_config["regular_cols_count"]
        is_special = False
        if (special_is_top is True and r_logic == 0) or \
           (special_is_top is False and r_logic == layout_config["expected_total_rows"] - 1):
            cols_this_logic_row = layout_config["special_row_cols_count"]
            is_special = True

        for c_logic in range(cols_this_logic_row):
            if current_obu_count >= layout_config["total_obus"]: break

            ideal_cx = col_x_func(r_logic, c_logic, cols_this_logic_row, is_special)
            ideal_cy = logical_row_ys[r_logic]

            ideal_slots.append({
                'logical_row': r_logic, 'logical_col': c_logic,
                'cx': int(ideal_cx), 'cy': int(ideal_cy),
                'w': int(grid_geometric_params['avg_col_w']), # 使用平均尺寸
                'h': int(grid_geometric_params['avg_row_h'])
            })
            current_obu_count += 1
        if current_obu_count >= layout_config["total_obus"]: break

    print(f"  成功生成 {len(ideal_slots)} 个理想坑位坐标。")
    return ideal_slots


def match_paddle_to_ideal_slots_and_fill_matrix(ideal_grid_slots, paddle_results_input, layout_config):
    """将PaddleOCR识别结果填充到理想坑位，构建最终矩阵。"""
    print("  正在将PaddleOCR结果匹配并填充到理想网格...")
    # 矩阵维度由预期的总行数和常规列数决定
    final_matrix = [["未识别"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])]
    matrix_filled_count = 0

    # 为paddle_results_input计算中心点 (如果它们还没有的话)
    paddle_obus_with_center = []
    if paddle_results_input:
        for pr in paddle_results_input:
            cx, cy, w, h = get_box_center_and_dims(pr['box'])
            if cx is not None:
                paddle_obus_with_center.append({**pr, 'cx': cx, 'cy': cy, 'w':w, 'h':h, 'used': False})

    if not ideal_grid_slots or not paddle_obus_with_center:
        print("  无理想坑位或无PaddleOCR结果可用于匹配。")
        return final_matrix, 0

    ideal_coords_np = np.array([[slot['cx'], slot['cy']] for slot in ideal_grid_slots])
    paddle_coords_np = np.array([[p['cx'], p['cy']] for p in paddle_obus_with_center])

    if paddle_coords_np.size == 0: print("  警告: 没有有效的PaddleOCR中心点用于匹配。"); return final_matrix, 0

    dist_matrix = cdist(ideal_coords_np, paddle_coords_np)

    for i_slot, slot_info in enumerate(ideal_grid_slots):
        log_r, log_c = slot_info['logical_row'], slot_info['logical_col']
        # 确保不会超出final_matrix的边界 (虽然理论上ideal_slots的行列应该与final_matrix匹配)
        if log_r >= len(final_matrix) or log_c >= len(final_matrix[0]): continue

        best_paddle_idx = -1; min_dist_to_slot = float('inf')
        # 匹配阈值基于理想坑位的估算宽度
        max_dist_thresh = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * slot_info.get('w', 100)

        for j_paddle, p_obu in enumerate(paddle_obus_with_center):
            if p_obu['used']: continue
            if i_slot < dist_matrix.shape[0] and j_paddle < dist_matrix.shape[1]:
                current_dist = dist_matrix[i_slot, j_paddle]
                if current_dist < max_dist_thresh and current_dist < min_dist_to_slot:
                    min_dist_to_slot = current_dist; best_paddle_idx = j_paddle

        if best_paddle_idx != -1:
            final_matrix[log_r][log_c] = paddle_obus_with_center[best_paddle_idx]['text']
            paddle_obus_with_center[best_paddle_idx]['used'] = True
            matrix_filled_count += 1

    print(f"  矩阵填充完成，共填充 {matrix_filled_count} 个OBU。")
    return final_matrix, matrix_filled_count

# --- 核心：智能网格生成与填充 (V2.9.7 初步框架) ---
def build_matrix_smart_grid(yolo_anchors_input, paddle_results_input, layout_config, image_wh):
    """
    核心函数：通过YOLO锚点、特殊行识别和布局先验，精确推断理想网格，并用PaddleOCR结果填充。
    Args:
        yolo_anchors_input (list): 预处理后的YOLO锚点列表 [{'cx', 'cy', 'w', 'h', 'box_yolo'}, ...]
        paddle_results_input (list): PaddleOCR筛选后的OBU结果列表 [{"text":..., "score":..., "box":poly, 'cx', 'cy', 'w', 'h'}, ...]
                                      (注意：此函数期望 paddle_results_input 中的每个元素也已计算了cx,cy,w,h)
        layout_config (dict): 包含布局先验的配置
        image_wh (tuple): 原始图片的宽度和高度 (w, h)
    Returns:
        list: 二维矩阵, int: 填充的OBU数量
    """
    print("  正在执行智能网格矩阵构建...")
    if not yolo_anchors_input or len(yolo_anchors_input) < MIN_YOLO_ANCHORS_FOR_LAYOUT:
        print(f"  YOLO锚点数量 ({len(yolo_anchors_input)}) 不足 ({MIN_YOLO_ANCHORS_FOR_LAYOUT}个)，无法进行可靠布局推断。")
        return [["YOLO锚点不足"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])], 0

    # 1. 对YOLO锚点按Y坐标进行初步行分组 (此步骤已在V2.9.2的main函数中完成，传入的yolo_anchors_input已经是排序和包含cx,cy的了)
    #    但我们在这里重新进行一次更严格的行分组，并尝试识别特殊行
    yolo_anchors_sorted = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))

    yolo_rows_grouped = []
    avg_h_yolo = np.mean([a['h'] for a in yolo_anchors_sorted if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted) else 30
    y_threshold = avg_h_yolo * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    if not yolo_anchors_sorted: return [["无有效YOLO锚点"] * layout_config["regular_cols_count"]], 0
    current_row_for_grouping = [yolo_anchors_sorted[0]]
    for i in range(1, len(yolo_anchors_sorted)):
        if abs(yolo_anchors_sorted[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold:
            current_row_for_grouping.append(yolo_anchors_sorted[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
            current_row_for_grouping = [yolo_anchors_sorted[i]]
    if current_row_for_grouping: yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
    print(f"  YOLO锚点精确行分组为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")

    # 2. 智能识别特殊行，并估算布局参数
    special_row_is_at_top = None
    num_detected_yolo_rows = len(yolo_rows_grouped)
    inferred_regular_cols = layout_config["regular_cols_count"]
    if num_detected_yolo_rows > 0:
        if len(yolo_rows_grouped[0]) == layout_config["special_row_cols_count"] and num_detected_yolo_rows > 1 and len(yolo_rows_grouped[1]) == layout_config["regular_cols_count"]:
            special_row_is_at_top = True; print("  初步判断: 特殊行在顶部。")
        elif len(yolo_rows_grouped[-1]) == layout_config["special_row_cols_count"] and num_detected_yolo_rows > 1 and len(yolo_rows_grouped[-2]) == layout_config["regular_cols_count"]:
            special_row_is_at_top = False; print("  初步判断: 特殊行在底部。")
        else:
            print("  警告: 未能明确判断特殊行位置。将尝试基于多数行推断常规列数。")
            col_counts = [len(r) for r in yolo_rows_grouped if len(r) > 1];
            if col_counts: mode_res = Counter(col_counts).most_common(1);
            if mode_res and mode_res[0][0] > 0: inferred_regular_cols = mode_res[0][0]
            if special_row_is_at_top is None : special_row_is_at_top = False
    else: print("  YOLO行分组为空，无法推断布局。"); return [["YOLO行分组失败"] * layout_config["regular_cols_count"]], 0
    print(f"  推断常规列数: {inferred_regular_cols}, 特殊行是否在顶部: {special_row_is_at_top}")

    # 3b. 生成50个理想坑位坐标 (仍是高度简化的占位逻辑)
    ideal_grid_slots = []
    all_yolo_cx = [a['cx'] for a in yolo_anchors_sorted]; all_yolo_cy = [a['cy'] for a in yolo_anchors_sorted]
    all_yolo_w = [a['w'] for a in yolo_anchors_sorted if a['w'] > 0]; all_yolo_h = [a['h'] for a in yolo_anchors_sorted if a['h'] > 0]
    avg_obu_w_yolo = np.mean(all_yolo_w) if all_yolo_w else 100
    avg_obu_h_yolo = np.mean(all_yolo_h) if all_yolo_h else 40
    ideal_row_y_coords = [np.mean([a['cy'] for a in r_group]) for r_group in yolo_rows_grouped]
    reference_row_for_x = yolo_rows_grouped[0] if yolo_rows_grouped else [] # 添加检查
    if len(yolo_rows_grouped) > 1: reference_row_for_x = max(yolo_rows_grouped, key=len)
    start_x = reference_row_for_x[0]['cx'] - (inferred_regular_cols / 2.0 - 0.5) * avg_obu_w_yolo if reference_row_for_x else np.min(all_yolo_cx) if all_yolo_cx else 100
    current_obu_count = 0
    for r_idx in range(layout_config["expected_total_rows"]):
        cols_for_this_logical_row = layout_config["regular_cols_count"]; is_this_row_special = False
        if (special_row_is_at_top is True and r_idx == 0) or \
           (special_row_is_at_top is False and r_idx == layout_config["expected_total_rows"] - 1): # 确保 special_row_is_at_top 不是 None
            cols_for_this_logical_row = layout_config["special_row_cols_count"]; is_this_row_special = True
        current_y = ideal_row_y_coords[r_idx] if r_idx < len(ideal_row_y_coords) else ideal_row_y_coords[-1] + avg_obu_h_yolo * (r_idx - (len(ideal_row_y_coords) -1)) if ideal_row_y_coords else 100 + r_idx * (avg_obu_h_yolo + 10)
        x_offset_for_centering = (inferred_regular_cols - cols_for_this_logical_row) * avg_obu_w_yolo / 2.0 if is_this_row_special and cols_for_this_logical_row < inferred_regular_cols else 0
        for c_idx in range(cols_for_this_logical_row):
            if current_obu_count >= layout_config["total_obus"]: break
            current_x = start_x + x_offset_for_centering + c_idx * (avg_obu_w_yolo * 1.1)
            ideal_grid_slots.append({'logical_row': r_idx, 'logical_col': c_idx, 'cx': int(current_x), 'cy': int(current_y), 'w': int(avg_obu_w_yolo), 'h': int(avg_obu_h_yolo)})
            current_obu_count += 1
        if current_obu_count >= layout_config["total_obus"]: break
    if not ideal_grid_slots: print("  未能生成理想坑位坐标。"); return [["无理想坑位"]*layout_config["regular_cols_count"]],0
    print(f"  已生成 {len(ideal_grid_slots)} 个理想坑位坐标。")

    # 4. 将PaddleOCR识别结果填充到理想坑位
    final_matrix = [["未识别"] * layout_config["regular_cols_count"] for _ in range(layout_config["expected_total_rows"])]
    matrix_filled_count = 0
    paddle_results_with_center_and_dims = []
    if paddle_results_input:
        for pr in paddle_results_input:
            cx, cy, pw, ph = get_box_center_and_dims(pr['box'])
            if cx is not None: paddle_results_with_center_and_dims.append({**pr, 'cx': cx, 'cy': cy, 'w':pw, 'h':ph, 'used': False})
    if ideal_grid_slots and paddle_results_with_center_and_dims:
        ideal_coords_np = np.array([[slot['cx'], slot['cy']] for slot in ideal_grid_slots])
        paddle_coords_np = np.array([[p['cx'], p['cy']] for p in paddle_results_with_center_and_dims])
        if paddle_coords_np.size == 0: print("  警告: 没有有效的PaddleOCR中心点用于匹配。")
        else:
            dist_matrix = cdist(ideal_coords_np, paddle_coords_np)
            for i_slot, slot in enumerate(ideal_grid_slots):
                log_r, log_c = slot['logical_row'], slot['logical_col']
                if log_r >= len(final_matrix) or log_c >= len(final_matrix[0]): continue
                best_paddle_idx = -1; min_dist_to_slot = float('inf')
                avg_yolo_anchor_width_for_thresh = np.mean([a['w'] for a in yolo_anchors_input if a.get('w', 0) > 0]) if yolo_anchors_input and any(a.get('w',0)>0 for a in yolo_anchors_input) else 30
                max_dist_thresh = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * slot.get('w', avg_yolo_anchor_width_for_thresh)
                for j_paddle, p_obu in enumerate(paddle_results_with_center_and_dims):
                    if p_obu['used']: continue
                    if i_slot < dist_matrix.shape[0] and j_paddle < dist_matrix.shape[1]:
                        current_dist = dist_matrix[i_slot, j_paddle]
                        if current_dist < max_dist_thresh and current_dist < min_dist_to_slot:
                            min_dist_to_slot = current_dist; best_paddle_idx = j_paddle
                if best_paddle_idx != -1:
                    final_matrix[log_r][log_c] = paddle_results_with_center_and_dims[best_paddle_idx]['text']
                    paddle_results_with_center_and_dims[best_paddle_idx]['used'] = True; matrix_filled_count += 1

    print(f"  智能网格方案: 构建矩阵 {len(final_matrix)}x{len(final_matrix[0]) if final_matrix and final_matrix[0] else 0}, 填充OBU数: {matrix_filled_count}")
    return final_matrix, matrix_filled_count # 修正：返回两个值

# --- 主程序 (V2.9.7 调用 build_matrix_smart_grid) ---
if __name__ == "__main__":
    overall_start_time = time.time()
    print(f"--- OBU识别与矩阵输出工具 {VERSION} ---")
    print(f"输出目录: {os.path.abspath(OUTPUT_ROOT_DIR)}") # 使用修正后的目录变量
    print(f"本次运行特定日志子目录: {os.path.abspath(CURRENT_RUN_LOG_SUBDIR)}")
    print(f"过程图片保存目录: {os.path.abspath(PROCESS_PHOTO_DIR)}")

    clear_directory(PROCESS_PHOTO_DIR) # 每次运行时清理过程图片目录

    if not initialize_paddleocr():
        exit()
    if not load_yolo_model():
        print("警告: YOLO模型加载失败，部分功能可能受限。")
        # 根据您的策略，如果YOLO是必需的，这里也可以 exit()

    for image_path_current in IMAGE_PATHS:
        print(f"\n\n========== 处理图片: {image_path_current} ==========")
        img_filename_base = os.path.splitext(os.path.basename(image_path_current))[0]
        original_image = cv2.imread(image_path_current)
        if original_image is None:
            print(f"错误: 无法读取图片 {image_path_current}");
            continue

        # --- 步骤1: PaddleOCR 文本检测与识别 ---
        print(f"\n--- 步骤1: PaddleOCR 文本检测与识别 ---")
        t_start_paddle = time.time()
        ocr_prediction_result = paddle_ocr_engine_global.predict(original_image)
        paddle_predict_time = time.time() - t_start_paddle
        print(f"PaddleOCR predict() 完成, 耗时: {paddle_predict_time:.3f}s")

        all_paddle_ocr_data = []
        potential_obu_list_paddle = [] # 这是传给 build_matrix_smart_grid 的 paddle_results_input

        if ocr_prediction_result and ocr_prediction_result[0] is not None:
            ocr_result_object = ocr_prediction_result[0]
            dt_polys = ocr_result_object.get('dt_polys')
            rec_texts = ocr_result_object.get('rec_texts')
            rec_scores = ocr_result_object.get('rec_scores')

            if rec_texts and rec_scores and dt_polys:
                max_items = 0
                # 再次确保所有列表都存在且有内容才计算min_len
                if rec_texts and rec_scores and dt_polys:
                    max_items = min(len(rec_texts), len(rec_scores), len(dt_polys))
                    # 打印警告的条件也应基于此
                    if not (len(rec_texts) == len(rec_scores) == len(dt_polys)):
                         print(f"  警告: PaddleOCR原始输出长度不匹配: texts({len(rec_texts)}), scores({len(rec_scores)}), boxes({len(dt_polys)}). 将按最短有效长度 {max_items} 处理。")

                for i in range(max_items): # 使用确保存储的长度
                    item_data = {"text": str(rec_texts[i]), "score": float(rec_scores[i]), "box": dt_polys[i]}
                    all_paddle_ocr_data.append(item_data)
                    text_check = item_data['text'].strip()
                    # 使用 PADDLE_OCR_FINE_PARAMS 中的识别阈值进行筛选
                    if text_check.startswith(OBU_CODE_PREFIX_FILTER_CFG) and \
                       len(text_check) == OBU_CODE_LENGTH_FILTER_CFG and \
                       text_check.isdigit() and \
                       item_data['score'] >= PADDLE_OCR_FINE_PARAMS['text_rec_score_thresh']:
                        potential_obu_list_paddle.append(item_data)
        print(f"PaddleOCR 原始有效文本 {len(all_paddle_ocr_data)} 条, 内容筛选后潜在OBU {len(potential_obu_list_paddle)} 个。")

        # --- 步骤2: YOLO 条码锚点检测 ---
        yolo_anchors_for_matrix_build = [] # 存储YOLO锚点及其cx,cy,w,h
        if yolo_session_global:
            print(f"\n--- 步骤2: YOLO 条码锚点检测 ---")
            yolo_barcode_boxes_xyxy, _ = get_yolo_barcode_anchors(original_image.copy()) # 传入副本以防修改
            for box in yolo_barcode_boxes_xyxy: # box is [x1,y1,x2,y2]
                cx, cy, w, h = get_box_center_and_dims(box)
                if cx is not None:
                    yolo_anchors_for_matrix_build.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'box_yolo': box})
        else:
            print("警告: YOLO会话未初始化，无法进行YOLO锚点检测。")

        # --- 步骤3: 构建智能网格并填充矩阵 ---
        # 调用我们新的核心函数 build_matrix_smart_grid
        final_matrix, filled_obu_count_in_matrix = build_matrix_smart_grid(
            yolo_anchors_input=yolo_anchors_for_matrix_build,
            paddle_results_input=potential_obu_list_paddle,
            layout_config=LAYOUT_CONFIG,
            image_wh=original_image.shape[:2]
        )
        print_matrix_to_console(final_matrix, f"智能网格矩阵 - {img_filename_base}")

        # 可视化PaddleOCR的原始和筛选结果 (保存到 Process_Photo 目录)
        paddle_viz_filename = f"paddle_ocr_viz_{img_filename_base}_{VERSION}.jpg"
        paddle_viz_path = os.path.join(PROCESS_PHOTO_DIR, paddle_viz_filename) # 使用 PROCESS_PHOTO_DIR
        if original_image is not None:
            draw_ocr_results_refined(original_image, all_paddle_ocr_data, potential_obu_list_paddle, paddle_viz_path)

        # TODO (未来): 增加一个可视化函数，专门绘制最终矩阵在原图上的效果，
        # 例如，在每个理想坑位画框，并标上识别的数字或“红”。

    overall_end_time = time.time()
    total_execution_time = overall_end_time - overall_start_time
    print(f"\n总运行时间: {total_execution_time:.3f} 秒。")
    print(f"-------------------------------------------------")