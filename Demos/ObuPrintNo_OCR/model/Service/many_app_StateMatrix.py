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
from collections import Counter
from scipy.spatial.distance import cdist
import uuid # For generating session IDs if needed

# --- 配置 ---
VERSION = "v5.3_db_check_layout_learning"

VALID_OBU_CODES = { # 您提供的有效OBU码列表
    "5001240700323449", "5001240700323450", "5001240700323445", "5001240700323446",
    "5001240700323447", "5001240700323448", "5001240700323441", "5001240700323442",
    "5001240700323443", "5001240700323444", "5001240700323437", "5001240700323438",
    "5001240700323439", "5001240700323440", "5001240700323433", "5001240700323434",
    "5001240700323435", "5001240700323436", "5001240700323430", "5001240700323431",
    "5001240700323432", "5001240700323429", "5001240700323428", "5001240700323427",
    "5001240700323426", "5001240700323425", "5001240700323424", "5001240700323423",
    "5001240700323422", "5001240700323421", "5001240700323420", "5001240700323419",
    "5001240700323418", "5001240700323417", "5001240700323416", "5001240700323415",
    "5001240700323414", "5001240700323413", "5001240700323412", "5001240700323411",
    "5001240700323410", "5001240700323409", "5001240700323408", "5001240700323407",
    "5001240700323406", "5001240700323405", "5001240700323404", "5001240700323403",
    "5001240700323402", "5001240700323401"
}

V8_MIN_CORE_ANCHORS_FOR_LAYOUT = 5
V8_MIN_VALID_ROWS_FOR_LAYOUT = 1 # 调整为1，因为单行特殊行也可能提供有用信息
V8_MIN_ANCHORS_PER_RELIABLE_ROW = 2 # 可靠行至少包含的锚点数
V8_MIN_ANCHORS_FOR_SPECIAL_ROW_GUESS = 2 # 特殊行至少应有的锚点数 (通常就是配置中的special_row_cols_count)

V8_ROW_GAP_WARN_MULTIPLIER = 2.0      # 行间距超过平均行高此倍数则警告
V8_COL_GAP_WARN_MULTIPLIER = 2.0      # 列间距超过平均OBU宽此倍数则警告 (在优质行内)

V8_MIN_REASONABLE_PIXEL_SIZE = 10
V8_MAX_REASONABLE_PIXEL_HEIGHT_FRACTION = 0.3 # 稍微放宽一点
V8_MAX_REASONABLE_PIXEL_WIDTH_FRACTION = 0.6  # 稍微放宽一点

V8_YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.4 # 用于行分组的因子

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
LOG_DIR = "log"
LOG_FILE = "app.log"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_MODEL_PATH_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "BarCode_Detect", "BarCode_Detect_dynamic.onnx")
SERVER_REC_MODEL_DIR_CFG_CONFIG = os.path.join(BASE_PROJECT_DIR, "model", "model", "PaddleOCR", "PP-OCRv5_server_rec_infer")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MIN_DETECTION_AREA = 2000
MAX_DETECTION_AREA = 0.1

DIGIT_ROI_Y_OFFSET_FACTOR = -0.15
DIGIT_ROI_HEIGHT_FACTOR = 0.7
DIGIT_ROI_WIDTH_EXPAND_FACTOR = 1.05
TARGET_OCR_INPUT_HEIGHT = 48
NUM_OCR_WORKERS_CONFIG = 4

PROCESS_PHOTO_DIR = "process_photo"
SAVE_PROCESS_PHOTOS = True
PROCESS_PHOTO_JPG_QUALITY = 85

LAYOUT_CONFIG = {
    "total_obus": 50,
    "regular_rows_count": 12,
    "regular_cols_count": 4,
    "special_row_cols_count": 2,
    "expected_total_rows": 13
}
YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR = 0.5
PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR = 0.75
MIN_YOLO_ANCHORS_FOR_LAYOUT = 10
COCO_CLASSES = ['Barcode']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

onnx_session = None
ocr_processing_pool = None
actual_num_ocr_workers = 1
session_data_store = {}

def learn_initial_layout_from_yolo_v81(current_yolo_boxes_with_orig_idx,
                                       current_config,
                                       image_wh,
                                       logger,
                                       session_id_for_log="N/A"):
    """
    V8.1 - 核心初始布局学习函数
    信任YOLO，聚焦关键结构（特别是特殊行），简化异常检测。
    目标：从单帧YOLO结果中，稳健地学习出一个（可能是局部的）高质量布局骨架。
    """
    log_prefix = f"会话 {session_id_for_log} (learn_initial_layout_V8.1):"
    logger.info(f"{log_prefix} 开始首次布局学习 (V8.1)...")

    warnings_list = []
    # 初始化返回的参数字典
    # avg_obu_w/h 和 avg_physical_row_height 稍后会从学习结果填充
    initial_layout_params = {
        "row_y_estimates": [],
        "col_x_estimates_regular": [],
        "avg_obu_w": 0.0,
        "avg_obu_h": 0.0,
        "avg_physical_row_height": 0.0,
        "special_row_at_logical_top": False, # 默认特殊行在底部
        "inferred_regular_cols_from_yolo": current_config.get("regular_cols_count", 4),
        "is_calibrated": False
    }
    img_w, img_h = image_wh
    expected_total_logical_rows = current_config.get("expected_total_rows", 13)
    expected_regular_cols = current_config.get("regular_cols_count", 4)
    special_row_expected_cols = current_config.get("special_row_cols_count", 2)

    # 1. 筛选核心锚点
    core_anchors = [a for a in current_yolo_boxes_with_orig_idx if
                    a.get('w', 0) > V8_MIN_REASONABLE_PIXEL_SIZE / 2 and # 宽度不能太小
                    a.get('h', 0) > V8_MIN_REASONABLE_PIXEL_SIZE / 2 and # 高度不能太小
                    a.get('score', 0.0) > 0.1] # 简单置信度过滤 (可调)
    if len(core_anchors) < V8_MIN_CORE_ANCHORS_FOR_LAYOUT:
        logger.warning(f"{log_prefix} 核心锚点数量 ({len(core_anchors)}) 过少 (少于{V8_MIN_CORE_ANCHORS_FOR_LAYOUT})。")
        return initial_layout_params, False, warnings_list
    logger.info(f"{log_prefix} 用于学习的核心锚点数量: {len(core_anchors)}")

    # 2. 行分组
    anchors_sorted_by_y = sorted(core_anchors, key=lambda a: (a['cy'], a['cx']))
    median_h_all_core = np.median([a['h'] for a in anchors_sorted_by_y]) if anchors_sorted_by_y else 30
    y_threshold_for_grouping = median_h_all_core * V8_YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    physical_rows_grouped = []
    if not anchors_sorted_by_y:
        logger.warning(f"{log_prefix} 排序后锚点列表为空。"); return initial_layout_params, False, warnings_list

    _current_row_group = [anchors_sorted_by_y[0]]
    for i in range(1, len(anchors_sorted_by_y)):
        if abs(anchors_sorted_by_y[i]['cy'] - _current_row_group[-1]['cy']) < y_threshold_for_grouping:
            _current_row_group.append(anchors_sorted_by_y[i])
        else:
            physical_rows_grouped.append(sorted(_current_row_group, key=lambda a: a['cx']))
            _current_row_group = [anchors_sorted_by_y[i]]
    if _current_row_group:
        physical_rows_grouped.append(sorted(_current_row_group, key=lambda a: a['cx']))

    reliable_physical_rows = [row for row in physical_rows_grouped if len(row) >= V8_MIN_ANCHORS_PER_RELIABLE_ROW]
    if len(reliable_physical_rows) < V8_MIN_VALID_ROWS_FOR_LAYOUT:
        logger.warning(f"{log_prefix} 可靠物理行数量 ({len(reliable_physical_rows)}) 过少 (少于{V8_MIN_VALID_ROWS_FOR_LAYOUT})。")
        # 即使只有一行，如果它是特殊行，也可能有用，所以这里不直接返回失败，后续判断
        if not any(len(r) == special_row_expected_cols for r in physical_rows_grouped): # 如果连疑似特殊行都没有
             return initial_layout_params, False, warnings_list

    logger.info(f"{log_prefix} 分组得到 {len(physical_rows_grouped)} 个物理行，其中 {len(reliable_physical_rows)} 个为可靠行。")

    # 3. 学习OBU平均像素尺寸 (从所有核心锚点)
    avg_obu_w = np.median([a['w'] for a in core_anchors])
    avg_obu_h = np.median([a['h'] for a in core_anchors])
    if not (V8_MIN_REASONABLE_PIXEL_SIZE < avg_obu_w < img_w * V8_MAX_REASONABLE_PIXEL_WIDTH_FRACTION and \
            V8_MIN_REASONABLE_PIXEL_SIZE < avg_obu_h < img_h * V8_MAX_REASONABLE_PIXEL_HEIGHT_FRACTION):
        logger.warning(f"{log_prefix} 学习到的OBU平均像素尺寸 (W:{avg_obu_w:.1f}, H:{avg_obu_h:.1f}) 超出合理范围。")
        return initial_layout_params, False, warnings_list
    initial_layout_params["avg_obu_w"] = avg_obu_w
    initial_layout_params["avg_obu_h"] = avg_obu_h
    logger.info(f"{log_prefix} OBU平均像素尺寸 (中位数): W={avg_obu_w:.1f}, H={avg_obu_h:.1f}")

    # 4. 学习行级参数 (avg_physical_row_height)
    avg_physical_row_height = avg_obu_h * 1.2 # 初始fallback，基于OBU高度
    if len(reliable_physical_rows) >= 2:
        row_centers_y = [np.mean([a['cy'] for a in row]) for row in reliable_physical_rows]
        row_gaps_y = np.abs(np.diff(row_centers_y))
        if row_gaps_y.size > 0:
            median_row_gap = np.median(row_gaps_y)
            if median_row_gap > avg_obu_h * 0.5 : # 确保学习到的行高比OBU自身高度要大一些
                avg_physical_row_height = median_row_gap
                # 检测“隔行”警告
                for gap in row_gaps_y:
                    if gap > avg_physical_row_height * V8_ROW_GAP_WARN_MULTIPLIER:
                        warn_msg = f"检测到可能的隔行，间距 {gap:.0f} > {avg_physical_row_height * V8_ROW_GAP_WARN_MULTIPLIER:.0f}"
                        warnings_list.append(warn_msg); logger.warning(f"{log_prefix} {warn_msg}")
                        break
    logger.info(f"{log_prefix} 平均物理行高估算: {avg_physical_row_height:.1f}像素")
    initial_layout_params["avg_physical_row_height"] = avg_physical_row_height


    # 5. 尝试定位特殊行，并确定其逻辑位置 (special_row_at_logical_top)
    #    同时，如果特殊行被定位，可以得到一个关于中间列X坐标的强提示
    special_row_physical_y = None
    special_row_cols_x_centers = [] # 如果找到特殊行，存储其两列的中心X

    # 优先从物理底部和顶部寻找特殊行
    candidate_special_rows_info = [] # (physical_y, cols_x_centers, is_at_bottom_or_top_edge)
    if physical_rows_grouped:
        # 检查物理最底部行
        last_phys_row = physical_rows_grouped[-1]
        if len(last_phys_row) == special_row_expected_cols:
            # 可加入更严格的居中判断，但先简化
            candidate_special_rows_info.append({
                "physical_y": np.mean([a['cy'] for a in last_phys_row]),
                "cols_x_centers": sorted([a['cx'] for a in last_phys_row]),
                "is_at_edge": "bottom",
                "num_anchors": len(last_phys_row)
            })
        # 检查物理最顶部行
        first_phys_row = physical_rows_grouped[0]
        if len(first_phys_row) == special_row_expected_cols:
            candidate_special_rows_info.append({
                "physical_y": np.mean([a['cy'] for a in first_phys_row]),
                "cols_x_centers": sorted([a['cx'] for a in first_phys_row]),
                "is_at_edge": "top",
                "num_anchors": len(first_phys_row)
            })

    # 如果在边缘找到了疑似特殊行
    if candidate_special_rows_info:
        # 简单策略：如果底部疑似行存在，优先用它；否则用顶部疑似行
        # (可以加入更复杂的选择逻辑，例如比较其“居中程度”或清晰度)
        chosen_special_row_info = None
        bottom_edge_special = next((info for info in candidate_special_rows_info if info["is_at_edge"] == "bottom"), None)
        top_edge_special = next((info for info in candidate_special_rows_info if info["is_at_edge"] == "top"), None)

        if bottom_edge_special:
            chosen_special_row_info = bottom_edge_special
            initial_layout_params["special_row_at_logical_top"] = False # 特殊行在逻辑底部
        elif top_edge_special:
            chosen_special_row_info = top_edge_special
            initial_layout_params["special_row_at_logical_top"] = True # 特殊行在逻辑顶部

        if chosen_special_row_info:
            special_row_physical_y = chosen_special_row_info["physical_y"]
            special_row_cols_x_centers = chosen_special_row_info["cols_x_centers"]
            logger.info(f"{log_prefix} 定位到疑似特殊行在物理 {chosen_special_row_info['is_at_edge']} (Y:{special_row_physical_y:.0f}). "
                        f"逻辑上是否在顶部: {initial_layout_params['special_row_at_logical_top']}. "
                        f"其列X中心: {special_row_cols_x_centers}")

    # 6. 推断常规列数 (从非特殊行中学习)
    non_special_like_rows = [row for row in reliable_physical_rows if len(row) != special_row_expected_cols]
    if not non_special_like_rows and reliable_physical_rows: # 如果所有可靠行都像特殊行，就用所有可靠行
        non_special_like_rows = reliable_physical_rows

    inferred_reg_cols = expected_regular_cols # Default
    if non_special_like_rows:
        col_counts = [len(r) for r in non_special_like_rows]
        if col_counts:
            mode_res = Counter(col_counts).most_common(1)
            if mode_res and mode_res[0][0] > 0 and mode_res[0][0] <= expected_regular_cols : # 不能超过期望的常规列数
                inferred_reg_cols = mode_res[0][0]
    initial_layout_params["inferred_regular_cols_from_yolo"] = inferred_reg_cols
    logger.info(f"{log_prefix} 推断常规列数 (从非特殊行): {inferred_reg_cols}")


    # 7. 确定锚定逻辑行 (用于生成row_y_estimates)
    anchor_physical_y = 0
    anchor_log_row = 0
    if reliable_physical_rows: # 必须有可靠行才能锚定
        if special_row_physical_y is not None: # 如果已定位特殊行
            if initial_layout_params["special_row_at_logical_top"]:
                anchor_physical_y = special_row_physical_y
                anchor_log_row = 0
            else: # 特殊行在逻辑底部
                anchor_physical_y = special_row_physical_y
                anchor_log_row = expected_total_logical_rows - 1
                # 此时，也可以尝试用最前景的可靠行作为逻辑0的参考，但要确保它不是特殊行本身
                first_reliable_non_special_row = next((r for r in reliable_physical_rows if len(r) != special_row_expected_cols), None)
                if first_reliable_non_special_row:
                     # 如果最前景的可靠行Y远小于特殊行Y，且特殊行确实在底部，那么前景行更适合做逻辑0的锚点
                     y_first_reliable = np.mean([a['cy'] for a in first_reliable_non_special_row])
                     if y_first_reliable < special_row_physical_y - avg_physical_row_height * 1.5: # 确保有足够距离
                        anchor_physical_y = y_first_reliable
                        anchor_log_row = 0
                        logger.info(f"{log_prefix} 特殊行在底，使用最前景非特殊行作为逻辑0锚点。")
        else: # 未明确找到特殊行，则默认特殊行在底部，使用最前景可靠行作为逻辑0锚点
            initial_layout_params["special_row_at_logical_top"] = False
            anchor_physical_y = np.mean([a['cy'] for a in reliable_physical_rows[0]])
            anchor_log_row = 0
        logger.info(f"{log_prefix} 最终锚定: 物理Y {anchor_physical_y:.1f} 对应逻辑行 {anchor_log_row}")
    else: # 没有可靠行，无法进行精确锚定 (理论上前面已返回失败)
        logger.error(f"{log_prefix} 无可靠行进行锚定，布局学习失败。")
        return initial_layout_params, False, warnings_list

    # 8. 生成 row_y_estimates
    initial_layout_params["row_y_estimates"] = [0.0] * expected_total_logical_rows
    for r_log in range(expected_total_logical_rows):
        initial_layout_params["row_y_estimates"][r_log] = anchor_physical_y + (r_log - anchor_log_row) * avg_physical_row_height
    logger.info(f"{log_prefix} 逻辑行Y估算: {[int(y) for y in initial_layout_params['row_y_estimates']]}")

    # 9. 生成 col_x_estimates_regular (核心细化)
    initial_layout_params["col_x_estimates_regular"] = [None] * expected_regular_cols

    # 优先使用特殊行（如果找到且为2列）来定位中间两列 (假设是4列常规布局)
    if special_row_cols_x_centers and len(special_row_cols_x_centers) == 2 and expected_regular_cols == 4:
        initial_layout_params["col_x_estimates_regular"][1] = special_row_cols_x_centers[0]
        initial_layout_params["col_x_estimates_regular"][2] = special_row_cols_x_centers[1]
        logger.info(f"{log_prefix} 使用特殊行信息估算逻辑列1和2的X中心: "
                    f"{initial_layout_params['col_x_estimates_regular'][1]:.0f}, "
                    f"{initial_layout_params['col_x_estimates_regular'][2]:.0f}")

    # 从“标准”常规行学习其他列或所有列（如果特殊行未提供信息）
    standard_rows_for_x = [row for row in reliable_physical_rows
                           if len(row) == inferred_reg_cols and len(row) != special_row_expected_cols]
    if not standard_rows_for_x and inferred_reg_cols > 0 :
        standard_rows_for_x = [row for row in reliable_physical_rows if len(row) > 1 and len(row) <= expected_regular_cols and len(row) != special_row_expected_cols]

    if standard_rows_for_x:
        temp_cols_x_accumulator = [[] for _ in range(inferred_reg_cols)] # 基于学习到的inferred_reg_cols
        for row in standard_rows_for_x:
            # 只处理那些列数与inferred_reg_cols完全匹配的行，以保证列索引的对应性
            if len(row) == inferred_reg_cols:
                for c_idx, anchor in enumerate(row):
                    temp_cols_x_accumulator[c_idx].append(anchor['cx'])

        for c_idx in range(inferred_reg_cols): # 遍历学习到的inferred_reg_cols
            if temp_cols_x_accumulator[c_idx]:
                learned_x_for_col = np.median(temp_cols_x_accumulator[c_idx])
                # 填充到 initial_layout_params["col_x_estimates_regular"]
                # 如果 inferred_reg_cols 小于 expected_regular_cols，这只会填充前面几列
                if c_idx < expected_regular_cols and initial_layout_params["col_x_estimates_regular"][c_idx] is None:
                    initial_layout_params["col_x_estimates_regular"][c_idx] = learned_x_for_col

    # --- 开始列X填充逻辑 (V8.1.1 修正版) ---
    current_cols_x_est = initial_layout_params["col_x_estimates_regular"]
    num_filled_cols = sum(1 for x_val in current_cols_x_est if x_val is not None)

    if num_filled_cols < expected_regular_cols:
        logger.info(f"{log_prefix} 需要填充 {expected_regular_cols - num_filled_cols} 个缺失的列X估算。")

        # 策略1: 如果是4列布局，且中间两列已知，则向外推断
        if expected_regular_cols == 4 and current_cols_x_est[1] is not None and current_cols_x_est[2] is not None:
            col1_x, col2_x = float(current_cols_x_est[1]), float(current_cols_x_est[2]) #确保是float
            if col1_x < col2_x:
                spacing_12_direct = col2_x - col1_x
                effective_spacing = spacing_12_direct

                # 判断 spacing_12_direct 的合理性
                # avg_obu_w 此时应该是有效的 (之前已计算并检查过)
                if not (avg_obu_w * 0.7 < spacing_12_direct < avg_obu_w * 1.5):
                    effective_spacing = avg_obu_w * 1.1
                    logger.warning(f"{log_prefix} 列1和2间距 {spacing_12_direct:.0f} 与avg_obu_w {avg_obu_w:.0f}差异大，使用估算间距 {effective_spacing:.0f} 外推。")
                else:
                    logger.info(f"{log_prefix} 使用列1和2的直接间距 {effective_spacing:.0f} 进行外推。")

                if current_cols_x_est[0] is None:
                    current_cols_x_est[0] = col1_x - effective_spacing
                    logger.info(f"{log_prefix} 推断列0 X: {current_cols_x_est[0]:.0f}")
                if current_cols_x_est[3] is None:
                    current_cols_x_est[3] = col2_x + effective_spacing
                    logger.info(f"{log_prefix} 推断列3 X: {current_cols_x_est[3]:.0f}")

        # 策略2: 如果仍有None，且至少有一列已知，则基于该列和avg_obu_w向两边推算
        num_filled_after_strategy1 = sum(1 for x_val in current_cols_x_est if x_val is not None)
        if num_filled_after_strategy1 < expected_regular_cols and num_filled_after_strategy1 > 0:
            first_known_idx, first_known_x = -1, 0.0
            for i_fx, x_fx_val in enumerate(current_cols_x_est):
                if x_fx_val is not None:
                    first_known_idx, first_known_x = i_fx, float(x_fx_val)
                    break

            if first_known_idx != -1:
                logger.info(f"{log_prefix} 基于首个已知列 {first_known_idx} (X:{first_known_x:.0f}) 和 avg_obu_w ({avg_obu_w:.0f}) 进行列填充。")
                for i_fill in range(expected_regular_cols):
                    if current_cols_x_est[i_fill] is None:
                        current_cols_x_est[i_fill] = first_known_x + (i_fill - first_known_idx) * (avg_obu_w * 1.1) # 使用1.1倍avg_obu_w作为估算列宽
                        logger.info(f"{log_prefix} Fallback推断列{i_fill} X: {current_cols_x_est[i_fill]:.0f}")

        # 策略3: 如果仍然有None (例如，一列都未学到)，则在图像中均匀分布
        num_filled_after_strategy2 = sum(1 for x_val in current_cols_x_est if x_val is not None)
        if num_filled_after_strategy2 < expected_regular_cols:
            logger.warning(f"{log_prefix} 列X估算仍含None ({expected_regular_cols - num_filled_after_strategy2}个)，尝试最终均匀分布。")
            # (这里的 all_core_anchors 应该在函数开始处定义，或者用 core_anchors)
            all_cxs_for_distribute = [a['cx'] for a in core_anchors] if core_anchors else []
            min_cx_overall = min(all_cxs_for_distribute) if all_cxs_for_distribute else img_w * 0.1
            max_cx_overall = max(all_cxs_for_distribute) if all_cxs_for_distribute else img_w * 0.9
            dist_region_w = max_cx_overall - min_cx_overall

            if dist_region_w < avg_obu_w * expected_regular_cols * 0.5 :
                 min_cx_overall = img_w * 0.1; max_cx_overall = img_w * 0.9
                 dist_region_w = max_cx_overall - min_cx_overall

            eff_w_per_col = dist_region_w / expected_regular_cols if expected_regular_cols > 0 else avg_obu_w
            for c_fill in range(expected_regular_cols):
                if current_cols_x_est[c_fill] is None:
                    current_cols_x_est[c_fill] = min_cx_overall + (c_fill + 0.5) * eff_w_per_col
                    logger.info(f"{log_prefix} 均匀分布填充列{c_fill} X: {current_cols_x_est[c_fill]:.0f}")

    initial_layout_params["col_x_estimates_regular"] = current_cols_x_est
    # --- 结束列X填充逻辑 ---

    logger.info(f"{log_prefix} 最终常规列X估算: {[int(x) if x is not None else 'N' for x in initial_layout_params['col_x_estimates_regular']]}")

    # （可选）“隔列”警告 - 基于最终的 col_x_estimates_regular
    if not any(x is None for x in initial_layout_params["col_x_estimates_regular"]) and len(initial_layout_params["col_x_estimates_regular"]) > 1:
        # 确保列表中的元素是数值类型，并且已排序
        valid_col_xs = sorted([float(x) for x in initial_layout_params["col_x_estimates_regular"] if x is not None])
        if len(valid_col_xs) > 1:
            final_col_gaps = np.diff(valid_col_xs)
            if final_col_gaps.size > 0:
                for gap_x in final_col_gaps:
                    if gap_x > avg_obu_w * V8_COL_GAP_WARN_MULTIPLIER:
                        warn_msg = f"最终列X估算中可能存在隔列，间距 {gap_x:.0f} > {avg_obu_w * V8_COL_GAP_WARN_MULTIPLIER:.0f}"
                        if warn_msg not in warnings_list:
                            warnings_list.append(warn_msg); logger.warning(f"{log_prefix} {warn_msg}")
                        break

    # 10. 参数合理性最终检查与返回
    if not initial_layout_params["row_y_estimates"] or \
       not initial_layout_params["col_x_estimates_regular"] or \
       any(x is None for x in initial_layout_params["col_x_estimates_regular"]) or \
       initial_layout_params["avg_physical_row_height"] <= V8_MIN_REASONABLE_PIXEL_SIZE / 2 or \
       initial_layout_params["avg_obu_w"] <= V8_MIN_REASONABLE_PIXEL_SIZE or \
       initial_layout_params["avg_obu_h"] <= V8_MIN_REASONABLE_PIXEL_SIZE:
        logger.error(f"{log_prefix} 学习到的最终布局参数不合理或不完整。学习失败。Params: {initial_layout_params}")
        # 即使学习失败，也尝试返回收集到的警告
        return initial_layout_params, False, warnings_list

    initial_layout_params["is_calibrated"] = True
    logger.info(f"{log_prefix} 首次布局学习成功 (V8.1.1)。警告: {warnings_list if warnings_list else '无'}")
    return initial_layout_params, True, warnings_list

    # 10. 参数合理性最终检查与返回
    if not initial_layout_params["row_y_estimates"] or \
       not initial_layout_params["col_x_estimates_regular"] or \
       any(x is None for x in initial_layout_params["col_x_estimates_regular"]) or \
       initial_layout_params["avg_physical_row_height"] <= V8_MIN_REASONABLE_PIXEL_SIZE / 2 or \
       initial_layout_params["avg_obu_w"] <= V8_MIN_REASONABLE_PIXEL_SIZE or \
       initial_layout_params["avg_obu_h"] <= V8_MIN_REASONABLE_PIXEL_SIZE:
        logger.error(f"{log_prefix} 学习到的最终布局参数不合理或不完整。学习失败。Params: {initial_layout_params}")
        return initial_layout_params, False, warnings_list

    initial_layout_params["is_calibrated"] = True
    logger.info(f"{log_prefix} 首次布局学习成功 (V8.1)。警告: {warnings_list if warnings_list else '无'}")
    return initial_layout_params, True, warnings_list

# --- 新增辅助函数：从锚点学习布局参数 ---
def _learn_layout_parameters_from_anchors(source_anchors, current_config, image_wh, logger, session_id_for_log):
    """
    辅助函数：从给定的锚点列表中学习布局参数。
    Args:
        source_anchors (list): 包含锚点字典的列表，每个字典应有 'cx', 'cy', 'w', 'h'。
        current_config (dict): 当前会话的布局配置。
        image_wh (tuple): (image_width, image_height)。
        logger: 日志记录器。
        session_id_for_log (str): 用于日志的会话ID。
    Returns:
        tuple: (learned_params_dict, success_flag)
               learned_params_dict: 包含学习到的布局参数的字典，如果学习失败则可能不完整或为None。
               success_flag: 布尔值，表示参数学习是否成功。
    """
    log_prefix = f"会话 {session_id_for_log} (_learn_layout):"
    logger.info(f"{log_prefix} 开始从 {len(source_anchors)} 个源锚点学习布局参数...")

    learned_params = {
        "is_calibrated": False, # 默认未校准，成功后会被设置为True
        "special_row_at_logical_top": False,
        "avg_physical_row_height": 0.0,
        "row_y_estimates": [],
        "col_x_estimates_regular": [],
        "inferred_regular_cols_from_yolo": 0, # 从这批锚点推断的常规列数
        "avg_obu_w": 0.0,
        "avg_obu_h": 0.0,
    }

    expected_total_logical_rows = current_config["expected_total_rows"]
    expected_regular_cols = current_config["regular_cols_count"]

    if not source_anchors or len(source_anchors) < current_config.get("min_anchors_for_layout_learning", MIN_YOLO_ANCHORS_FOR_LAYOUT // 2): # 使用配置或默认值
        logger.warning(f"{log_prefix} 源锚点数量 ({len(source_anchors)}) 过少，无法进行有效学习。")
        return learned_params, False

    # 1. 行分组 (与之前 refine_layout... 中的逻辑类似)
    anchors_sorted_by_y = sorted(source_anchors, key=lambda a: (a['cy'], a['cx']))
    rows_grouped_from_source = []
    avg_h_for_grouping = np.mean([a['h'] for a in anchors_sorted_by_y if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in anchors_sorted_by_y) else 30
    y_threshold_for_grouping = avg_h_for_grouping * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    if not anchors_sorted_by_y: # Should be caught by len(source_anchors) check, but defensive
        logger.warning(f"{log_prefix} 排序后的锚点列表为空。")
        return learned_params, False

    _current_row_group = [anchors_sorted_by_y[0]]
    for i in range(1, len(anchors_sorted_by_y)):
        if abs(anchors_sorted_by_y[i]['cy'] - _current_row_group[-1]['cy']) < y_threshold_for_grouping:
            _current_row_group.append(anchors_sorted_by_y[i])
        else:
            rows_grouped_from_source.append(sorted(_current_row_group, key=lambda a: a['cx']))
            _current_row_group = [anchors_sorted_by_y[i]]
    if _current_row_group:
        rows_grouped_from_source.append(sorted(_current_row_group, key=lambda a: a['cx']))

    if not rows_grouped_from_source:
        logger.warning(f"{log_prefix} 从源锚点行分组后结果为空。")
        return learned_params, False
    logger.info(f"{log_prefix} 源锚点行分组为 {len(rows_grouped_from_source)} 行。数量: {[len(r) for r in rows_grouped_from_source]}")

    # 2. 筛选高质量参考物理行
    reliable_physical_rows = [row for row in rows_grouped_from_source if len(row) >= 2]
    if len(reliable_physical_rows) < 2: # 需要至少2行可靠行
        logger.warning(f"{log_prefix} 可靠物理行数量 ({len(reliable_physical_rows)}) 不足2行。")
        return learned_params, False

    # 3. 智能判断特殊行位置 和 推断常规列数
    num_detected_reliable_rows = len(reliable_physical_rows)
    inferred_reg_cols = expected_regular_cols # Default
    possible_reg_cols_counts = [len(r) for r in reliable_physical_rows if len(r) != current_config["special_row_cols_count"]]
    if possible_reg_cols_counts:
        mode_res = Counter(possible_reg_cols_counts).most_common(1)
        if mode_res and mode_res[0][0] > 0:
            inferred_reg_cols = mode_res[0][0]
    learned_params["inferred_regular_cols_from_yolo"] = inferred_reg_cols # Store what was inferred from this batch
    logger.info(f"{log_prefix} 从本批锚点推断常规列数: {inferred_reg_cols}")

    special_row_at_bottom = False
    if len(reliable_physical_rows[-1]) == current_config["special_row_cols_count"] and \
       (num_detected_reliable_rows == 1 or abs(len(reliable_physical_rows[-2]) - inferred_reg_cols) <=1):
        special_row_at_bottom = True
    special_row_at_top = False
    if len(reliable_physical_rows[0]) == current_config["special_row_cols_count"] and \
       (num_detected_reliable_rows == 1 or abs(len(reliable_physical_rows[1]) - inferred_reg_cols) <=1):
        special_row_at_top = True

    if special_row_at_top and not special_row_at_bottom:
        learned_params["special_row_at_logical_top"] = True
    else:
        learned_params["special_row_at_logical_top"] = False
    logger.info(f"{log_prefix} 判断特殊行在逻辑顶部: {learned_params['special_row_at_logical_top']}")

    # 4. 学习平均物理行高 (MODIFIED: 使用中位数，并确保fallback逻辑正确对齐)
    physical_row_centers_y = [np.mean([a['cy'] for a in row]) for row in reliable_physical_rows]
    # Initial fallback for avg_phys_row_h
    avg_phys_row_h = avg_h_for_grouping * 1.2 if avg_h_for_grouping > 0 else 50 # Ensure a positive fallback

    if len(physical_row_centers_y) > 1:
        phys_row_heights_diffs = np.abs(np.diff(physical_row_centers_y))
        if phys_row_heights_diffs.size > 0:
            median_row_height_candidate = np.median(phys_row_heights_diffs)
            if median_row_height_candidate > 0: # 确保中位数有效
                avg_phys_row_h = median_row_height_candidate
                logger.info(f"{log_prefix} 使用行间距中位数更新 avg_phys_row_h: {avg_phys_row_h:.1f}")
            else:
                logger.info(f"{log_prefix} 行间距中位数 ({median_row_height_candidate:.1f}) 无效，avg_phys_row_h 保持为: {avg_phys_row_h:.1f}")
        else:
            logger.info(f"{log_prefix} 无有效行间距差异，avg_phys_row_h 保持为: {avg_phys_row_h:.1f}")
    else:
        logger.info(f"{log_prefix} 可靠物理行数量不足2 ({len(physical_row_centers_y)})，无法计算行间距，avg_phys_row_h 保持为: {avg_phys_row_h:.1f}")

    # Final check and fallback for avg_phys_row_h if it ended up non-positive
    if avg_phys_row_h <= 0:
        logger.warning(f"{log_prefix} 学习到的平均物理行高 ({avg_phys_row_h:.1f}) 无效，强制使用Fallback值 50。")
        avg_phys_row_h = 50 # Absolute fallback to a positive reasonable value
        logger.warning(f"{log_prefix} 学习到的平均物理行高 ({avg_phys_row_h:.1f}) 无效，使用Fallback。")
        avg_phys_row_h = avg_h_for_grouping * 1.2 if avg_h_for_grouping > 0 else 50

    learned_params["avg_physical_row_height"] = avg_phys_row_h
    logger.info(f"{log_prefix} 学习到平均物理行高: {avg_phys_row_h:.1f}")

    # 5. 估算所有逻辑行的物理Y坐标 (MODIFIED: 尝试从前景锚定)
    learned_params["row_y_estimates"] = [0.0] * expected_total_logical_rows

    # reliable_physical_rows 已经按Y坐标升序排列 (前景在前，远景在后)
    # physical_row_centers_y 对应 reliable_physical_rows 中每行的平均Y坐标

    anchor_phys_y = 0.0
    anchor_log_row = 0 # 默认锚定到逻辑行0

    if not reliable_physical_rows: # 防御性检查，理论上前面已经返回了
        logger.error(f"{log_prefix} 严重错误: reliable_physical_rows 为空，无法锚定行Y。")
        return learned_params, False

    # 策略：优先使用最前景的可靠行作为基准，尝试将其对应到逻辑上的前景行。
    # 我们将 reliable_physical_rows[0] (最前景的可靠行) 作为物理锚定点。
    anchor_phys_y = physical_row_centers_y[0]

    # 现在需要确定这个 anchor_phys_y 对应的 anchor_log_row
    # 这取决于我们如何解释 "special_row_at_logical_top" 和前景行的关系

    # 简化策略：我们将最前景的可靠物理行 (reliable_physical_rows[0])
    # 尝试映射到逻辑上的最前景行。
    # 如果特殊行在逻辑顶部 (L0)，并且 reliable_physical_rows[0] 就是那个特殊行，则 anchor_log_row = 0.
    # 如果特殊行在逻辑顶部 (L0)，但 reliable_physical_rows[0] 是一个常规行，则它应该是 L1 (或更高，但我们先简化为L1)。
    # 如果特殊行在逻辑底部 (L12)，那么 reliable_physical_rows[0] 自然对应 L0。

    is_first_reliable_row_special = (len(reliable_physical_rows[0]) == current_config["special_row_cols_count"])

    if learned_params["special_row_at_logical_top"]:
        if is_first_reliable_row_special:
            anchor_log_row = 0 # 最前景的可靠行就是顶部的特殊行 (L0)
            logger.info(f"{log_prefix} 锚定：最前景可靠行为顶部特殊行，作为逻辑行 0。")
        else:
            # 最前景的可靠行是一个常规行，而特殊行在逻辑顶部(L0)。
            # 这意味着这个前景常规行至少是L1。
            # 为了从前景开始推算，我们将其视为L1。
            anchor_log_row = 1
            logger.info(f"{log_prefix} 锚定：特殊行在逻辑顶部(L0)，最前景可靠常规行作为逻辑行 1。")
            # 注意：这种情况下，L0的Y坐标将通过 L1的Y - avg_phys_row_h 计算得到。
            # 这要求avg_phys_row_h的估算相对准确。
    else: # 特殊行在逻辑底部 (L12)
        anchor_log_row = 0 # 最前景的可靠行自然是逻辑行 0
        logger.info(f"{log_prefix} 锚定：特殊行在逻辑底部(L12)，最前景可靠行作为逻辑行 0。")

    logger.info(f"{log_prefix} 锚定物理Y: {anchor_phys_y:.1f} (来自reliable_physical_rows[0])，对应逻辑行: {anchor_log_row}")

    # 从锚定点开始，向两个方向推算所有逻辑行的Y坐标
    for r_log in range(expected_total_logical_rows):
        learned_params["row_y_estimates"][r_log] = anchor_phys_y + (r_log - anchor_log_row) * avg_phys_row_h

    logger.info(f"{log_prefix} 新估算逻辑行Y坐标: {[int(y) for y in learned_params['row_y_estimates']]}")

    # 6. 学习列X参数 (确保为所有 expected_regular_cols 提供估算)
    learned_params["col_x_estimates_regular"] = [None] * expected_regular_cols # Initialize with None
    all_source_cxs = [a['cx'] for a in source_anchors if 'cx' in a]
    avg_w_for_x_fallback = np.mean([a['w'] for a in source_anchors if a.get('w',0)>0]) if any(a.get('w',0)>0 for a in source_anchors) else 100

    learned_col_xs_map = {} # {logical_col_index: learned_x_value}
    candidate_rows_for_x = [r for r in reliable_physical_rows if len(r) == inferred_reg_cols and inferred_reg_cols > 0]
    if not candidate_rows_for_x and inferred_reg_cols > 0:
         candidate_rows_for_x = [r for r in reliable_physical_rows if abs(len(r) - inferred_reg_cols) <= 1 and len(r) > 0]

    if candidate_rows_for_x:
        temp_col_xs_acc = [[] for _ in range(inferred_reg_cols)]
        for row in candidate_rows_for_x:
            for col_idx_in_row, anchor in enumerate(row):
                if col_idx_in_row < inferred_reg_cols:
                    temp_col_xs_acc[col_idx_in_row].append(anchor['cx'])
        for c_log_inferred in range(inferred_reg_cols):
            if temp_col_xs_acc[c_log_inferred]:
                learned_col_xs_map[c_log_inferred] = np.mean(temp_col_xs_acc[c_log_inferred])

    for c_idx in range(expected_regular_cols):
        if c_idx in learned_col_xs_map:
            learned_params["col_x_estimates_regular"][c_idx] = learned_col_xs_map[c_idx]

    # Fallback for columns not learned (same logic as before in refine_layout...)
    num_learned_direct_cols = len(learned_col_xs_map)
    if num_learned_direct_cols < expected_regular_cols:
        logger.info(f"{log_prefix} 需要为 {expected_regular_cols - num_learned_direct_cols} 个常规列进行X坐标Fallback。")
        # (Fallback logic: copy from previous refine_layout... implementation)
        # Define a region for distributing columns
        min_cx_overall = min(all_source_cxs) if all_source_cxs else image_wh[0] * 0.1
        max_cx_overall = max(all_source_cxs) if all_source_cxs else image_wh[0] * 0.9
        if not all_source_cxs or min_cx_overall >= max_cx_overall - avg_w_for_x_fallback:
            min_cx_overall = image_wh[0] * 0.15
            max_cx_overall = image_wh[0] * 0.85
        distribute_region_w = max_cx_overall - min_cx_overall

        current_col_xs_est = list(learned_params["col_x_estimates_regular"]) # Make a mutable copy

        if num_learned_direct_cols == 1:
            learned_c_idx = list(learned_col_xs_map.keys())[0]
            learned_x_val = list(learned_col_xs_map.values())[0]
            eff_col_spacing = avg_w_for_x_fallback * 1.1
            for c_target in range(expected_regular_cols):
                if current_col_xs_est[c_target] is None:
                     current_col_xs_est[c_target] = learned_x_val + (c_target - learned_c_idx) * eff_col_spacing
        elif num_learned_direct_cols >= 2:
            sorted_learned_indices = sorted(learned_col_xs_map.keys())
            spacings = [(learned_col_xs_map[idx2] - learned_col_xs_map[idx1]) / (idx2 - idx1)
                        for i in range(len(sorted_learned_indices) - 1)
                        for idx1, idx2 in [(sorted_learned_indices[i], sorted_learned_indices[i+1])] if idx2 > idx1]
            avg_spacing = np.mean(spacings) if spacings else avg_w_for_x_fallback * 1.1
            first_l_idx, last_l_idx = min(sorted_learned_indices), max(sorted_learned_indices)
            for c_target in range(expected_regular_cols):
                if current_col_xs_est[c_target] is None:
                    if c_target < first_l_idx: current_col_xs_est[c_target] = learned_col_xs_map[first_l_idx] - (first_l_idx - c_target) * avg_spacing
                    elif c_target > last_l_idx: current_col_xs_est[c_target] = learned_col_xs_map[last_l_idx] + (c_target - last_l_idx) * avg_spacing
                    else: # Interpolate
                        prev_l = max([l for l in sorted_learned_indices if l < c_target])
                        next_l = min([l for l in sorted_learned_indices if l > c_target])
                        ratio = (c_target - prev_l) / (next_l - prev_l) if next_l > prev_l else 0
                        current_col_xs_est[c_target] = learned_col_xs_map[prev_l] + ratio * (learned_col_xs_map[next_l] - learned_col_xs_map[prev_l])

        if any(x is None for x in current_col_xs_est): # Still Nones, distribute evenly
            logger.info(f"{log_prefix} 部分X坐标使用均匀分布Fallback。区域: {min_cx_overall:.0f}-{max_cx_overall:.0f}")
            eff_w_per_col = distribute_region_w / expected_regular_cols
            if eff_w_per_col < avg_w_for_x_fallback * 0.8:
                min_cx_overall, max_cx_overall = image_wh[0] * 0.15, image_wh[0] * 0.85
                distribute_region_w = max_cx_overall - min_cx_overall
                eff_w_per_col = distribute_region_w / expected_regular_cols
            for c_target in range(expected_regular_cols):
                if current_col_xs_est[c_target] is None:
                    current_col_xs_est[c_target] = min_cx_overall + (c_target + 0.5) * eff_w_per_col
        learned_params["col_x_estimates_regular"] = current_col_xs_est

    logger.info(f"{log_prefix} 最终常规列X估算 (长度 {len(learned_params['col_x_estimates_regular'])}): "
                f"{[int(x) if x is not None else 'None' for x in learned_params['col_x_estimates_regular']]}")

    if any(x is None for x in learned_params["col_x_estimates_regular"]):
        logger.error(f"{log_prefix} 学习后 col_x_estimates_regular 仍含None。学习失败。")
        return learned_params, False # Failed

    # 7. 估算平均OBU尺寸
    valid_widths = [a['w'] for a in source_anchors if a.get('w', 0) > 0]
    valid_heights = [a['h'] for a in source_anchors if a.get('h', 0) > 0]

    if valid_widths:
        learned_params["avg_obu_w"] = np.median(valid_widths)
        logger.info(f"{log_prefix} 估算 avg_obu_w (中位数): {learned_params['avg_obu_w']:.1f} (基于 {len(valid_widths)} 个样本)")
    else:
        learned_params["avg_obu_w"] = 100 # Fallback
        logger.warning(f"{log_prefix} 无有效宽度样本估算 avg_obu_w，使用 fallback: 100")

    if valid_heights:
        learned_params["avg_obu_h"] = np.median(valid_heights)
        logger.info(f"{log_prefix} 估算 avg_obu_h (中位数): {learned_params['avg_obu_h']:.1f} (基于 {len(valid_heights)} 个样本)")
    else:
        learned_params["avg_obu_h"] = 40 # Fallback
        logger.warning(f"{log_prefix} 无有效高度样本估算 avg_obu_h，使用 fallback: 40")

    learned_params["is_calibrated"] = True # Mark as successfully learned from this batch
    logger.info(f"{log_prefix} 布局参数学习成功。")
    return learned_params, True
# --- 结束新增辅助函数 ---

# --- 修改 refine_layout_and_map_yolo_to_logical V8.1 ---
# (假设 learn_initial_layout_from_yolo_v81 和相关常量已定义)

def refine_layout_and_map_yolo_to_logical(current_yolo_boxes_with_orig_idx,
                                              session_id,
                                              image_wh,
                                              logger):
    log_prefix = f"会话 {session_id} (refine_layout V8.1):"
    logger.info(f"{log_prefix} 开始处理布局与映射 (集成V8.1初始学习)...")

    # 平滑参数 (与V7.3.1类似，但主要用于位置参数)
    ALPHA_INITIAL_CALIBRATION = 0.7
    ALPHA_STABLE_STATE = 0.4
    STABLE_STATE_THRESHOLD_FRAMES = 3

    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"{log_prefix} 严重错误 - 未找到会话数据。")
        return {}, False # 只返回 mapping_result, updated_flag

    current_layout_params = session["layout_parameters"] # 包含 is_calibrated, 以及各种估算值
    current_config = session["current_layout_config"]
    obu_evidence_pool = session.get("obu_evidence_pool", {})
    frames_since_recal = session.get("frames_since_last_recalibration", 0)

    # V8.1中，avg_obu_w/h 由 learn_initial_layout_from_yolo_v81 学习并直接存入 layout_params
    # 后续映射直接从 layout_params 取用，不再单独管理 stable_avg_obu_w/h 或累积列表

    layout_updated_this_run = False
    warnings_from_current_learn = []
    source_anchors_for_learning = []

    # 1. 准备学习锚点 (与之前版本逻辑类似)
    is_currently_calibrated = current_layout_params.get("is_calibrated", False)
    # 何时需要重新学习/优化参数？
    # a. 尚未校准 (is_currently_calibrated is False)
    # b. 或者，即使已校准，我们也可能希望每隔几帧或在特定条件下尝试优化 (简化：暂时只在未校准时学习)
    #    对于V8.1的首次迭代，我们先简化为：只有在未校准时才调用 learn_initial_layout_from_yolo_v81
    #    后续可以加入基于obu_evidence_pool的参数微调逻辑，但这会更复杂。

    should_learn_params = not is_currently_calibrated

    if should_learn_params:
        logger.info(f"{log_prefix} 状态: 未校准或需要重新学习。准备学习锚点...")
        # 首次学习，可以只用当前帧的，或者也考虑合并历史（如果历史的首次校准失败了）
        # 为简化，V8.1的首次学习严格只用当前帧
        source_anchors_for_learning = list(current_yolo_boxes_with_orig_idx)
        if not source_anchors_for_learning:
            logger.warning(f"{log_prefix} 首次学习时无源锚点，无法进行。")
            return {}, False

        logger.info(f"{log_prefix} 使用 {len(source_anchors_for_learning)} 个当前帧锚点进行初始布局学习。")

        # 调用新的V8.1初始布局学习函数
        newly_learned_params, learning_success, warnings_from_learn = learn_initial_layout_from_yolo_v81(
            source_anchors_for_learning, current_config, image_wh, logger, session_id
        )
        warnings_from_current_learn.extend(warnings_from_learn)

        if learning_success:
            logger.info(f"{log_prefix} 初始布局学习成功。应用新参数。")
            session["layout_parameters"] = newly_learned_params # is_calibrated 已在内部设为True
            session["frames_since_last_recalibration"] = 1 # 重置/开始计数
            layout_updated_this_run = True
            current_layout_params = newly_learned_params # 更新本地引用
        else:
            logger.error(f"{log_prefix} 初始布局学习失败。保持未校准状态。")
            session["layout_parameters"]["is_calibrated"] = False
            return {}, False
    else: # 已校准，暂时不进行参数优化，直接使用现有参数
        logger.info(f"{log_prefix} 布局已校准，使用现有参数。Frames since recal: {frames_since_recal}")
        # 如果需要后续帧的参数平滑/优化，逻辑应在此处添加，
        # 例如，可以每隔N帧或当检测到布局漂移时，再次调用 learn_initial_layout_from_yolo_v81
        # (基于 obu_evidence_pool + current_frame_yolo) 并进行平滑。
        # 但对于V8.1的首次迭代，我们先简化，已校准后参数暂时固定。
        # 如果要平滑，frames_since_recal 需要在这里递增：
        # session["frames_since_last_recalibration"] = frames_since_recal + 1
        pass


    # 2. 将当前帧的YOLO锚点映射到逻辑坐标
    # 始终使用 session["layout_parameters"] 中最新的有效参数
    final_params_for_mapping = session["layout_parameters"]
    current_frame_mapping = {}

    if not final_params_for_mapping.get("is_calibrated", False):
        logger.warning(f"{log_prefix} 布局参数仍未校准/无效，无法为当前帧YOLO锚点进行精确映射。")
    return {}, layout_updated_this_run

    row_ys_est = final_params_for_mapping.get("row_y_estimates", [])
    col_xs_est_reg = final_params_for_mapping.get("col_x_estimates_regular", [])
    expected_total_logical_rows = current_config.get("expected_total_rows", 13)
    expected_regular_cols = current_config.get("regular_cols_count", 4)

    # 从布局参数中获取学习到的OBU平均尺寸，用于计算匹配阈值
    # learn_initial_layout_from_yolo_v81 应该已经填充了 avg_obu_w 和 avg_obu_h
    map_avg_h = final_params_for_mapping.get("avg_obu_h", 0)
    map_avg_w = final_params_for_mapping.get("avg_obu_w", 0)

    if map_avg_h <= 0: map_avg_h = 50; logger.warning(f"{log_prefix} avg_obu_h 无效，使用默认值50进行映射。")
    if map_avg_w <= 0: map_avg_w = 100; logger.warning(f"{log_prefix} avg_obu_w 无效，使用默认值100进行映射。")

    # 使用固定的阈值因子 (之前讨论的0.7或0.9，这里用一个常量)
    # 或者，这个因子也可以考虑动态调整，但先简化
    MAP_THRESHOLD_FACTOR = 0.8 # 可以调整这个因子
    y_match_threshold = map_avg_h * MAP_THRESHOLD_FACTOR
    x_match_threshold = map_avg_w * MAP_THRESHOLD_FACTOR

    logger.info(f"{log_prefix} 用于映射的OBU尺寸: H={map_avg_h:.2f}, W={map_avg_w:.2f}")
    logger.info(f"{log_prefix} 计算得到的匹配阈值 (因子 {MAP_THRESHOLD_FACTOR}): Y_thresh={y_match_threshold:.2f}, X_thresh={x_match_threshold:.2f}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{log_prefix} 使用的行Y估算: {[int(y) for y in row_ys_est]}")
        logger.debug(f"{log_prefix} 使用的列X估算: {[int(x) if x is not None else 'N' for x in col_xs_est_reg]}")

    if not row_ys_est or not col_xs_est_reg or any(x is None for x in col_xs_est_reg):
        logger.error(f"{log_prefix} 映射时布局参数无效 (行列估算为空或含None)。ColX: {col_xs_est_reg}")
        return {}, layout_updated_this_run

    # --- 映射逻辑 (与V7.3.1的日志增强版类似) ---
    for anchor in current_yolo_boxes_with_orig_idx:
        if 'cx' not in anchor or 'cy' not in anchor: continue
        cand_r = -1; min_y_d_sq = float('inf')
        for r_idx, est_y in enumerate(row_ys_est):
            dist_y_sq = (anchor['cy'] - est_y)**2
            if dist_y_sq < min_y_d_sq and dist_y_sq < y_match_threshold**2 :
                min_y_d_sq = dist_y_sq; cand_r = r_idx

        if cand_r == -1:
            if logger.isEnabledFor(logging.DEBUG):
                distances_y = [abs(anchor['cy'] - est_y) for est_y in row_ys_est]
                logger.debug(f"{log_prefix} 锚点(idx:{anchor.get('original_index','N/A')}, cy:{anchor['cy']:.0f}) 未匹配到任何逻辑行。 "
                             f"Y距离: {[f'{d:.1f}' for d in distances_y]}. Y阈值: {y_match_threshold:.1f}")
            continue

        is_special_row_map = (cand_r == (expected_total_logical_rows - 1) and not final_params_for_mapping.get("special_row_at_logical_top", False)) or \
                             (cand_r == 0 and final_params_for_mapping.get("special_row_at_logical_top", False))
        current_row_col_xs_to_match = []
        # (获取 current_row_col_xs_to_match 的逻辑与V7.3.1一致)
        if is_special_row_map and special_row_expected_cols == 2 and expected_regular_cols == 4:
            if len(col_xs_est_reg) == 4 and col_xs_est_reg[1] is not None and col_xs_est_reg[2] is not None:
                 current_row_col_xs_to_match = [col_xs_est_reg[1], col_xs_est_reg[2]]
            else:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(f"{log_prefix} 特殊行{cand_r} col_xs_est_reg 无效，跳过锚点 {anchor.get('original_index','N/A')}");
                continue
        else: # 常规行
            # 假设 col_xs_est_reg 的长度就是 expected_regular_cols 且不含None (learn_initial_layout_from_yolo_v81应保证)
            if len(col_xs_est_reg) == expected_regular_cols:
                current_row_col_xs_to_match = col_xs_est_reg
            else: # 如果长度不匹配，这是个问题
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{log_prefix} 常规行{cand_r} col_xs_est_reg 长度 ({len(col_xs_est_reg)}) "
                                 f"与期望 ({expected_regular_cols}) 不符，跳过锚点 {anchor.get('original_index','N/A')}")
                continue
        if not current_row_col_xs_to_match:
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"{log_prefix} 逻辑行 {cand_r} 无有效列X坐标，跳过锚点 {anchor.get('original_index','N/A')}");
            continue

        cand_c_in_options_idx = -1; min_x_d_sq = float('inf')
        for c_idx_in_row, est_x in enumerate(current_row_col_xs_to_match):
            if est_x is None: continue # 防御：如果列X估算中有None
            dist_x_sq = (anchor['cx'] - est_x)**2
            if dist_x_sq < min_x_d_sq and dist_x_sq < x_match_threshold**2 :
                min_x_d_sq = dist_x_sq; cand_c_in_options_idx = c_idx_in_row

        if cand_c_in_options_idx != -1:
            best_r_final = cand_r
            best_c_final = cand_c_in_options_idx + 1 if (is_special_row_map and special_row_expected_cols == 2 and expected_regular_cols == 4) else cand_c_in_options_idx
            if 0 <= best_c_final < expected_regular_cols:
                 current_frame_mapping[anchor['original_index']] = (best_r_final, best_c_final)
            else: # 无效逻辑列
                 logger.warning(f"{log_prefix} 锚点 {anchor.get('original_index','N/A')} (cx:{anchor['cx']:.0f}) "
                                f"在逻辑行 {best_r_final} 映射后得无效逻辑列 {best_c_final}. ")
        else: # 列匹配失败
            if logger.isEnabledFor(logging.DEBUG):
                distances_x = [abs(anchor['cx'] - est_x) for est_x in current_row_col_xs_to_match if est_x is not None]
                logger.debug(f"{log_prefix} 锚点(idx:{anchor.get('original_index','N/A')}, cx:{anchor['cx']:.0f}) 在逻辑行 {cand_r} (Y匹配成功) 未匹配到任何逻辑列。 "
                             f"尝试的列X估算: {[int(x) if x is not None else 'N' for x in current_row_col_xs_to_match]}. "
                             f"X距离: {[f'{d:.1f}' for d in distances_x]}. X阈值: {x_match_threshold:.1f}")

    logger.info(f"{log_prefix} 当前帧YOLO锚点映射完成，共映射 {len(current_frame_mapping)} 个。布局更新标志: {layout_updated_this_run}")
    # warnings_from_current_learn 已经在函数内部被logger记录，暂时不作为主要返回值向上传递
    return current_frame_mapping, layout_updated_this_run

# --- V5.3: 更新会话状态 (V5.1 P6 - 调试最终矩阵重构) ---
def update_session_state_from_ocr(session_id, current_frame_yolo_logical_map, ocr_results_this_frame, logger):
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 在 update_session_state_from_ocr 中未找到会话数据。")
        return

    logger.info(f"会话 {session_id}: (V5.1 P6 Debug) 开始更新状态矩阵和证据池...")

    logical_matrix = session["logical_matrix"]
    recognized_texts_map = session["recognized_texts_map"]
    obu_evidence_pool = session["obu_evidence_pool"]
    layout_params = session["layout_parameters"] # is_calibrated, row_y_estimates, col_x_estimates_regular, avg_physical_row_height etc.
    current_config = session["current_layout_config"]
    frame_count = session.get("frame_count", 0)

    # 新增：从会话获取稳定OBU尺寸
    stable_avg_obu_w_sess = session.get("stable_avg_obu_w")
    stable_avg_obu_h_sess = session.get("stable_avg_obu_h")
    logger.debug(f"会话 {session_id}: (update_session) 获取到的稳定尺寸 W: {stable_avg_obu_w_sess}, H: {stable_avg_obu_h_sess}")

    # 1. 将当前帧成功校验的OCR结果更新到 obu_evidence_pool (此部分逻辑不变)
    # ... (原有的更新 obu_evidence_pool 的代码保持不变) ...
    newly_added_to_pool_this_frame = []
    for ocr_item in ocr_results_this_frame:
        if not ocr_item: continue
        original_yolo_idx = ocr_item.get("original_index")
        ocr_text = ocr_item.get("ocr_final_text", "")
        ocr_confidence = ocr_item.get("ocr_confidence", 0.0)
        is_valid_in_db = ocr_text in VALID_OBU_CODES
        if is_valid_in_db:
            yolo_anchor_details = ocr_item.get('yolo_anchor_details')
            if not yolo_anchor_details or not all(k in yolo_anchor_details for k in ['cx', 'cy', 'w', 'h']):
                logger.warning(f"会话 {session_id}: 有效OBU '{ocr_text}' 缺少关联YOLO锚点详细信息，仅尝试更新文本和置信度。")
                if ocr_text not in obu_evidence_pool:
                    obu_evidence_pool[ocr_text] = {'physical_anchors': [], 'ocr_confidence': 0.0, 'logical_coord': None, 'last_seen_frame':0}
                    newly_added_to_pool_this_frame.append(ocr_text)
                obu_evidence_pool[ocr_text]['ocr_confidence'] = max(obu_evidence_pool[ocr_text]['ocr_confidence'], ocr_confidence)
                obu_evidence_pool[ocr_text]['last_seen_frame'] = frame_count
                continue
            if ocr_text not in obu_evidence_pool:
                obu_evidence_pool[ocr_text] = {'physical_anchors': [], 'ocr_confidence': 0.0, 'logical_coord': None, 'last_seen_frame':0}
                newly_added_to_pool_this_frame.append(ocr_text)
            current_anchor_data = {
                'cx': yolo_anchor_details['cx'], 'cy': yolo_anchor_details['cy'],
                'w': yolo_anchor_details['w'], 'h': yolo_anchor_details['h'],
                'score': yolo_anchor_details.get('score', 0.0),
                'frame_id': frame_count
            }
            obu_evidence_pool[ocr_text]['physical_anchors'] = [current_anchor_data]
            obu_evidence_pool[ocr_text]['ocr_confidence'] = max(obu_evidence_pool[ocr_text]['ocr_confidence'], ocr_confidence)
            obu_evidence_pool[ocr_text]['last_seen_frame'] = frame_count
    logger.info(f"会话 {session_id}: 本轮新增/更新 {len(newly_added_to_pool_this_frame)} 个OBU到证据池。证据池总数: {len(obu_evidence_pool)}")


    # 2. 基于更新后的 obu_evidence_pool 和最新的 layout_params, 重新构建最终的输出逻辑矩阵
    if not layout_params.get("is_calibrated"):
        logger.warning(f"会话 {session_id}: (update_session) 布局参数未校准，无法基于证据池高质量重构矩阵。将尝试使用当前帧的初步映射。")
        # ... (布局未校准时的处理逻辑，与之前版本一致，保持不变) ...
        for yolo_idx, (r_log, c_log) in current_frame_yolo_logical_map.items():
            ocr_item_for_this_yolo = next((ocr for ocr in ocr_results_this_frame if ocr and ocr.get("original_index") == yolo_idx), None)
            if ocr_item_for_this_yolo:
                ocr_text = ocr_item_for_this_yolo.get("ocr_final_text", "")
                if ocr_text in VALID_OBU_CODES:
                    if logical_matrix[r_log][c_log] == 0 or logical_matrix[r_log][c_log] == 2 :
                        logical_matrix[r_log][c_log] = 1
                        recognized_texts_map[(r_log, c_log)] = ocr_text
                elif logical_matrix[r_log][c_log] == 0:
                    logical_matrix[r_log][c_log] = 2
    else: # 布局已校准，执行全局重排版
        logger.info(f"会话 {session_id}: (update_session) 基于证据池({len(obu_evidence_pool)}个OBU)和已校准布局，开始重构最终逻辑矩阵...")

        # a. 先“清空”输出矩阵和文本映射 (逻辑不变)
        # ... (清空矩阵和文本映射的代码，与之前版本一致，保持不变) ...
        for r_clear in range(len(logical_matrix)):
            for c_clear in range(len(logical_matrix[r_clear])):
                if logical_matrix[r_clear][c_clear] != -1:
                    logical_matrix[r_clear][c_clear] = 0
                    if (r_clear,c_clear) in recognized_texts_map:
                        del recognized_texts_map[(r_clear,c_clear)]
        logger.debug(f"会话 {session_id}: (重构前) 逻辑矩阵和文本映射已清空。")

        if not obu_evidence_pool:
            logger.warning(f"会话 {session_id}: (重构矩阵) OBU证据池为空，矩阵将保持清空状态。")

        # b. 遍历 obu_evidence_pool, 为每个已验证OBU重新推断最佳逻辑位置并填充
        row_ys_lp = layout_params.get("row_y_estimates", [])
        col_xs_lp_reg = layout_params.get("col_x_estimates_regular", [])
        expected_total_rows_lp = current_config["expected_total_rows"]
        # 使用 layout_params 中存储的、由 _learn_layout_parameters_from_anchors 推断的常规列数
        inferred_regular_cols_lp = layout_params.get("inferred_regular_cols_from_yolo", current_config["regular_cols_count"])


        if not row_ys_lp or not col_xs_lp_reg or not col_xs_lp_reg: # 确保行列估算存在
            logger.warning(f"会话 {session_id}: (重构矩阵) 布局参数中的行列估算为空或无效，无法进行映射填充。")
        else:
            # --- 使用布局参数中学习到的OBU尺寸计算匹配阈值 (for evidence pool mapping) ---
            map_avg_h_evidence = layout_params.get("avg_obu_h", 0)
            map_avg_w_evidence = layout_params.get("avg_obu_w", 0)

            if map_avg_h_evidence <= 0: map_avg_h_evidence = 50; logger.warning(f"{log_prefix} (重构) avg_obu_h 无效，使用默认值50。")
            if map_avg_w_evidence <= 0: map_avg_w_evidence = 100; logger.warning(f"{log_prefix} (重构) avg_obu_w 无效，使用默认值100。")

            # 使用与 refine_layout_and_map_yolo_to_logical_v81 中一致的因子
            MAP_THRESHOLD_FACTOR_OCR = 0.8 # 保持与 refine_layout 一致
            y_match_thresh_lp = map_avg_h_evidence * MAP_THRESHOLD_FACTOR_OCR
            x_match_thresh_lp = map_avg_w_evidence * MAP_THRESHOLD_FACTOR_OCR
            logger.debug(f"  (重构) 用于证据池映射的OBU尺寸: H={map_avg_h_evidence:.1f}, W={map_avg_w_evidence:.1f} "
                         f"=> Y_thresh={y_match_thresh_lp:.1f}, X_thresh={x_match_thresh_lp:.1f} (因子 {MAP_THRESHOLD_FACTOR_OCR})")

            candidate_obus_for_slots = {}
            logger.debug(f"会话 {session_id}: (重构矩阵) 开始遍历 {len(obu_evidence_pool)} 个OBU证据...")
            # ... (后续的 for obu_text_verified, evidence in obu_evidence_pool.items(): 循环及其内部的映射和冲突解决逻辑，与V7.2版本基本一致，它们将使用上面新计算的阈值)

        if not row_ys_lp or not col_xs_lp_reg:
            logger.warning(f"会话 {session_id}: (重构矩阵) 布局参数中的行列估算为空，无法进行映射填充。")
        else:
            candidate_obus_for_slots = {}
            logger.debug(f"会话 {session_id}: (重构矩阵) 开始遍历 {len(obu_evidence_pool)} 个OBU证据...")
            for obu_text_verified, evidence in obu_evidence_pool.items():
                if not evidence.get('physical_anchors'):
                    logger.debug(f"  (重构) OBU '{obu_text_verified}' 无物理锚点，跳过。")
                    continue

                anchor = evidence['physical_anchors'][-1]
                r_final, c_final = -1, -1
                min_dist_sq_final = float('inf')
                logger.debug(f"  (重构) 尝试映射 OBU '{obu_text_verified}' (锚点 cx:{anchor['cx']}, cy:{anchor['cy']})")

                for r_idx, est_y in enumerate(row_ys_lp):
                    dist_y_sq = (anchor['cy'] - est_y)**2
                    if dist_y_sq < y_match_thresh_lp**2 :
                        is_special_row_rebuild = (r_idx == (expected_total_rows_lp - 1) and not layout_params.get("special_row_at_logical_top", False)) or \
                                                 (r_idx == 0 and layout_params.get("special_row_at_logical_top", False))
                        cols_in_this_logical_row_rebuild = current_config["special_row_cols_count"] if is_special_row_rebuild else inferred_regular_cols_lp

                        current_row_col_xs_to_match_rebuild = []
                        if is_special_row_rebuild and current_config["special_row_cols_count"] == 2 and inferred_regular_cols_lp == 4:
                            if len(col_xs_lp_reg) == 4: current_row_col_xs_to_match_rebuild = [col_xs_lp_reg[1], col_xs_lp_reg[2]]
                        else:
                            if 0 <= cols_in_this_logical_row_rebuild <= len(col_xs_lp_reg): current_row_col_xs_to_match_rebuild = col_xs_lp_reg[:cols_in_this_logical_row_rebuild]
                            else: current_row_col_xs_to_match_rebuild = col_xs_lp_reg
                        if not current_row_col_xs_to_match_rebuild: continue

                        for c_idx_in_row, est_x in enumerate(current_row_col_xs_to_match_rebuild):
                            dist_sq_cand = dist_y_sq + (anchor['cx'] - est_x)**2
                            if dist_sq_cand < min_dist_sq_final and (anchor['cx'] - est_x)**2 < x_match_thresh_lp**2:
                                min_dist_sq_final = dist_sq_cand
                                r_final = r_idx
                                if is_special_row_rebuild and current_config["special_row_cols_count"] == 2 and inferred_regular_cols_lp == 4:
                                    c_final = c_idx_in_row + 1
                                else: c_final = c_idx_in_row
                                if not (0 <= c_final < current_config["regular_cols_count"]): c_final = -1

                if r_final != -1 and c_final != -1:
                    logger.debug(f"    OBU '{obu_text_verified}' 初步映射到 ({r_final},{c_final})，距离平方 {min_dist_sq_final:.2f}")
                    current_score = evidence.get('ocr_confidence', 0.0)

                    if (r_final, c_final) not in candidate_obus_for_slots or \
                       current_score > candidate_obus_for_slots[(r_final, c_final)]['score']:
                        if (r_final, c_final) in candidate_obus_for_slots:
                             old_candidate = candidate_obus_for_slots[(r_final, c_final)]
                             logger.debug(f"    坑位({r_final},{c_final})冲突: 新 '{obu_text_verified}'(分:{current_score:.2f}) 替换旧 '{old_candidate['text']}'(分:{old_candidate['score']:.2f})")
                        candidate_obus_for_slots[(r_final, c_final)] = {
                            'text': obu_text_verified, 'score': current_score, 'anchor_cy': anchor['cy']
                        }
                    else:
                        existing_candidate = candidate_obus_for_slots[(r_final, c_final)]
                        logger.debug(f"    OBU '{obu_text_verified}'(分:{current_score:.2f}) 竞争坑位 ({r_final},{c_final}) 失败，已有 '{existing_candidate['text']}'(分:{existing_candidate['score']:.2f})")
                else:
                    logger.debug(f"    OBU '{obu_text_verified}' 在证据池中，但未能为其找到新的最佳逻辑坑位。锚点cy: {anchor['cy']}")

            # 将最终候选者填充到矩阵
            filled_this_rebuild_count = 0
            for (r,c), obu_data in candidate_obus_for_slots.items():
                if logical_matrix[r][c] != -1 :
                    logical_matrix[r][c] = 1
                    recognized_texts_map[(r,c)] = obu_data['text']
                    # 更新obu_evidence_pool中该OBU的最终逻辑坐标 (如果需要，但可能不是必须的，因为每次都重算)
                    # obu_evidence_pool[obu_data['text']]['logical_coord'] = (r, c)
                    logger.info(f"会话 {session_id}: (重构矩阵完成) OBU '{obu_data['text']}' 最终填入逻辑坑位 ({r},{c})")
                    filled_this_rebuild_count +=1
            logger.info(f"会话 {session_id}: (重构矩阵完成) 本轮共填充 {filled_this_rebuild_count} 个OBU到矩阵。")


    logger.info(f"会话 {session_id}: (V5.3 P5d Debug) 状态矩阵和证据池更新完成。")

# --- 新的核心函数：YOLO映射与布局更新 (V5.1 - 阶段二：初步智能化) ---
def map_yolo_and_update_layout(current_yolo_boxes, session_id, logger):
    """
    V5.1 - 阶段二：初步智能化版本。
    将当前帧的YOLO检测框映射到逻辑坐标，并尝试学习/更新会话的布局参数。
    """
    logger.info(f"会话 {session_id}: (智能版V5.1 P2) 开始映射YOLO锚点并更新布局...")
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 在映射YOLO时未找到会话数据。")
        return [], False

    layout_params = session["layout_parameters"]
    yolo_anchor_map = session["yolo_anchor_map"] # 用于存储每个逻辑坑位最可信的物理锚点
    current_config = session["current_layout_config"]

    layout_updated_this_run = False
    current_frame_mapping = [] # [(original_yolo_index, (logical_row, logical_col))]

    if not current_yolo_boxes:
        logger.info(f"会话 {session_id}: 当前帧无YOLO检测框。")
        return [], False

    # 1. 对当前帧的YOLO框进行行分组 (与之前相同)
    yolo_anchors_sorted_by_y = sorted(current_yolo_boxes, key=lambda a: (a['cy'], a['cx']))
    yolo_rows_grouped_current_frame = []
    # ... (完整的行分组逻辑，得到 yolo_rows_grouped_current_frame) ...
    avg_h_yolo_for_grouping = np.mean([a['h'] for a in yolo_anchors_sorted_by_y if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted_by_y) else 30
    y_threshold_for_grouping = avg_h_yolo_for_grouping * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR
    if not yolo_anchors_sorted_by_y: logger.info(f"会话 {session_id}: 当前帧无有效YOLO锚点进行行分组。"); return [], False
    _current_row_group = [yolo_anchors_sorted_by_y[0]]
    for i in range(1, len(yolo_anchors_sorted_by_y)):
        if abs(yolo_anchors_sorted_by_y[i]['cy'] - _current_row_group[-1]['cy']) < y_threshold_for_grouping:
            _current_row_group.append(yolo_anchors_sorted_by_y[i])
        else:
            yolo_rows_grouped_current_frame.append(sorted(_current_row_group, key=lambda a: a['cx']))
            _current_row_group = [yolo_anchors_sorted_by_y[i]]
    if _current_row_group: yolo_rows_grouped_current_frame.append(sorted(_current_row_group, key=lambda a: a['cx']))

    if not yolo_rows_grouped_current_frame:
        logger.info(f"会话 {session_id}: 当前帧YOLO行分组为空。")
        return [], False
    logger.info(f"会话 {session_id}: 当前帧YOLO行分组为 {len(yolo_rows_grouped_current_frame)} 行。")


    # 2. 学习/更新布局参数 (layout_params)
    # 简化版：如果尚未校准，或当前帧提供了“更好”的行信息，则尝试校准
    # “更好”的定义：例如，检测到的物理行数更接近预期的总行数

    num_detected_physical_rows = len(yolo_rows_grouped_current_frame)
    expected_total_logical_rows = current_config["expected_total_rows"]

    # 筛选高质量的物理行作为参考 (例如，行内OBU数量 > 1)
    reference_physical_rows = [row for row in yolo_rows_grouped_current_frame if len(row) > 1]
    if not reference_physical_rows:
        logger.warning(f"会话 {session_id}: 当前帧未找到足够的高质量参考物理行 (每行至少2个锚点)。")
        # 如果已有校准参数，则使用旧的；否则无法进行映射
        if not layout_params.get("is_calibrated", False):
            return [], False
    else: # 有参考行，尝试更新布局参数
        # a. 学习平均物理行高
        learned_avg_physical_row_height = layout_params.get("avg_obu_h", avg_h_yolo_for_grouping * 1.2) # Default
        if len(reference_physical_rows) > 1:
            y_diffs = [np.mean([a['cy'] for a in reference_physical_rows[i+1]]) - np.mean([a['cy'] for a in reference_physical_rows[i]])
                       for i in range(len(reference_physical_rows)-1) if reference_physical_rows[i+1] and reference_physical_rows[i]]
            if y_diffs and np.mean(y_diffs) > 0 : learned_avg_physical_row_height = np.mean(y_diffs)

        # b. 学习平均物理列间距和起始X (从参考行中数量最多的那几行学习)
        #    这里用一个简化的方法：取所有参考行中X分布的统计值
        all_ref_cx = [a['cx'] for row in reference_physical_rows for a in row]
        all_ref_w = [a['w'] for row in reference_physical_rows for a in row if a.get('w',0)>0]

        learned_avg_start_x = min(all_ref_cx) if all_ref_cx else layout_params.get("avg_start_x", 100)
        learned_avg_col_spacing = layout_params.get("avg_col_spacing", (np.mean(all_ref_w) if all_ref_w else 100) * 1.1)

        temp_spacings = []
        for row in reference_physical_rows:
            if len(row) > 1:
                xs_in_row = sorted([a['cx'] for a in row])
                temp_spacings.extend(np.diff(xs_in_row))
        if temp_spacings: learned_avg_col_spacing = np.mean(temp_spacings)

        # c. 更新会话的布局参数 (简化：如果未校准，或当前帧行数更多，则直接使用学习到的新参数)
        if not layout_params.get("is_calibrated", False) or \
           num_detected_physical_rows > layout_params.get("last_used_num_physical_rows_for_calib", 0):

            layout_params["avg_physical_row_height"] = learned_avg_physical_row_height
            layout_params["avg_start_x"] = learned_avg_start_x
            layout_params["avg_col_spacing"] = learned_avg_col_spacing

            # 估算所有逻辑行的Y坐标
            # 假设检测到的物理行块对应于逻辑行的底部（更清晰）
            # (这是一个非常强的假设，需要后续优化为基于特殊行锚定)
            estimated_row_y = [0.0] * expected_total_logical_rows
            if reference_physical_rows:
                # 以检测到的最底部物理行为基准，向上外推
                base_physical_y = np.mean([a['cy'] for a in reference_physical_rows[-1]])
                # 假设这个最底部物理行对应逻辑上的某一行 (例如，特殊行是最后一行，则为 expected_total_logical_rows -1)
                # 简化：假设它就是 expected_total_logical_rows - len(reference_physical_rows) + (len(reference_physical_rows)-1)
                # 即，检测到的行块的最后一行，对应它在块内的索引所暗示的逻辑行（从底部数起）
                # 这部分逻辑需要非常小心，很容易出错，先用一个简单版本

                # 简化：假设检测到的行是连续的逻辑行块，这个块的底部对应图片的底部
                # 我们需要将这个块“对齐”到13个逻辑行中的某个位置
                # 暂时：将检测到的物理行，从底部开始，依次映射到逻辑行 12, 11, 10 ...
                # 这仍然不够智能，但比之前的全局平均要好一点

                # 尝试基于检测到的行的Y范围，在13个逻辑行中均匀分布
                all_detected_cy = [a['cy'] for row in reference_physical_rows for a in row]
                if all_detected_cy:
                    min_detected_y, max_detected_y = min(all_detected_cy), max(all_detected_cy)
                    if expected_total_logical_rows == 1:
                        estimated_row_y[0] = (min_detected_y + max_detected_y) / 2
                    elif max_detected_y > min_detected_y : # 确保范围有效
                        y_step_logic = (max_detected_y - min_detected_y) / (num_detected_physical_rows -1 if num_detected_physical_rows > 1 else 1)
                        # 将这个学习到的物理行高，用于扩展到所有逻辑行
                        # 以检测到的最底部行的Y为基准，向上/下扩展
                        # 假设检测到的最后一行是逻辑上的第 (expected_total_logical_rows - 1) 行 (如果它是特殊行)
                        # 或者 (expected_total_logical_rows - num_detected_physical_rows + i)
                        # 这是一个非常粗略的Y锚定，后续必须优化
                        ref_y_for_logic_map = max_detected_y
                        ref_logic_row_for_map = expected_total_logical_rows -1

                        for r_log in range(expected_total_logical_rows):
                             estimated_row_y[r_log] = ref_y_for_logic_map - (ref_logic_row_for_map - r_log) * learned_avg_physical_row_height
                    else: # 如果所有检测行Y坐标都一样（不太可能）或只有一个检测行
                        for r_log in range(expected_total_logical_rows):
                            estimated_row_y[r_log] = min_detected_y - ( (expected_total_logical_rows -1)/2 - r_log) * learned_avg_physical_row_height


            layout_params["row_y_estimates"] = estimated_row_y

            # X坐标也用学习到的参数（暂时不考虑每行不同）
            temp_col_x = [0.0] * current_config["regular_cols_count"]
            for c_log in range(current_config["regular_cols_count"]):
                temp_col_x[c_log] = learned_avg_start_x + c_log * learned_avg_col_spacing + (np.mean(all_ref_w) if all_ref_w else 100)/2 # 中心点
            layout_params["col_x_estimates_regular"] = temp_col_x # 存储常规4列的X中心

            layout_params["is_calibrated"] = True
            layout_params["last_used_num_physical_rows_for_calib"] = num_detected_physical_rows
            layout_updated_this_run = True
            logger.info(f"会话 {session_id}: 布局参数已通过当前帧学习/更新。")

    # 3. 将当前帧的YOLO锚点映射到逻辑坐标 (使用最新或已校准的布局参数)
    if not layout_params.get("is_calibrated", False):
        logger.warning(f"会话 {session_id}: 布局参数尚未校准，无法进行精确映射。")
        # 可以选择返回空映射，或者进行一次非常粗略的映射
        # 暂时返回空，依赖首次校准成功
        return [], False

    expected_logical_rows = current_config["expected_total_rows"]
    expected_regular_cols = current_config["regular_cols_count"]

    for anchor in current_yolo_boxes: # 遍历的是原始的 current_yolo_boxes
        if 'cx' not in anchor or 'cy' not in anchor: continue

        # a. 找到最近的逻辑行 (基于Y坐标)
        best_logical_row = -1
        min_y_dist = float('inf')
        if layout_params.get("row_y_estimates"):
            for r_idx, est_y in enumerate(layout_params["row_y_estimates"]):
                dist = abs(anchor['cy'] - est_y)
                if dist < min_y_dist and dist < layout_params.get("avg_physical_row_height", 50) * 0.6 : # Y距离阈值
                    min_y_dist = dist
                    best_logical_row = r_idx

        if best_logical_row == -1: continue # 无法匹配到逻辑行

        # b. 在该逻辑行内，找到最近的逻辑列 (基于X坐标)
        best_logical_col = -1
        min_x_dist = float('inf')

        # 判断此逻辑行是否为特殊行
        is_special_row = (best_logical_row == expected_logical_rows - 1) # 假设特殊行总在底部

        cols_to_consider_in_this_row = current_config["special_row_cols_count"] if is_special_row else expected_regular_cols

        # 获取该逻辑行的预估X列坐标
        # (简化：所有行暂时使用统一的 col_x_estimates_regular)
        # (未来：col_x_estimates 应该是一个二维列表或函数 layout_params["col_x_at_row_estimates"][best_logical_row][c])

        estimated_col_xs_for_this_row = []
        if layout_params.get("col_x_estimates_regular"):
            if is_special_row and current_config["special_row_cols_count"] == 2 and expected_regular_cols == 4:
                # 特殊行2列居中，使用常规列1和2的X坐标
                if len(layout_params["col_x_estimates_regular"]) == 4:
                    estimated_col_xs_for_this_row = [
                        layout_params["col_x_estimates_regular"][1],
                        layout_params["col_x_estimates_regular"][2]
                    ]
            else: # 常规行
                estimated_col_xs_for_this_row = layout_params["col_x_estimates_regular"][:cols_to_consider_in_this_row]

        if not estimated_col_xs_for_this_row: continue

        for c_idx, est_x in enumerate(estimated_col_xs_for_this_row):
            dist = abs(anchor['cx'] - est_x)
            # X距离阈值，可以用平均列间距的一半，或平均OBU宽度的一半
            x_dist_threshold = layout_params.get("avg_col_spacing", 100) * 0.6
            if dist < min_x_dist and dist < x_dist_threshold :
                min_x_dist = dist
                best_logical_col = c_idx # 这是在 cols_to_consider_in_this_row 内的索引

        if best_logical_col != -1:
            # 如果是特殊行，需要将这个在特殊行内的列索引 (0或1) 转换回全局的逻辑列索引 (1或2)
            final_logical_col = best_logical_col
            if is_special_row and current_config["special_row_cols_count"] == 2 and expected_regular_cols == 4:
                final_logical_col = best_logical_col + 1 # 0->1, 1->2

            current_frame_mapping.append( (anchor['original_index'], (best_logical_row, final_logical_col)) )

            # (可选) 更新 yolo_anchor_map
            # ...

    logger.info(f"会话 {session_id}: (智能版V5.1 P2) YOLO锚点映射完成，共映射 {len(current_frame_mapping)} 个。")
    return current_frame_mapping, layout_updated_this_run

def update_session_matrix_from_image_data(session_id, yolo_boxes_with_logical_coords, ocr_results_this_frame, logger):
    """
    根据YOLO的逻辑坐标映射和OCR结果，更新会话的状态矩阵。
    Args:
        session_id: 当前会话ID.
        yolo_boxes_with_logical_coords (list): [(original_yolo_index, (logical_row, logical_col))]
        ocr_results_this_frame (list): 当前帧的OCR结果列表.
        logger: Flask app logger.
    """
    session = session_data_store.get(session_id)
    if not session: return

    logical_matrix = session["logical_matrix"]
    recognized_texts = session["recognized_texts"]

    # 创建一个从 original_yolo_index 到 logical_coords 的快速查找字典
    yolo_idx_to_logical_map = {idx: coords for idx, coords in yolo_boxes_with_logical_coords}

    for ocr_item in ocr_results_this_frame:
        if not ocr_item: continue
        original_yolo_idx = ocr_item.get("original_index")

        if original_yolo_idx in yolo_idx_to_logical_map:
            r_log, c_log = yolo_idx_to_logical_map[original_yolo_idx]

            # 确保行列在矩阵范围内 (虽然map_yolo应该已经保证了)
            if not (0 <= r_log < len(logical_matrix) and 0 <= c_log < len(logical_matrix[0])):
                logger.warning(f"会话 {session_id}: OCR结果的逻辑坐标 ({r_log},{c_log}) 超出矩阵范围，跳过。")
                continue

            # 跳过不可用格子
            if logical_matrix[r_log][c_log] == -1: continue

            ocr_text = ocr_item.get("ocr_final_text", "")
            is_success = ocr_text.startswith("5001") and len(ocr_text) == 16 # 简易成功判断

            if is_success:
                if logical_matrix[r_log][c_log] != 1 or recognized_texts.get((r_log, c_log)) != ocr_text: # 新成功或文本变化
                    logical_matrix[r_log][c_log] = 1
                    recognized_texts[(r_log, c_log)] = ocr_text
                    logger.info(f"会话 {session_id}: 矩阵[{r_log}][{c_log}] 更新为成功: {ocr_text}")
            else: # OCR 失败
                if logical_matrix[r_log][c_log] == 0: # 只有当之前是未知时才标记为失败
                    logical_matrix[r_log][c_log] = 2
                    logger.info(f"会话 {session_id}: 矩阵[{r_log}][{c_log}] 更新为失败。")

    # (可选) 对于那些YOLO检测到但没有对应OCR结果的（可能OCR失败或被过滤）
    # 也可以在 logical_matrix 中标记为2 (如果当前是0)
    for original_yolo_idx, (r_log, c_log) in yolo_boxes_with_logical_coords:
        is_ocr_processed_for_this_yolo = any(ocr.get("original_index") == original_yolo_idx for ocr in ocr_results_this_frame if ocr)
        if not is_ocr_processed_for_this_yolo and logical_matrix[r_log][c_log] == 0:
            logical_matrix[r_log][c_log] = 2 # 标记为YOLO检测到但OCR失败/缺失
            logger.info(f"会话 {session_id}: 矩阵[{r_log}][{c_log}] 因YOLO检测到但无OCR成功结果而标记为失败。")

def setup_logging():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("Flask应用日志系统已启动。")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_box_center_and_dims(box_xyxy):
    if box_xyxy is None or len(box_xyxy) != 4: return None, None, None, None
    x1, y1, x2, y2 = box_xyxy; w = x2 - x1; h = y2 - y1
    return int(x1 + w / 2), int(y1 + h / 2), int(w), int(h)

def preprocess_onnx_for_main(img_data, target_shape_hw):
    img_h_orig, img_w_orig = img_data.shape[:2]; target_h, target_w = target_shape_hw
    ratio = min(target_w / img_w_orig, target_h / img_h_orig)
    new_w, new_h = int(img_w_orig * ratio), int(img_h_orig * ratio)
    resized_img = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0), ratio, pad_x, pad_y

def non_max_suppression_global(boxes_xyxy, scores, iou_threshold):
    if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0: return []
    x1,y1,x2,y2 = boxes_xyxy[:,0],boxes_xyxy[:,1],boxes_xyxy[:,2],boxes_xyxy[:,3]
    areas=(x2-x1+1e-6)*(y2-y1+1e-6); order=scores.argsort()[::-1]; keep=[]
    while order.size > 0:
        i = order[0]; keep.append(i); order = order[1:]
        if order.size == 0: break
        xx1=np.maximum(x1[i],x1[order]);yy1=np.maximum(y1[i],y1[order])
        xx2=np.minimum(x2[i],x2[order]);yy2=np.minimum(y2[i],y2[order])
        w=np.maximum(0.0,xx2-xx1);h=np.maximum(0.0,yy2-yy1);inter=w*h
        ovr=inter/(areas[i]+areas[order]-inter+1e-6)
        order=order[np.where(ovr<=iou_threshold)[0]]
    return keep

def postprocess_yolo_onnx_for_main(outputs_onnx, conf_threshold, iou_threshold,
                                   original_shape_hw, model_input_shape_hw,
                                   ratio_preproc, pad_x_preproc, pad_y_preproc,
                                   num_classes=1): # <--- 确保 num_classes 参数在这里
    logger = current_app.logger if current_app else logging.getLogger(__name__)
    raw_output_tensor = np.squeeze(outputs_onnx[0])
    if raw_output_tensor.ndim != 2:
        logger.error(f"错误: Main Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}")
        return []

    # YOLOv8通常输出是 [batch, num_predictions, 4_coords + 1_obj_conf + num_classes_conf]
    # 或者 [batch, 4_coords + 1_obj_conf + num_classes_conf, num_predictions] -> 转置后处理
    # 我们假设处理后是 num_predictions 行, (4+1+num_classes) 列 或类似结构
    predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor

    boxes_candidate, scores_candidate, class_ids_candidate = [], [], []

    # 确定预期的属性数量，这取决于YOLO模型的输出格式
    # 常见的YOLOv5/v8格式是 cx,cy,w,h, objectness_confidence, class1_score, class2_score, ...
    # 如果是单类别且没有明确的objectness，可能是 cx,cy,w,h, class_confidence
    expected_attributes_with_obj = 4 + 1 + num_classes
    expected_attributes_simple_conf = 4 + 1

    actual_attributes = predictions_to_iterate.shape[1]

    for i_pred, pred_data in enumerate(predictions_to_iterate):
        box_coords_raw = pred_data[:4]
        final_confidence = 0.0
        class_id = 0 # Default for single class or if class determination fails

        if actual_attributes == expected_attributes_with_obj:
            objectness = float(pred_data[4])
            class_scores_all = pred_data[5:]
            if num_classes == 1:
                final_confidence = objectness * float(class_scores_all[0])
                # class_id remains 0
            else: # Multi-class scenario
                class_id = np.argmax(class_scores_all)
                max_class_score = float(class_scores_all[class_id])
                final_confidence = objectness * max_class_score
        elif actual_attributes == expected_attributes_simple_conf and num_classes == 1:
            # This case assumes the 5th element is directly the class confidence for the single class
            final_confidence = float(pred_data[4])
            # class_id remains 0
        else:
            if i_pred == 0: # Log warning only once per call if format is unexpected
                logger.warning(
                    f"YOLO postprocess: Unexpected number of attributes per prediction ({actual_attributes}). "
                    f"Expected {expected_attributes_simple_conf} (for simple conf) or {expected_attributes_with_obj} (with objectness). "
                    f"Attempting to use attribute 4 as confidence."
                )
            # Fallback: try to interpret the 5th element as confidence if other checks fail
            if actual_attributes >= 5:
                 final_confidence = float(pred_data[4])
            else:
                 logger.error(f"YOLO postprocess: Prediction data too short ({actual_attributes} attributes) to extract confidence.")
                 continue # Skip this prediction


        if final_confidence >= float(conf_threshold):
            cx, cy, w, h = box_coords_raw
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            boxes_candidate.append([x1, y1, x2, y2])
            scores_candidate.append(final_confidence)
            class_ids_candidate.append(class_id) # Store determined class_id

    if not boxes_candidate:
        return []

    keep_indices = non_max_suppression_global(np.array(boxes_candidate), np.array(scores_candidate), iou_threshold)

    final_detections = []
    orig_h, orig_w = original_shape_hw
    for k_idx in keep_indices:
        idx = int(k_idx)
        box_model_coords = boxes_candidate[idx]
        score = scores_candidate[idx]
        class_id_val = class_ids_candidate[idx] # Use the stored class_id

        box_no_pad_x1 = box_model_coords[0] - pad_x_preproc
        box_no_pad_y1 = box_model_coords[1] - pad_y_preproc
        box_no_pad_x2 = box_model_coords[2] - pad_x_preproc
        box_no_pad_y2 = box_model_coords[3] - pad_y_preproc

        if ratio_preproc == 0: continue

        orig_x1 = box_no_pad_x1 / ratio_preproc
        orig_y1 = box_no_pad_y1 / ratio_preproc
        orig_x2 = box_no_pad_x2 / ratio_preproc
        orig_y2 = box_no_pad_y2 / ratio_preproc

        final_x1 = int(np.clip(orig_x1, 0, orig_w))
        final_y1 = int(np.clip(orig_y1, 0, orig_h))
        final_x2 = int(np.clip(orig_x2, 0, orig_w))
        final_y2 = int(np.clip(orig_y2, 0, orig_h))

        final_detections.append([final_x1, final_y1, final_x2, final_y2, score, class_id_val])

    return final_detections

def draw_detections(image, boxes, scores, class_ids, class_names=None, ocr_texts=None, roi_indices=None):
    img_out = image.copy() # Ensure all drawing happens on a copy
    # ... (rest of draw_detections from v2.5.5, ensure it uses img_out)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int); score = scores[i]; class_id = int(class_ids[i])
        label_name = class_names[class_id] if class_names and 0<=class_id<len(class_names) else f"ClassID:{class_id}"
        yolo_label_text = f"{label_name}: {score:.2f}"; cv2.rectangle(img_out,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(img_out,yolo_label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if roi_indices and i < len(roi_indices): cv2.putText(img_out,f"ROI:{roi_indices[i]}",(x1+5,y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i] != "N/A": cv2.putText(img_out,ocr_texts[i],(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    return img_out

_worker_ocr_predictor = None
def init_ocr_worker(ocr_model_dir):
    global _worker_ocr_predictor; worker_pid = os.getpid()
    print(f"[Worker PID {worker_pid}] Initializing OCR predictor: {ocr_model_dir}")
    try:
        _worker_ocr_predictor = pdx.inference.create_predictor(model_dir=ocr_model_dir, model_name='PP-OCRv5_server_rec', device='cpu')
        print(f"[Worker PID {worker_pid}] OCR predictor initialized.")
    except Exception as e: print(f"[Worker PID {worker_pid}] OCR predictor init FAILED: {e}\n{traceback.format_exc()}"); _worker_ocr_predictor = None

def ocr_task_for_worker(task_data):
    global _worker_ocr_predictor; roi_idx, _, img_bgr = task_data; pid = os.getpid(); start_t = time.time()
    if img_bgr is None: return roi_idx, {'rec_text': 'PREPROC_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t}
    if _worker_ocr_predictor is None: return roi_idx, {'rec_text': 'WORKER_INIT_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t}
    try:
        res_gen = _worker_ocr_predictor.predict([img_bgr]); res_list = next(res_gen, None); final_res = {'rec_text': '', 'rec_score': 0.0}
        if res_list and isinstance(res_list, list) and len(res_list) > 0: final_res = res_list[0]
        elif res_list and isinstance(res_list, dict): final_res = res_list
        return roi_idx, {**final_res, 'pid': pid, 'duration': time.time() - start_t}
    except Exception as e: return roi_idx, {'rec_text': 'PREDICT_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_t, 'error': str(e)}

def generate_ideal_layout_and_matrix(yolo_anchors_input, current_layout_config, image_wh, logger):
    logger.info("智能布局 V4.1: 基于可靠YOLO行构建局部确认网格 (修正变量名)...") # 版本号微调以便追踪
    if not yolo_anchors_input or len(yolo_anchors_input) < MIN_YOLO_ANCHORS_FOR_LAYOUT / 2:
        logger.warning(f"智能布局 V4.1: YOLO锚点数量 ({len(yolo_anchors_input)}) 过少。")
        return None, None

    # 1. YOLO锚点行分组
    yolo_anchors_sorted_by_y = sorted(yolo_anchors_input, key=lambda a: (a['cy'], a['cx']))
    yolo_rows_grouped = []
    avg_h_yolo_for_grouping = np.mean([a['h'] for a in yolo_anchors_sorted_by_y if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in yolo_anchors_sorted_by_y) else 30
    y_threshold_for_grouping = avg_h_yolo_for_grouping * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR
    if not yolo_anchors_sorted_by_y:
        logger.warning("智能布局 V4.1: 无有效YOLO锚点进行行分组。")
        return None, None
    current_row_for_grouping = [yolo_anchors_sorted_by_y[0]]
    for i in range(1, len(yolo_anchors_sorted_by_y)):
        if abs(yolo_anchors_sorted_by_y[i]['cy'] - current_row_for_grouping[-1]['cy']) < y_threshold_for_grouping:
            current_row_for_grouping.append(yolo_anchors_sorted_by_y[i])
        else:
            yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
            current_row_for_grouping = [yolo_anchors_sorted_by_y[i]]
    if current_row_for_grouping:
        yolo_rows_grouped.append(sorted(current_row_for_grouping, key=lambda a: a['cx']))
    logger.info(f"智能布局 V4.1: YOLO锚点行分组为 {len(yolo_rows_grouped)} 行。每行数量: {[len(r) for r in yolo_rows_grouped]}")
    if not yolo_rows_grouped:
        logger.warning("智能布局 V4.1: 行分组后结果为空。")
        return None, None

    # 2. 筛选可靠物理行
    reliable_physical_rows = [row for row in yolo_rows_grouped if len(row) >= 2]
    if not reliable_physical_rows:
        logger.warning("智能布局 V4.1: 未找到足够可靠的物理行。")
        return None, None
    logger.info(f"智能布局 V4.1: 筛选出 {len(reliable_physical_rows)} 个可靠物理行。")

    # 3. 推断常规列数和特殊行位置 (与V4相同)
    inferred_regular_cols = current_layout_config["regular_cols_count"]
    # ... (此处省略详细的 inferred_regular_cols 和 special_row_is_at_logical_top 判断逻辑，与V4版本一致)
    possible_regular_col_counts = [len(r) for r in reliable_physical_rows if len(r) != current_layout_config["special_row_cols_count"]]
    if possible_regular_col_counts:
        mode_res = Counter(possible_regular_col_counts).most_common(1)
        if mode_res and mode_res[0][0] > 0: inferred_regular_cols = mode_res[0][0]
    logger.info(f"智能布局 V4.1: 从可靠行推断常规列数为: {inferred_regular_cols}")
    physical_bottom_special_row_idx = -1
    if len(reliable_physical_rows[-1]) == current_layout_config["special_row_cols_count"]:
        if len(reliable_physical_rows) > 1 and abs(len(reliable_physical_rows[-2]) - inferred_regular_cols) <=1 :
            physical_bottom_special_row_idx = len(reliable_physical_rows) - 1
        elif len(reliable_physical_rows) == 1: physical_bottom_special_row_idx = 0
    if physical_bottom_special_row_idx != -1: logger.info(f"智能布局 V4.1: 物理底部检测到特殊行 (索引 {physical_bottom_special_row_idx} in reliable_physical_rows)。")


    # 4. 计算平均物理行高 (与V4相同)
    avg_physical_row_height = avg_h_yolo_for_grouping * 1.2
    if len(reliable_physical_rows) > 1:
        y_diffs = [reliable_physical_rows[i+1][0]['cy'] - reliable_physical_rows[i][0]['cy'] for i in range(len(reliable_physical_rows)-1)]
        if y_diffs: avg_physical_row_height = np.mean(y_diffs)
    if avg_physical_row_height <=0: avg_physical_row_height = avg_h_yolo_for_grouping * 1.2
    logger.info(f"智能布局 V4.1: 平均物理行高估算为: {avg_physical_row_height:.1f}")

    # 5. 生成理想坑位 (只为可靠物理行中的锚点生成，并赋予逻辑坐标)
    ideal_grid_slots = [] # <--- 确保初始化！！！
    # assigned_yolo_anchors_to_logical_slots = [] # 这个变量在V4中定义了但未使用，可以移除或按需使用

    expected_total_logical_rows = current_layout_config["expected_total_rows"]

    for i_physical_row, physical_row_anchors in enumerate(reliable_physical_rows):
        estimated_logical_row = -1
        if physical_bottom_special_row_idx != -1:
            logical_idx_of_bottom_special = expected_total_logical_rows - 1
            physical_row_diff_from_special = i_physical_row - physical_bottom_special_row_idx
            estimated_logical_row = logical_idx_of_bottom_special + physical_row_diff_from_special
        else:
            estimated_logical_row = (expected_total_logical_rows - len(reliable_physical_rows)) + i_physical_row
            if estimated_logical_row < 0 : estimated_logical_row = i_physical_row

        if not (0 <= estimated_logical_row < expected_total_logical_rows):
            logger.warning(f"智能布局 V4.1: 物理行 {i_physical_row} 估算的逻辑行 {estimated_logical_row} 超出范围，跳过。")
            continue

        is_this_logical_row_special = (estimated_logical_row == expected_total_logical_rows - 1)

        logical_col_indices_to_assign = []
        if is_this_logical_row_special and len(physical_row_anchors) == current_layout_config["special_row_cols_count"]:
            if inferred_regular_cols == 4 and current_layout_config["special_row_cols_count"] == 2:
                logical_col_indices_to_assign = [1, 2]
            else:
                logical_col_indices_to_assign = list(range(len(physical_row_anchors)))
        else:
            logical_col_indices_to_assign = list(range(len(physical_row_anchors)))

        for i_anchor_in_row, yolo_anchor in enumerate(physical_row_anchors):
            if i_anchor_in_row < len(logical_col_indices_to_assign):
                logical_col = logical_col_indices_to_assign[i_anchor_in_row]

                ideal_grid_slots.append({ # <--- 使用 ideal_grid_slots.append
                    'logical_row': estimated_logical_row,
                    'logical_col': logical_col,
                    'cx': int(yolo_anchor['cx']),
                    'cy': int(yolo_anchor['cy']),
                    'w': int(yolo_anchor['w']),
                    'h': int(yolo_anchor['h'])
                })
                # assigned_yolo_anchors_to_logical_slots.append({**yolo_anchor, 'logical_row': estimated_logical_row, 'logical_col': logical_col})

    if not ideal_grid_slots: # <--- 使用 ideal_grid_slots
        logger.warning("智能布局 V4.1: 未能从可靠行生成任何理想坑位。")
        return None, None
    logger.info(f"智能布局 V4.1: 基于可靠行共生成 {len(ideal_grid_slots)} 个高置信度理想坑位。") # <--- 使用 ideal_grid_slots

    # 6. 构建初始矩阵
    initial_obu_matrix = [["无检测"] * inferred_regular_cols for _ in range(expected_total_logical_rows)]
    for slot in ideal_grid_slots: # <--- 使用 ideal_grid_slots
        r, c = slot['logical_row'], slot['logical_col']
        if 0 <= r < expected_total_logical_rows and 0 <= c < inferred_regular_cols:
            initial_obu_matrix[r][c] = "未识别"
            if r == expected_total_logical_rows - 1 and \
               current_layout_config["special_row_cols_count"] == 2 and \
               inferred_regular_cols == 4:
                if c == 0 or c == 3:
                    initial_obu_matrix[r][c] = "---"

    return ideal_grid_slots, initial_obu_matrix

def fill_matrix_incrementally(matrix, slots, paddle_res, processed_texts, logger):
    logger.info("增量填充矩阵...")
    filled_new = 0
    if not paddle_res or not slots: return matrix, filled_new
    slot_coords = np.array([[s['cx'],s['cy']] for s in slots])
    paddle_coords = np.array([[p['cx'],p['cy']] for p in paddle_res if p.get('cx') is not None])
    if paddle_coords.size == 0: return matrix, filled_new
    dist_m = cdist(slot_coords, paddle_coords)
    avg_w_slot = np.mean([s['w'] for s in slots if s.get('w',0)>0]) or 100

    for j_paddle, p_obu in enumerate(paddle_res):
        if p_obu.get('cx') is None or p_obu['text'] in processed_texts: continue
        best_slot_i = -1; min_d = float('inf')
        for i_slot, s_info in enumerate(slots):
            if j_paddle < dist_m.shape[1] and i_slot < dist_m.shape[0]:
                curr_d = dist_m[i_slot, j_paddle]
                max_d_thr = PADDLE_OBU_TO_GRID_MAX_DIST_FACTOR * s_info.get('w', avg_w_slot)
                if curr_d < max_d_thr and curr_d < min_d: min_d=curr_d; best_slot_i=i_slot
        if best_slot_i != -1:
            s_fill = slots[best_slot_i]; r,c = s_fill['logical_row'],s_fill['logical_col']
            if r<len(matrix) and c<len(matrix[r]) and (matrix[r][c]=="未识别" or not matrix[r][c].startswith("5001")):
                matrix[r][c] = p_obu['text']; processed_texts.add(p_obu['text']); filled_new+=1
                logger.info(f"OBU '{p_obu['text']}' 填入矩阵 [{r},{c}]")
    logger.info(f"本轮新填充 {filled_new} 个OBU")
    return matrix, filled_new

def draw_obu_matrix_on_image(obu_matrix, layout_cfg, logger): # From v2.5.5
    # ... (Full function body from previous correct version) ...
    if not obu_matrix: logger.warning("矩阵数据为空，无法生成逻辑矩阵图。"); return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)
    num_r = layout_cfg.get("expected_total_rows", len(obu_matrix)); num_c = layout_cfg.get("regular_cols_count", len(obu_matrix[0]) if obu_matrix else 0)
    if num_r == 0 or num_c == 0: logger.warning("矩阵维度为0，无法生成逻辑矩阵图。"); return np.full((100, 400, 3), (200, 200, 200), dtype=np.uint8)
    cell_sz=60; pad=15; space=10; txt_off_y=-5
    img_w = num_c*cell_sz+(num_c-1)*space+2*pad; img_h = num_r*cell_sz+(num_r-1)*space+2*pad
    canvas = np.full((img_h, img_w, 3), (255,255,255), dtype=np.uint8)
    c_ok=(0,180,0); c_fail=(0,0,200); c_txt_ok=(255,255,255); c_txt_fail=(255,255,255); c_empty=(220,220,220)
    f_scale=0.7; f_thick=2
    for r_idx in range(num_r):
        actual_cols_r = num_c; special_cfg_cols = layout_cfg.get("special_row_cols_count", num_c)
        if r_idx < len(obu_matrix):
            actual_cols_data = len(obu_matrix[r_idx])
            if actual_cols_data != num_c and actual_cols_data == special_cfg_cols: actual_cols_r = actual_cols_data
            elif actual_cols_data != num_c : logger.warning(f"逻辑图:行{r_idx}数据列{actual_cols_data}与常规{num_c}或特殊{special_cfg_cols}不符"); actual_cols_r = actual_cols_data
        for c_idx in range(num_c):
            cx_start=pad+c_idx*(cell_sz+space); cy_start=pad+r_idx*(cell_sz+space)
            cen_x=cx_start+cell_sz//2; cen_y=cy_start+cell_sz//2
            cell_color=c_empty; disp_txt=""; txt_color=(0,0,0)
            if c_idx < actual_cols_r and r_idx < len(obu_matrix):
                txt_slot = obu_matrix[r_idx][c_idx]
                if isinstance(txt_slot, str):
                    if txt_slot=="未识别" or txt_slot.startswith("锚点不足") or txt_slot.startswith("无YOLO锚点") or txt_slot.startswith("无理想坑位"): disp_txt="X"; cell_color=c_fail; txt_color=c_txt_fail
                    elif txt_slot.startswith("5001") and len(txt_slot)>4: disp_txt=txt_slot[-4:]; cell_color=c_ok; txt_color=c_txt_ok
                    else: disp_txt=txt_slot[:4]; cell_color=c_fail; txt_color=c_txt_fail
                else: disp_txt="N/A"; cell_color=c_fail; txt_color=c_txt_fail
            else: disp_txt="-"
            cv2.rectangle(canvas,(cx_start,cy_start),(cx_start+cell_sz,cy_start+cell_sz),cell_color,-1)
            cv2.rectangle(canvas,(cx_start,cy_start),(cx_start+cell_sz,cy_start+cell_sz),(50,50,50),1)
            if disp_txt:
                (tw,th),_ = cv2.getTextSize(disp_txt,cv2.FONT_HERSHEY_SIMPLEX,f_scale,f_thick)
                tx=cen_x-tw//2; ty=cen_y+th//2+txt_off_y
                cv2.putText(canvas,disp_txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,f_scale,txt_color,f_thick,cv2.LINE_AA)
    logger.info("逻辑矩阵可视化图绘制完成。")
    return canvas

# --- [修改后的] Main Image Processing Function ---
def process_image_with_ocr_logic(image_path, current_onnx_session, session_id, current_layout_config, min_area_cfg, max_area_cfg):
    logger = current_app.logger
    logger.info(f"会话 {session_id}: 处理图片 {os.path.basename(image_path)} (V6.0.2 相对布局分析框架 - 完整集成)") # 版本号更新
    timing_profile = {}
    t_start_overall_processing = time.time()

    # 1. Read Image
    t_start_step = time.time()
    original_image = cv2.imread(image_path)
    timing_profile['1_image_reading'] = time.time() - t_start_step
    if original_image is None:
        logger.error(f"错误: 无法读取图片: {image_path}")
        empty_matrix = [[0] * current_layout_config.get("regular_cols_count", 4)
                        for _ in range(current_layout_config.get("expected_total_rows", 13))]
        return empty_matrix, {}, timing_profile, [{"message": f"无法读取图片: {os.path.basename(image_path)}"}]
    orig_img_h, orig_img_w = original_image.shape[:2]
    logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    # 2. YOLO Detection & Area Filtering (恢复您之前版本V5.x的完整逻辑)
    t_start_step = time.time() # Start timing for entire YOLO block

    actual_max_area_threshold_px = None
    if max_area_cfg is not None:
        if isinstance(max_area_cfg, float) and 0 < max_area_cfg <= 1.0:
            actual_max_area_threshold_px = (orig_img_h * orig_img_w) * max_area_cfg
        elif isinstance(max_area_cfg, (int, float)) and max_area_cfg > 1:
            actual_max_area_threshold_px = float(max_area_cfg)

    logger.info("--- 开始整图检测 (YOLO) ---")
    input_cfg = current_onnx_session.get_inputs()[0]
    input_name = input_cfg.name
    input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640) # Default
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int):
        model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]

    _t_yolo_pre = time.time()
    input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref))
    timing_profile['2a_yolo_preprocessing'] = time.time() - _t_yolo_pre

    _t_yolo_inf = time.time()
    outputs_main = current_onnx_session.run(None, {input_name: input_tensor})
    timing_profile['2b_yolo_inference'] = time.time() - _t_yolo_inf

    _t_yolo_post = time.time()
    detections_result_list = postprocess_yolo_onnx_for_main(
        outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
        original_image.shape[:2], (model_input_h_ref, model_input_w_ref),
        ratio_main, pad_x_main, pad_y_main,
        num_classes=len(COCO_CLASSES) # Ensure COCO_CLASSES is defined globally or passed
    )
    timing_profile['2c_yolo_postprocessing'] = time.time() - _t_yolo_post

    # Extract boxes, scores, class_ids from detections_result_list
    aggregated_boxes_xyxy = []
    aggregated_scores = []
    aggregated_class_ids = [] # Keep for consistency, though new layout might not use it directly
    if detections_result_list:
        aggregated_boxes_xyxy = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]
        aggregated_scores = [d[4] for d in detections_result_list]
        aggregated_class_ids = [d[5] for d in detections_result_list]

    logger.info(f"YOLO检测完成。找到了 {len(aggregated_boxes_xyxy)} 个原始框。")

    # Area Filtering
    _t_area_filter_start = time.time()
    if len(aggregated_boxes_xyxy) > 0 and ((min_area_cfg is not None and min_area_cfg > 0) or actual_max_area_threshold_px is not None):
        filtered_boxes_temp, filtered_scores_temp, filtered_ids_temp = [], [], []
        # initial_count_before_filter = len(aggregated_boxes_xyxy) # For logging if needed
        for i_box_filter, box_xyxy_filter in enumerate(aggregated_boxes_xyxy):
            b_w_filter, b_h_filter = box_xyxy_filter[2] - box_xyxy_filter[0], box_xyxy_filter[3] - box_xyxy_filter[1]
            area_filter = b_w_filter * b_h_filter
            valid_filter = True
            if min_area_cfg is not None and min_area_cfg > 0 and area_filter < min_area_cfg:
                valid_filter = False
            if actual_max_area_threshold_px is not None and area_filter > actual_max_area_threshold_px:
                valid_filter = False

            if valid_filter:
                filtered_boxes_temp.append(box_xyxy_filter)
                filtered_scores_temp.append(aggregated_scores[i_box_filter])
                if aggregated_class_ids: # Ensure it's not empty
                     filtered_ids_temp.append(aggregated_class_ids[i_box_filter])

        aggregated_boxes_xyxy = filtered_boxes_temp
        aggregated_scores = filtered_scores_temp
        aggregated_class_ids = filtered_ids_temp
        logger.info(f"面积筛选后剩 {len(aggregated_boxes_xyxy)} 个框.")
    timing_profile['2d_area_filtering'] = time.time() - _t_area_filter_start
    timing_profile['2_yolo_detection_total'] = time.time() - t_start_step # End timing for entire YOLO block

    # Prepare yolo_boxes_for_mapping_with_details (used by analyze_frame_relative_layout)
    # This list contains dicts with cx, cy, w, h, original_index etc. for *filtered* boxes
    yolo_boxes_for_mapping_with_details = []
    for i_filtered_box, box_xyxy_abs_filtered in enumerate(aggregated_boxes_xyxy):
        cx, cy, w, h = get_box_center_and_dims(box_xyxy_abs_filtered)
        if cx is not None: # Ensure get_box_center_and_dims was successful
            yolo_boxes_for_mapping_with_details.append({
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'box_yolo_xyxy': box_xyxy_abs_filtered,
                'score': aggregated_scores[i_filtered_box],
                'original_index': i_filtered_box # Index relative to the filtered list
            })

    # --- [新] 步骤 2.5: 调用 analyze_frame_relative_layout ---
    t_start_step = time.time()
    frame_layout_success, current_frame_local_layout_data, layout_status_msg = analyze_frame_relative_layout(
        yolo_boxes_for_mapping_with_details,
        (orig_img_w, orig_img_h), # Pass image dimensions
        logger,
        session_id
    )
    timing_profile['2.5_analyze_frame_relative_layout'] = time.time() - t_start_step
    logger.info(f"会话 {session_id}: 单帧相对布局分析状态: {layout_status_msg}")

    warnings_for_return = []
    if not frame_layout_success:
        logger.warning(f"会话 {session_id}: 单帧相对布局分析失败或质量不高 ({layout_status_msg})，可能影响后续处理。")
        warnings_for_return.append({"message": f"当前帧布局分析问题: {layout_status_msg}. 结果可能不完整。"})
        if current_frame_local_layout_data is None:
            current_frame_local_layout_data = [] # Ensure it's an empty list if None

    # 3. OCR Preprocessing & Task Preparation
    t_start_step = time.time()
    tasks_for_ocr = []
    ocr_input_metadata = [None] * len(yolo_boxes_for_mapping_with_details) # Length matches filtered YOLO boxes

    if yolo_boxes_for_mapping_with_details:
        logger.info(f"--- 对 {len(yolo_boxes_for_mapping_with_details)} 个YOLO框进行OCR预处理 ---")
        for i, yolo_item_for_ocr_prep in enumerate(yolo_boxes_for_mapping_with_details):
            yolo_box_coords_prep = yolo_item_for_ocr_prep['box_yolo_xyxy']

            # --- OCR ROI CROP AND PREPROCESS ---
            # (This is the detailed logic from your V5.x process_image_with_ocr_logic)
            # Ensure DIGIT_ROI_Y_OFFSET_FACTOR, etc., are defined globally or passed
            x1_y_prep, y1_y_prep, x2_y_prep, y2_y_prep = yolo_box_coords_prep
            h_y_prep, w_y_prep = y2_y_prep - y1_y_prep, x2_y_prep - x1_y_prep

            y1_d_ideal_prep = y1_y_prep + int(h_y_prep * DIGIT_ROI_Y_OFFSET_FACTOR)
            h_d_ideal_prep = int(h_y_prep * DIGIT_ROI_HEIGHT_FACTOR)
            y2_d_ideal_prep = y1_d_ideal_prep + h_d_ideal_prep

            w_d_exp_prep = int(w_y_prep * DIGIT_ROI_WIDTH_EXPAND_FACTOR)
            cx_y_img_prep = x1_y_prep + w_y_prep / 2.0

            x1_d_ideal_prep = int(cx_y_img_prep - w_d_exp_prep / 2.0)
            x2_d_ideal_prep = int(cx_y_img_prep + w_d_exp_prep / 2.0)

            y1_d_clip_prep = max(0, y1_d_ideal_prep)
            y2_d_clip_prep = min(orig_img_h, y2_d_ideal_prep)
            x1_d_clip_prep = max(0, x1_d_ideal_prep)
            x2_d_clip_prep = min(orig_img_w, x2_d_ideal_prep)

            img_for_ocr_task = None
            if x2_d_clip_prep > x1_d_clip_prep and y2_d_clip_prep > y1_d_clip_prep:
                digit_roi_prep = original_image[y1_d_clip_prep:y2_d_clip_prep, x1_d_clip_prep:x2_d_clip_prep]
                h_roi_prep, w_roi_prep = digit_roi_prep.shape[:2]
                if h_roi_prep > 0 and w_roi_prep > 0:
                    scale_prep = TARGET_OCR_INPUT_HEIGHT / h_roi_prep # Ensure TARGET_OCR_INPUT_HEIGHT is defined
                    target_w_prep = int(w_roi_prep * scale_prep)
                    if target_w_prep <= 0: target_w_prep = 1 # Avoid zero width

                    resized_roi_prep = cv2.resize(digit_roi_prep, (target_w_prep, TARGET_OCR_INPUT_HEIGHT),
                                                interpolation=cv2.INTER_CUBIC if scale_prep > 1 else cv2.INTER_AREA)
                    gray_roi_prep = cv2.cvtColor(resized_roi_prep, cv2.COLOR_BGR2GRAY)
                    _, binary_roi_prep = cv2.threshold(gray_roi_prep, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img_for_ocr_task = cv2.cvtColor(binary_roi_prep, cv2.COLOR_GRAY2BGR) # For PaddleOCR

                    if SAVE_PROCESS_PHOTOS and img_for_ocr_task is not None:
                        ocr_slice_dir_task = os.path.join(PROCESS_PHOTO_DIR, "ocr_slices", session_id)
                        if not os.path.exists(ocr_slice_dir_task): os.makedirs(ocr_slice_dir_task, exist_ok=True)
                        slice_filename_task = f"s_idx{yolo_item_for_ocr_prep['original_index']}_roi{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                        slice_output_path_task = os.path.join(ocr_slice_dir_task, slice_filename_task)
                        try:
                            cv2.imwrite(slice_output_path_task, img_for_ocr_task, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        except Exception as e_save_slice_task:
                            logger.error(f"保存OCR切片图失败 {slice_output_path_task}: {e_save_slice_task}")
            # --- END OCR ROI CROP AND PREPROCESS ---

            ocr_input_metadata[i] = {
                "original_index": yolo_item_for_ocr_prep['original_index'], # Index from yolo_boxes_for_mapping_with_details
                "bbox_yolo_abs": yolo_box_coords_prep,
                "yolo_anchor_details": {
                    'cx': yolo_item_for_ocr_prep['cx'], 'cy': yolo_item_for_ocr_prep['cy'],
                    'w': yolo_item_for_ocr_prep['w'], 'h': yolo_item_for_ocr_prep['h'],
                    'score': yolo_item_for_ocr_prep['score']
                }
            }
            # original_index, display_roi_index (i+1), image_data
            tasks_for_ocr.append((yolo_item_for_ocr_prep['original_index'], i + 1, img_for_ocr_task))
    timing_profile['3_ocr_preprocessing_prep'] = time.time() - t_start_step

    # 4. Parallel OCR Processing & Result Consolidation
    t_start_step = time.time()
    final_ocr_results_list = [None] * len(yolo_boxes_for_mapping_with_details)
    ocr_texts_for_drawing = ["N/A"] * len(yolo_boxes_for_mapping_with_details) # For drawing on annotated image

    if tasks_for_ocr:
        logger.info(f"提交 {len(tasks_for_ocr)} 个OCR任务...")
        ocr_results_indexed_from_pool = [None] * len(tasks_for_ocr) # Temp list based on task order

        # --- OCR PARALLEL/SERIAL LOGIC (from your V5.x) ---
        global ocr_processing_pool, actual_num_ocr_workers # Ensure these are accessible
        if actual_num_ocr_workers > 1 and ocr_processing_pool:
            try:
                # tasks_for_ocr contains (original_yolo_idx, roi_display_idx, image_data)
                # ocr_task_for_worker expects (task_id, display_id, image_data) and returns (task_id, result_dict)
                # We need to map results back using the original_yolo_idx (which is task_id here)

                # Create a temporary map from original_yolo_idx to its position in tasks_for_ocr
                # to correctly place results back if pool doesn't preserve order of original_yolo_idx
                # However, Pool.map should preserve order of input tasks_for_ocr.
                # The first element of task tuple is original_yolo_idx.

                pool_res_ocr = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                # pool_res_ocr is a list of (returned_original_yolo_idx, res_dict_ocr)

                # Create a dictionary to quickly find results by original_yolo_idx
                ocr_results_map_by_orig_idx = {ret_idx: res_dict for ret_idx, res_dict in pool_res_ocr}

                for i_task, task_tuple in enumerate(tasks_for_ocr):
                    original_yolo_idx_for_task = task_tuple[0]
                    ocr_results_indexed_from_pool[i_task] = ocr_results_map_by_orig_idx.get(original_yolo_idx_for_task)

            except Exception as e_map_ocr:
                logger.error(f"OCR Pool map error: {e_map_ocr}");
        else: # Serial OCR
            logger.info("OCR串行处理")
            # [...] (您的串行OCR处理逻辑，确保结果填充到 ocr_results_indexed_from_pool)
            # (确保 SERVER_REC_MODEL_DIR_CFG_CONFIG 和 pdx 导入正确)
            serial_ocr_predictor_local = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG):
                    raise FileNotFoundError(f"Serial OCR model dir not found: {SERVER_REC_MODEL_DIR_CFG_CONFIG}")
                serial_ocr_predictor_local = pdx.inference.create_predictor(
                    model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG,
                    model_name='PP-OCRv5_server_rec', device='cpu'
                )
                for i_task_s, task_data_s_tuple in enumerate(tasks_for_ocr):
                    orig_idx_s_task, _, img_data_s_task = task_data_s_tuple
                    if img_data_s_task is not None:
                        res_gen_s_task = serial_ocr_predictor_local.predict([img_data_s_task])
                        res_list_s_task = next(res_gen_s_task, None)
                        ocr_results_indexed_from_pool[i_task_s] = res_list_s_task[0] if (res_list_s_task and isinstance(res_list_s_task, list) and len(res_list_s_task) > 0) else (res_list_s_task if isinstance(res_list_s_task, dict) else {'rec_text': '', 'rec_score': 0.0})
                    else:
                        ocr_results_indexed_from_pool[i_task_s] = {'rec_text': 'PREPROC_FAIL_SERIAL', 'rec_score': 0.0}
                if serial_ocr_predictor_local: del serial_ocr_predictor_local
            except Exception as e_serial_ocr_task:
                logger.error(f"Serial OCR error: {e_serial_ocr_task}")
        # --- END OCR PARALLEL/SERIAL LOGIC ---

        # Consolidate OCR results with metadata
        for i_ocr_res, ocr_dict_from_pool in enumerate(ocr_results_indexed_from_pool):
            # i_ocr_res corresponds to the order in tasks_for_ocr and yolo_boxes_for_mapping_with_details
            full_res_item_ocr = {**(ocr_input_metadata[i_ocr_res] or {})} # Start with metadata
            if ocr_dict_from_pool and isinstance(ocr_dict_from_pool, dict):
                raw_txt_ocr = ocr_dict_from_pool.get('rec_text', "")
                score_ocr = ocr_dict_from_pool.get('rec_score', 0.0)
                if raw_txt_ocr and raw_txt_ocr not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_SERIAL']:
                    digits_ocr = "".join(re.findall(r'\d', raw_txt_ocr)) # Ensure re is imported
                    full_res_item_ocr["ocr_final_text"] = digits_ocr if digits_ocr else "N/A_NO_DIGITS"
                    ocr_texts_for_drawing[i_ocr_res] = digits_ocr if digits_ocr else "ERR"
                else:
                    full_res_item_ocr["ocr_final_text"] = raw_txt_ocr
                    ocr_texts_for_drawing[i_ocr_res] = "ERR"
                full_res_item_ocr["ocr_confidence"] = score_ocr
            else:
                full_res_item_ocr["ocr_final_text"] = "N/A_OCR_FAIL"
                ocr_texts_for_drawing[i_ocr_res] = "N/A"
                full_res_item_ocr["ocr_confidence"] = 0.0
            final_ocr_results_list[i_ocr_res] = full_res_item_ocr
    else: # No tasks for OCR
        logger.info("无任务进行OCR")
        for i_no_ocr in range(len(yolo_boxes_for_mapping_with_details)):
            final_ocr_results_list[i_no_ocr] = {
                **(ocr_input_metadata[i_no_ocr] or {}),
                "ocr_final_text": "N/A_NO_OCR_TASKS", "ocr_confidence": 0.0
            }
    timing_profile['4_ocr_processing_total'] = time.time() - t_start_step
    logger.info(f"OCR处理与结果整合完成 ({timing_profile['4_ocr_processing_total']:.3f}s)")

    # --- [新] 步骤 5: "拼图对齐与合并"逻辑 ---
    # (与上一版本我提供的框架相同，包含简化版初始帧处理逻辑)
    # [...]
    t_start_step = time.time()
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 严重错误 - 在拼图对齐前未找到会话数据。")
        empty_matrix = [[0] * current_layout_config.get("regular_cols_count", 4)
                        for _ in range(current_layout_config.get("expected_total_rows", 13))]
        return empty_matrix, {}, timing_profile, [{"message": "会话数据丢失"}]

    logical_matrix = session["logical_matrix"]
    recognized_texts_map = session["recognized_texts_map"]
    obu_evidence_pool = session["obu_evidence_pool"]
    frame_count = session.get("frame_count", 0)

    logger.info(f"会话 {session_id}: (拼图对齐V0.1) 开始处理当前帧的局部布局与OCR结果...")
    logger.info(f"  当前帧局部布局锚点数: {len(current_frame_local_layout_data)}")
    logger.info(f"  当前帧OCR结果数: {len(final_ocr_results_list)}") # Should match len(yolo_boxes_for_mapping_with_details)
    logger.info(f"  当前证据池OBU数: {len(obu_evidence_pool)}")

    map_orig_idx_to_local_layout_item = {
        item['yolo_anchor']['original_index']: item for item in current_frame_local_layout_data
    }

    current_frame_identified_obus_with_coords = []

    for ocr_result_item in final_ocr_results_list: # Iterates up to num of yolo boxes
        if not ocr_result_item: continue

        ocr_text = ocr_result_item.get("ocr_final_text", "")
        # original_index in ocr_result_item is the index within yolo_boxes_for_mapping_with_details
        original_idx_from_ocr = ocr_result_item.get("original_index")

        if ocr_text and ocr_text in VALID_OBU_CODES:
            local_layout_item = map_orig_idx_to_local_layout_item.get(original_idx_from_ocr)
            if local_layout_item:
                current_frame_identified_obus_with_coords.append({
                    'text': ocr_text,
                    'local_coord': local_layout_item['local_coord'],
                    'is_special_on_local': local_layout_item['is_special_row_anchor'],
                    'physical_anchor': ocr_result_item.get('yolo_anchor_details', {})
                })
            else:
                logger.warning(f"会话 {session_id} (拼图): OCR识别到有效OBU '{ocr_text}' (orig_idx {original_idx_from_ocr}), "
                               f"但在局部布局数据中未找到对应项。可能是布局分析未能覆盖此锚点。")

    logger.info(f"  当前帧识别并通过校验的OBU (待对齐): {len(current_frame_identified_obus_with_coords)} 个")

    # --- [占位符/简化版] 拼图逻辑 (与上一版本相同) ---
    if current_frame_identified_obus_with_coords:
        is_first_effective_frame = (len(obu_evidence_pool) == 0)
        calculated_offset_r, calculated_offset_c = None, None
        offset_calculation_successful = False

        if is_first_effective_frame:
            logger.info(f"会话 {session_id} (拼图简化版) 尝试初始帧定位...")
            special_anchors_at_top_local = [
                obu for obu in current_frame_identified_obus_with_coords
                if obu['is_special_on_local'] and obu['local_coord'][0] == 0
            ]
            if len(special_anchors_at_top_local) == 2:
                anchor_for_offset = special_anchors_at_top_local[0] # Use the first one found
                # Sort by local_c to ensure consistent anchor selection if order matters
                # special_anchors_at_top_local.sort(key=lambda x: x['local_coord'][1])
                # anchor_for_offset = special_anchors_at_top_local[0]

                local_r_anchor, local_c_anchor = anchor_for_offset['local_coord']

                global_r_target_for_special = current_layout_config["expected_total_rows"] - 1
                # Assuming the local_c for the first special anchor (when sorted) corresponds to global C=1
                global_c_target_for_special_anchor_mapped = 1

                calculated_offset_r = global_r_target_for_special - local_r_anchor
                calculated_offset_c = global_c_target_for_special_anchor_mapped - local_c_anchor
                offset_calculation_successful = True
                logger.info(f"会话 {session_id} (拼图简化版) 初始帧通过特殊行计算偏移: dR={calculated_offset_r}, dC={calculated_offset_c}")
            else:
                logger.warning(f"会话 {session_id} (拼图简化版) 初始帧未找到合适的特殊行锚点 (local_r=0, count=2) 进行定位。")
        else:
            logger.info(f"会话 {session_id} (拼图简化版) 非初始帧，应通过证据池锚定 (逻辑待实现)。")
            # TODO: Implement robust anchoring based on obu_evidence_pool
            pass

        if offset_calculation_successful:
            for obu_item_to_map in current_frame_identified_obus_with_coords:
                lr_map, lc_map = obu_item_to_map['local_coord']
                gr_map, gc_map = lr_map + calculated_offset_r, lc_map + calculated_offset_c

                if not (0 <= gr_map < len(logical_matrix) and 0 <= gc_map < len(logical_matrix[0])):
                    logger.warning(f"会话 {session_id} (拼图) OBU '{obu_item_to_map['text']}' 转换后全局坐标 ({gr_map},{gc_map}) 超出范围，跳过。")
                    continue
                if logical_matrix[gr_map][gc_map] == -1:
                    logger.warning(f"会话 {session_id} (拼图) OBU '{obu_item_to_map['text']}' 尝试映射到不可用坑位 ({gr_map},{gc_map})，跳过。")
                    continue

                # Simplified update logic
                if logical_matrix[gr_map][gc_map] != 1 or recognized_texts_map.get((gr_map,gc_map)) != obu_item_to_map['text']:
                    logical_matrix[gr_map][gc_map] = 1
                    recognized_texts_map[(gr_map, gc_map)] = obu_item_to_map['text']
                    logger.info(f"会话 {session_id} (拼图简化版) OBU '{obu_item_to_map['text']}' 填入全局逻辑坑位 ({gr_map},{gc_map})")

                    obu_evidence_pool[obu_item_to_map['text']] = {
                        'global_logical_coord': (gr_map, gc_map),
                        'latest_physical_anchor': obu_item_to_map['physical_anchor'],
                        'last_seen_frame': frame_count
                    }
        else:
            logger.warning(f"会话 {session_id} (拼图简化版) 未能计算出有效的逻辑偏移，本帧结果可能未合并。")
            if not any(w['message'].startswith("当前帧未能可靠对齐") for w in warnings_for_return): # Avoid duplicate
                 warnings_for_return.append({"message": "当前帧未能可靠对齐到全局布局，结果可能不完整。"})
    # --- END "拼图对齐与合并"逻辑 ---
    timing_profile['5_puzzle_alignment_and_merge'] = time.time() - t_start_step
    logger.info(f"拼图对齐与合并完成 ({timing_profile['5_puzzle_alignment_and_merge']:.3f}s)")

    # 6. (可选) 保存YOLO标注图
    if SAVE_PROCESS_PHOTOS and yolo_boxes_for_mapping_with_details:
        t_start_draw = time.time()
        image_to_draw_on_final = original_image.copy()

        # Prepare boxes_xyxy, scores, class_ids for draw_detections
        # These should correspond to yolo_boxes_for_mapping_with_details
        draw_boxes = [item['box_yolo_xyxy'] for item in yolo_boxes_for_mapping_with_details]
        draw_scores = [item['score'] for item in yolo_boxes_for_mapping_with_details]
        # Assuming single class for drawing, or if aggregated_class_ids was correctly populated for filtered boxes
        draw_class_ids = [0] * len(yolo_boxes_for_mapping_with_details) # Placeholder if class_ids not critical for this drawing

        # roi_indices for drawing can be based on the original_index from yolo_boxes_for_mapping_with_details
        draw_roi_indices = [item['original_index'] for item in yolo_boxes_for_mapping_with_details]

        annotated_img_final = draw_detections(
            image_to_draw_on_final,
            np.array(draw_boxes) if draw_boxes else np.array([]), # Handle empty case
            np.array(draw_scores) if draw_scores else np.array([]),
            np.array(draw_class_ids),
            COCO_CLASSES,
            ocr_texts=ocr_texts_for_drawing, # This should be aligned with yolo_boxes_for_mapping_with_details
            roi_indices=draw_roi_indices
        )
        img_name_base_final = os.path.splitext(os.path.basename(image_path))[0]
        ts_filename_final = datetime.now().strftime("%Y%m%d%H%M%S%f")
        annotated_path_final = os.path.join(PROCESS_PHOTO_DIR, f"annotated_{img_name_base_final}_{ts_filename_final}.jpg")
        try:
            cv2.imwrite(annotated_path_final, annotated_img_final, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY])
            logger.info(f"YOLO标注图已保存: {annotated_path_final}")
        except Exception as e_save_ann_final:
            logger.error(f"保存YOLO标注图失败: {e_save_ann_final}")
        timing_profile['6_drawing_yolo_annotations'] = time.time() - t_start_draw # Renamed key for clarity

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    # ... (日志打印 timing_profile) ...
    # ... (准备并返回 final_matrix_to_return, final_texts_to_return, etc.) ...
    logger.info(f"--- Timing profile for {os.path.basename(image_path)} ({session_id}) ---")
    for stage_key_log in sorted(timing_profile.keys()):
        logger.info(f"  {stage_key_log}: {timing_profile[stage_key_log]:.3f}s")

    final_session_state_ret = session_data_store.get(session_id)
    if not final_session_state_ret:
        logger.error(f"会话 {session_id}: 在 process_image_with_ocr_logic 末尾无法获取会话状态！")
        empty_matrix_ret = [[0] * current_layout_config.get("regular_cols_count", 4)
                        for _ in range(current_layout_config.get("expected_total_rows", 13))]
        return empty_matrix_ret, {}, timing_profile, [{"message": "Error: Session data lost before returning."}]

    final_matrix_to_return_val = final_session_state_ret.get("logical_matrix", [])
    final_texts_to_return_val = final_session_state_ret.get("recognized_texts_map", {})

    if layout_status_msg and not frame_layout_success :
         if not any(w['message'].startswith("布局分析提示:") for w in warnings_for_return): # Avoid duplicate
            warnings_for_return.append({"message": f"布局分析提示: {layout_status_msg}"})

    return final_matrix_to_return_val, final_texts_to_return_val, timing_profile, warnings_for_return

# --- Flask Routes (V5.1 P5: 支持强制重新校准) ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger

    if 'session_id' not in request.form:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400
    session_id = request.form.get('session_id')

    # 新增：接收 force_recalibrate 参数
    force_recalibrate_str = request.form.get('force_recalibrate', 'false').lower()
    force_recalibrate = (force_recalibrate_str == 'true')

    current_layout_config_for_session = LAYOUT_CONFIG

    if 'file' not in request.files: # ... (文件检查逻辑与之前相同)
        logger.warning(f"会话 {session_id}: 请求中未找到文件部分。")
        return jsonify({"error": "No file part in the request", "session_id": session_id}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning(f"会话 {session_id}: 未选择文件。")
        return jsonify({"error": "No selected file", "session_id": session_id}), 400
    if not (file and allowed_file(file.filename)):
        logger.warning(f"会话 {session_id}: 文件类型不允许: {file.filename}")
        return jsonify({"error": "File type not allowed", "session_id": session_id}), 400

    original_filename_for_exc = "N/A"
    try:
        original_filename = secure_filename(file.filename); original_filename_for_exc = original_filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f"); name, ext = os.path.splitext(original_filename); filename = f"{name}_{timestamp}{ext}"
        upload_dir = current_app.config['UPLOAD_FOLDER'];
        if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename); file.save(filepath)
        logger.info(f"会话 {session_id}: 文件 '{filename}' 已成功保存到 '{filepath}'")

        global onnx_session
        if onnx_session is None: # ... (ONNX session检查)
            logger.error(f"会话 {session_id}: ONNX session 未初始化!")
            return jsonify({"error": "ONNX session not initialized on server", "session_id": session_id}), 500

        # 获取或初始化会话数据
        session = session_data_store.get(session_id)
        if not session or force_recalibrate: # 如果是新会话，或者用户强制重新校准
            if force_recalibrate and session:
                 logger.info(f"会话 {session_id}: 用户触发强制重新校准布局。")
            else:
                 logger.info(f"会话 {session_id}: 新建会话或首次校准。")

            initial_matrix = [[0] * current_layout_config_for_session["regular_cols_count"]
                              for _ in range(current_layout_config_for_session["expected_total_rows"])]
            special_row_idx = current_layout_config_for_session["expected_total_rows"] - 1
            if 0 <= special_row_idx < len(initial_matrix) and \
               current_layout_config_for_session["regular_cols_count"] == 4 and \
               current_layout_config_for_session["special_row_cols_count"] == 2:
                initial_matrix[special_row_idx][0] = -1
                initial_matrix[special_row_idx][3] = -1

            session_data_store[session_id] = {
            "logical_matrix": initial_matrix,
            "recognized_texts_map": {},
            "obu_evidence_pool": {},
            "layout_parameters": {"is_calibrated": False},
            "current_layout_config": current_layout_config_for_session,
            "frame_count": 0,
            "last_activity": datetime.now(),
            "frames_since_last_recalibration": 0,
            "accumulated_obu_widths": [], # 新增：用于累积OBU宽度
            "accumulated_obu_heights": [], # 新增：用于累积OBU高度
            "stable_avg_obu_w": None,     # 新增：稳定后的平均宽度
            "stable_avg_obu_h": None      # 新增：稳定后的平均高度
        }

        session = session_data_store.get(session_id) # 获取最新的session引用
        session["frame_count"] = session.get("frame_count", 0) + 1
        session["last_activity"] = datetime.now()

        min_area_cfg_val = current_app.config.get('MIN_DETECTION_AREA_CFG', 2000)
        max_area_cfg_val = current_app.config.get('MAX_DETECTION_AREA_CFG', 0.1)

        logical_matrix_result, recognized_texts_result, timings, warnings_from_processing = process_image_with_ocr_logic(
            filepath, onnx_session, session_id,
            current_layout_config_for_session,
            min_area_cfg_val, max_area_cfg_val
        )

        response_data = {
            "message": "File processed successfully.", "session_id": session_id,
            "received_filename": original_filename,
            "obu_status_matrix": logical_matrix_result,
            "obu_texts": {f"{r}_{c}": text for (r,c), text in recognized_texts_result.items()},
            "timing_profile_seconds": timings,
            "warnings": warnings_from_processing # 新增警告信息
        }
        # ... (session_status 判断逻辑) ...
        num_identified_successfully = sum(1 for r in logical_matrix_result for status in r if status == 1)
        total_expected_obus = current_layout_config_for_session.get("total_obus", 50)
        response_data["session_status"] = "completed" if num_identified_successfully >= total_expected_obus else "in_progress"
        logger.info(f"会话 {session_id}: 已识别 {num_identified_successfully}/{total_expected_obus} 个OBU。状态: {response_data['session_status']}")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_exc}' 时发生严重错误: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

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