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

# --- 新的核心函数：YOLO映射与布局更新 (V5.1 - 阶段四：首次校准增强) ---
def refine_layout_and_map_yolo_to_logical(current_yolo_boxes_with_orig_idx, session_id, image_wh, logger):
    logger.info(f"会话 {session_id}: (V5.3 P6 MODIFIED) 开始智能映射YOLO并校准布局...") # MODIFIED: 版本号更新
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 严重错误 - 在映射YOLO时未找到会话数据。")
        return {}, False

    layout_params = session["layout_parameters"]
    current_config = session["current_layout_config"]
    expected_total_logical_rows = current_config["expected_total_rows"]
    expected_regular_cols = current_config["regular_cols_count"] # NEW: 获取配置的常规列数

    layout_updated_this_run = False
    current_frame_mapping = {} # Store mapping from original_index to (logical_row, logical_col)

    if not current_yolo_boxes_with_orig_idx:
        logger.info(f"会话 {session_id}: 当前帧无YOLO检测框。")
        return {}, False

    # 1. 对当前帧的YOLO框进行行分组 (逻辑与之前基本一致)
    yolo_anchors_sorted_by_y = sorted(current_yolo_boxes_with_orig_idx, key=lambda a: (a['cy'], a['cx']))
    yolo_rows_grouped_current_frame = []
    # MODIFIED: 使用 current_yolo_boxes_with_orig_idx 替代 yolo_anchors_sorted_by_y 进行均值计算，避免在空列表上操作
    avg_h_yolo_for_grouping = np.mean([a['h'] for a in current_yolo_boxes_with_orig_idx if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in current_yolo_boxes_with_orig_idx) else 30
    y_threshold_for_grouping = avg_h_yolo_for_grouping * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR

    if not yolo_anchors_sorted_by_y: # Redundant check, but safe
        logger.info(f"会话 {session_id}: 当前帧无有效YOLO锚点进行行分组。")
        return {}, False

    _current_row_group = [yolo_anchors_sorted_by_y[0]]
    for i in range(1, len(yolo_anchors_sorted_by_y)):
        if abs(yolo_anchors_sorted_by_y[i]['cy'] - _current_row_group[-1]['cy']) < y_threshold_for_grouping:
            _current_row_group.append(yolo_anchors_sorted_by_y[i])
        else:
            yolo_rows_grouped_current_frame.append(sorted(_current_row_group, key=lambda a: a['cx']))
            _current_row_group = [yolo_anchors_sorted_by_y[i]]
    if _current_row_group:
        yolo_rows_grouped_current_frame.append(sorted(_current_row_group, key=lambda a: a['cx']))

    if not yolo_rows_grouped_current_frame:
        logger.info(f"会话 {session_id}: 当前帧YOLO行分组为空。")
        return {}, False
    logger.info(f"会话 {session_id}: 当前帧YOLO行分组为 {len(yolo_rows_grouped_current_frame)} 行。每行数量: {[len(r) for r in yolo_rows_grouped_current_frame]}")

    # 2. 布局参数的初始化/校准 (仅在首次校准时或强制校准时)
    if not layout_params.get("is_calibrated", False): # 或者可以加入一个 force_recalibrate 标志
        logger.info(f"会话 {session_id}: 首次布局参数校准...")

        # a. 筛选高质量参考物理行 (至少包含2个OBU)
        reliable_physical_rows = [row for row in yolo_rows_grouped_current_frame if len(row) >= 2]
        if len(reliable_physical_rows) < 2 :
            logger.warning(f"会话 {session_id}: 可靠物理行数量 ({len(reliable_physical_rows)}) 不足2行，无法进行有效校准。")
            return {}, False

        # b. 智能判断特殊行位置 (逻辑与之前类似)
        num_detected_reliable_rows = len(reliable_physical_rows)

        # MODIFIED: 推断从YOLO锚点观察到的常规列数
        inferred_regular_cols_from_yolo = expected_regular_cols # Default to config
        possible_reg_cols_counts = [len(r) for r in reliable_physical_rows if len(r) != current_config["special_row_cols_count"]]
        if possible_reg_cols_counts:
            mode_res = Counter(possible_reg_cols_counts).most_common(1)
            if mode_res and mode_res[0][0] > 0:
                inferred_regular_cols_from_yolo = mode_res[0][0]
        logger.info(f"  从YOLO锚点推断出的常规列数 (inferred_regular_cols_from_yolo): {inferred_regular_cols_from_yolo}")
        layout_params["inferred_regular_cols_from_yolo"] = inferred_regular_cols_from_yolo # NEW: Store this

        special_row_at_bottom_of_detected_block = False
        if len(reliable_physical_rows[-1]) == current_config["special_row_cols_count"] and \
           (num_detected_reliable_rows == 1 or abs(len(reliable_physical_rows[-2]) - inferred_regular_cols_from_yolo) <=1): # MODIFIED: use inferred_regular_cols_from_yolo
            special_row_at_bottom_of_detected_block = True
            logger.info(f"  检测到物理底部为特殊行 (2列)。")

        special_row_at_top_of_detected_block = False
        if len(reliable_physical_rows[0]) == current_config["special_row_cols_count"] and \
           (num_detected_reliable_rows == 1 or abs(len(reliable_physical_rows[1]) - inferred_regular_cols_from_yolo) <=1): # MODIFIED: use inferred_regular_cols_from_yolo
            special_row_at_top_of_detected_block = True
            logger.info(f"  检测到物理顶部为特殊行 (2列)。")

        if special_row_at_top_of_detected_block and not special_row_at_bottom_of_detected_block:
            layout_params["special_row_at_logical_top"] = True
        else:
            layout_params["special_row_at_logical_top"] = False
        logger.info(f"  判断特殊行在逻辑顶部: {layout_params['special_row_at_logical_top']}")

        # c. 学习平均物理行高 (逻辑与之前类似)
        physical_row_centers_y = [np.mean([a['cy'] for a in row]) for row in reliable_physical_rows]
        learned_avg_physical_row_height = avg_h_yolo_for_grouping * 1.2
        if len(physical_row_centers_y) > 1:
            phys_row_heights = np.abs(np.diff(physical_row_centers_y))
            if phys_row_heights.size > 0 and np.mean(phys_row_heights) > 0 :
                learned_avg_physical_row_height = np.mean(phys_row_heights)
        layout_params["avg_physical_row_height"] = learned_avg_physical_row_height
        logger.info(f"  学习到平均物理行高: {learned_avg_physical_row_height:.1f}")

        # d. 估算所有13个逻辑行的物理Y坐标 (逻辑与之前类似)
        layout_params["row_y_estimates"] = [0.0] * expected_total_logical_rows
        anchor_physical_y = 0
        anchor_logical_row = 0
        if layout_params["special_row_at_logical_top"]:
            anchor_physical_y = physical_row_centers_y[0]
            anchor_logical_row = 0
        else:
            anchor_physical_y = physical_row_centers_y[-1]
            anchor_logical_row = expected_total_logical_rows - 1
        for r_log in range(expected_total_logical_rows):
            layout_params["row_y_estimates"][r_log] = anchor_physical_y + \
                (r_log - anchor_logical_row) * learned_avg_physical_row_height
        logger.info(f"  估算逻辑行Y坐标: {[int(y) for y in layout_params['row_y_estimates']]}")

        # e. MODIFIED: 学习列X参数, 确保为所有 expected_regular_cols (4列) 提供估算
        layout_params["col_x_estimates_regular"] = [0.0] * expected_regular_cols # Initialize with expected length

        # MODIFIED: 使用 current_yolo_boxes_with_orig_idx 替代 yolo_anchors_input
        all_current_frame_cxs = [a['cx'] for a in current_yolo_boxes_with_orig_idx if 'cx' in a]
        avg_obu_w_for_x_est = np.mean([a['w'] for a in current_yolo_boxes_with_orig_idx if a.get('w',0)>0]) if any(a.get('w',0)>0 for a in current_yolo_boxes_with_orig_idx) else 100

        learned_col_xs_from_yolo = {} # Store {logical_col_index: learned_x_value}

        # Try to learn from rows that match inferred_regular_cols_from_yolo
        candidate_rows_for_x_learning = [r for r in reliable_physical_rows if len(r) == inferred_regular_cols_from_yolo and inferred_regular_cols_from_yolo > 0]
        if not candidate_rows_for_x_learning and inferred_regular_cols_from_yolo > 0: # Fallback: use rows that are "close"
             candidate_rows_for_x_learning = [r for r in reliable_physical_rows if abs(len(r) - inferred_regular_cols_from_yolo) <= 1 and len(r) > 0]

        if candidate_rows_for_x_learning:
            temp_col_xs_accumulator = [[] for _ in range(inferred_regular_cols_from_yolo)]
            for row in candidate_rows_for_x_learning:
                # Assume anchors in these rows correspond to the first 'inferred_regular_cols_from_yolo' logical columns
                for col_idx_in_row, anchor in enumerate(row):
                    if col_idx_in_row < inferred_regular_cols_from_yolo:
                        temp_col_xs_accumulator[col_idx_in_row].append(anchor['cx'])

            for c_log_yolo in range(inferred_regular_cols_from_yolo):
                if temp_col_xs_accumulator[c_log_yolo]:
                    learned_col_xs_from_yolo[c_log_yolo] = np.mean(temp_col_xs_accumulator[c_log_yolo])

        # Fill layout_params["col_x_estimates_regular"] (length is expected_regular_cols)
        # First, use directly learned values if available (up to inferred_regular_cols_from_yolo)
        for c_idx in range(expected_regular_cols):
            if c_idx in learned_col_xs_from_yolo:
                layout_params["col_x_estimates_regular"][c_idx] = learned_col_xs_from_yolo[c_idx]
            else:
                # This column's X was not directly learned from YOLO, mark for fallback
                layout_params["col_x_estimates_regular"][c_idx] = None # Placeholder for fallback

        # Fallback for columns not learned from YOLO
        # Strategy: If some columns were learned, try to extrapolate.
        # If no columns were learned, or extrapolation is hard, distribute in a sensible range.

        num_learned_direct = len(learned_col_xs_from_yolo)

        if num_learned_direct < expected_regular_cols: # Need fallback for some/all columns
            logger.info(f"  需要为 {expected_regular_cols - num_learned_direct} 个常规列进行X坐标Fallback估算。")

            fallback_x_coords = [None] * expected_regular_cols
            # Copy over already learned X coords to the fallback list
            for c_idx in range(expected_regular_cols):
                if layout_params["col_x_estimates_regular"][c_idx] is not None:
                    fallback_x_coords[c_idx] = layout_params["col_x_estimates_regular"][c_idx]

            # Define a region for distributing columns if needed
            min_cx_overall = min(all_current_frame_cxs) if all_current_frame_cxs else image_wh[0] * 0.1
            max_cx_overall = max(all_current_frame_cxs) if all_current_frame_cxs else image_wh[0] * 0.9
            if not all_current_frame_cxs or min_cx_overall >= max_cx_overall - avg_obu_w_for_x_est: # Not enough spread or no anchors
                min_cx_overall = image_wh[0] * 0.15 # Default region: 15% to 85% of image width
                max_cx_overall = image_wh[0] * 0.85

            distribute_region_width = max_cx_overall - min_cx_overall

            # If only one column was learned, center it and distribute others around it
            if num_learned_direct == 1:
                learned_c_idx = list(learned_col_xs_from_yolo.keys())[0]
                learned_x_val = list(learned_col_xs_from_yolo.values())[0]
                # Assume this learned column is roughly at its logical position within the 4 cols
                # e.g. if learned_c_idx is 0, it's the first. If 1, it's the second.
                # Estimate effective column spacing based on avg_obu_w
                effective_col_spacing = avg_obu_w_for_x_est * 1.1 # OBU width + small gap
                for c_target in range(expected_regular_cols):
                    if fallback_x_coords[c_target] is None:
                         fallback_x_coords[c_target] = learned_x_val + (c_target - learned_c_idx) * effective_col_spacing

            # If 2 or more (but < expected_regular_cols) columns were learned, try to use their spacing
            elif num_learned_direct >= 2 and num_learned_direct < expected_regular_cols:
                sorted_learned_indices = sorted(learned_col_xs_from_yolo.keys())
                learned_spacings = []
                for i in range(len(sorted_learned_indices) - 1):
                    idx1, idx2 = sorted_learned_indices[i], sorted_learned_indices[i+1]
                    spacing = (learned_col_xs_from_yolo[idx2] - learned_col_xs_from_yolo[idx1]) / (idx2 - idx1)
                    learned_spacings.append(spacing)
                avg_learned_spacing = np.mean(learned_spacings) if learned_spacings else avg_obu_w_for_x_est * 1.1

                # Fill gaps or extrapolate
                # Find first and last learned column to anchor extrapolation/interpolation
                first_learned_idx = min(sorted_learned_indices)
                last_learned_idx = max(sorted_learned_indices)

                for c_target in range(expected_regular_cols):
                    if fallback_x_coords[c_target] is None:
                        if c_target < first_learned_idx: # Extrapolate left
                            fallback_x_coords[c_target] = learned_col_xs_from_yolo[first_learned_idx] - (first_learned_idx - c_target) * avg_learned_spacing
                        elif c_target > last_learned_idx: # Extrapolate right
                            fallback_x_coords[c_target] = learned_col_xs_from_yolo[last_learned_idx] + (c_target - last_learned_idx) * avg_learned_spacing
                        else: # Interpolate between two learned columns
                            # Find closest learned columns bracketing c_target
                            prev_l_idx = max([l_idx for l_idx in sorted_learned_indices if l_idx < c_target])
                            next_l_idx = min([l_idx for l_idx in sorted_learned_indices if l_idx > c_target])
                            # Simple linear interpolation
                            ratio = (c_target - prev_l_idx) / (next_l_idx - prev_l_idx)
                            fallback_x_coords[c_target] = learned_col_xs_from_yolo[prev_l_idx] + ratio * (learned_col_xs_from_yolo[next_l_idx] - learned_col_xs_from_yolo[prev_l_idx])

            # If still Nones (e.g. num_learned_direct == 0, or complex gaps not filled by above)
            # Fallback to even distribution in the defined region
            if any(x is None for x in fallback_x_coords):
                logger.info(f"  部分或全部常规列X坐标使用基于区域 '{min_cx_overall:.0f}-{max_cx_overall:.0f}' 的均匀分布Fallback。")
                # Calculate spacing for expected_regular_cols within the distribute_region_width
                # Ensure at least avg_obu_w_for_x_est per column if region is too small
                effective_width_per_col = distribute_region_width / expected_regular_cols
                if effective_width_per_col < avg_obu_w_for_x_est * 0.8 : # If calculated spacing is too tight
                    # Recenter the distribution if possible, or just use image center for very narrow cases
                    logger.warning(f"  Fallback分布区域宽度 ({distribute_region_width:.0f}) 对于 {expected_regular_cols} 列可能过窄。")
                    # Try to use a wider default region if current YOLO anchors are too clustered
                    min_cx_overall = image_wh[0] * 0.15
                    max_cx_overall = image_wh[0] * 0.85
                    distribute_region_width = max_cx_overall - min_cx_overall
                    effective_width_per_col = distribute_region_width / expected_regular_cols


                for c_target in range(expected_regular_cols):
                    if fallback_x_coords[c_target] is None: # Only fill if not already set by a prior fallback
                        fallback_x_coords[c_target] = min_cx_overall + (c_target + 0.5) * effective_width_per_col

            layout_params["col_x_estimates_regular"] = fallback_x_coords

        logger.info(f"  最终估算常规列X坐标 (长度 {len(layout_params['col_x_estimates_regular'])}): {[int(x) if x is not None else 'None' for x in layout_params['col_x_estimates_regular']]}")

        # MODIFIED: 使用 current_yolo_boxes_with_orig_idx
        layout_params["avg_obu_w"] = avg_obu_w_for_x_est
        layout_params["avg_obu_h"] = np.mean([a['h'] for a in current_yolo_boxes_with_orig_idx if a.get('h',0)>0]) if any(a.get('h',0)>0 for a in current_yolo_boxes_with_orig_idx) else 40

        # Check if all col_x_estimates are valid numbers
        if any(x is None for x in layout_params["col_x_estimates_regular"]):
            logger.error(f"会话 {session_id}: 严重错误 - 首次校准后，col_x_estimates_regular 包含 None 值。校准失败。")
            layout_params["is_calibrated"] = False # Mark as not calibrated
            return {}, False # Calibration failed

        layout_params["is_calibrated"] = True
        layout_updated_this_run = True
        session["layout_parameters"] = layout_params

    # --- 后续帧的参数微调/纠错逻辑 (占位符) ---
    # else:
    #    logger.info(f"会话 {session_id}: 使用已校准的布局参数。")
    #    pass

    # 3. 将当前帧的YOLO锚点映射到逻辑坐标 (使用最新或已校准的布局参数)
    if not layout_params.get("is_calibrated", False):
        logger.warning(f"会话 {session_id}: 布局参数仍未校准，无法进行精确映射。")
        return {}, False # Return empty map and False for layout_updated

    row_ys_est = layout_params["row_y_estimates"]
    col_xs_est_reg = layout_params["col_x_estimates_regular"]

    if not row_ys_est or not col_xs_est_reg or len(col_xs_est_reg) != expected_regular_cols: # NEW: Check col_xs_est_reg length
        logger.error(f"会话 {session_id}: 映射时布局参数无效 (行Y为空, 或列X为空/长度不为{expected_regular_cols})。")
        return {}, layout_updated_this_run # Return current map (empty) and layout_updated status

    y_match_threshold = layout_params.get("avg_physical_row_height", 50) * 0.7
    # MODIFIED: Use avg_obu_w for x_match_threshold as avg_col_spacing might not be explicitly learned
    x_match_threshold = layout_params.get("avg_obu_w", 100) * 0.7

    for anchor in current_yolo_boxes_with_orig_idx:
        if 'cx' not in anchor or 'cy' not in anchor: continue

        # a. 找到最近的逻辑行
        cand_r = -1; min_y_d_sq = float('inf') # Use squared distance for y
        for r_idx, est_y in enumerate(row_ys_est):
            dist_y_sq = (anchor['cy'] - est_y)**2 # Squared distance
            if dist_y_sq < min_y_d_sq and dist_y_sq < y_match_threshold**2 :
                min_y_d_sq = dist_y_sq; cand_r = r_idx
        if cand_r == -1: continue

        # b. 在该逻辑行内，找到最近的逻辑列
        is_special_row_map = (cand_r == (expected_total_logical_rows - 1) and not layout_params.get("special_row_at_logical_top", False)) or \
                             (cand_r == 0 and layout_params.get("special_row_at_logical_top", False))

        # MODIFIED: Use expected_regular_cols (from config) instead of inferred_regular_cols
        cols_in_this_logical_row_options = current_config["special_row_cols_count"] if is_special_row_map else expected_regular_cols

        current_row_col_xs_to_match = []
        # MODIFIED: Use expected_regular_cols for condition
        if is_special_row_map and current_config["special_row_cols_count"] == 2 and expected_regular_cols == 4:
            if len(col_xs_est_reg) == 4: # Should always be true now after calib
                current_row_col_xs_to_match = [col_xs_est_reg[1], col_xs_est_reg[2]] # Special row uses logical cols 1 and 2
            else: # Should not happen if calibration is correct
                logger.warning(f"会话 {session_id}: 映射特殊行时，col_xs_est_reg 长度不为4。")
                continue
        else: # Regular row
            if len(col_xs_est_reg) == expected_regular_cols:
                 current_row_col_xs_to_match = col_xs_est_reg[:cols_in_this_logical_row_options] # For regular row, this is all 4 cols
            else: # Should not happen
                logger.warning(f"会话 {session_id}: 映射常规行时，col_xs_est_reg 长度不为{expected_regular_cols}。")
                continue

        if not current_row_col_xs_to_match: continue

        cand_c_in_row_options_idx = -1; min_x_d_sq = float('inf') # Use squared distance for x
        for c_idx_in_options, est_x in enumerate(current_row_col_xs_to_match):
            dist_x_sq = (anchor['cx'] - est_x)**2 # Squared distance
            if dist_x_sq < min_x_d_sq and dist_x_sq < x_match_threshold**2 :
                min_x_d_sq = dist_x_sq; cand_c_in_row_options_idx = c_idx_in_options

        if cand_c_in_row_options_idx != -1:
            best_r_final = cand_r
            best_c_final = -1

            # Convert index from current_row_col_xs_to_match back to global logical column index
            if is_special_row_map and current_config["special_row_cols_count"] == 2 and expected_regular_cols == 4:
                best_c_final = cand_c_in_row_options_idx + 1 # 0th option (col_xs_est_reg[1]) -> logical col 1; 1st option (col_xs_est_reg[2]) -> logical col 2
            else: # Regular row
                best_c_final = cand_c_in_row_options_idx # Index in options is directly the logical column index

            if 0 <= best_c_final < expected_regular_cols: # Final check for column index validity
                 current_frame_mapping[anchor['original_index']] = (best_r_final, best_c_final)
            else:
                 logger.warning(f"会话 {session_id}: 锚点 {anchor['original_index']} 映射后得到无效逻辑列 {best_c_final} (行内选项索引 {cand_c_in_row_options_idx})。")

    logger.info(f"会话 {session_id}: (V5.3 P6 MODIFIED) YOLO锚点映射完成，共映射 {len(current_frame_mapping)} 个。")
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
    layout_params = session["layout_parameters"]
    current_config = session["current_layout_config"]
    frame_count = session.get("frame_count", 0)

    # 1. 将当前帧成功校验的OCR结果更新到 obu_evidence_pool (与V5.3 P5c逻辑基本一致)
    newly_added_to_pool_this_frame = [] # 记录本轮新加入或更新的OBU文本
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
            obu_evidence_pool[ocr_text]['physical_anchors'] = [current_anchor_data] # 总是用最新的锚点
            obu_evidence_pool[ocr_text]['ocr_confidence'] = max(obu_evidence_pool[ocr_text]['ocr_confidence'], ocr_confidence)
            obu_evidence_pool[ocr_text]['last_seen_frame'] = frame_count
            logger.debug(f"会话 {session_id}: OBU '{ocr_text}' 信息已更新/加入到证据池。锚点: {current_anchor_data}")

    logger.info(f"会话 {session_id}: 本轮新增/更新 {len(newly_added_to_pool_this_frame)} 个OBU到证据池。证据池总数: {len(obu_evidence_pool)}")

    # 2. 基于更新后的 obu_evidence_pool 和最新的 layout_params, 重新构建最终的输出逻辑矩阵
    if not layout_params.get("is_calibrated"):
        logger.warning(f"会话 {session_id}: 布局参数未校准，无法基于证据池高质量重构矩阵。将尝试使用当前帧的初步映射（如果current_frame_yolo_logical_map有内容）。")
        # 如果布局未校准，我们只用当前帧的映射结果来尝试填充一次，不清空历史（除非是会话第一帧）
        # (这部分逻辑确保即使首次校准不完美，当前帧的结果也能被尝试记录)
        for yolo_idx, (r_log, c_log) in current_frame_yolo_logical_map.items():
            ocr_item_for_this_yolo = next((ocr for ocr in ocr_results_this_frame if ocr and ocr.get("original_index") == yolo_idx), None)
            if ocr_item_for_this_yolo:
                ocr_text = ocr_item_for_this_yolo.get("ocr_final_text", "")
                if ocr_text in VALID_OBU_CODES:
                    if logical_matrix[r_log][c_log] == 0 or logical_matrix[r_log][c_log] == 2 : # 只填充未知或之前失败的
                        logical_matrix[r_log][c_log] = 1
                        recognized_texts_map[(r_log, c_log)] = ocr_text
                        logger.info(f"会话 {session_id}: (布局未校准，初步填充) OBU '{ocr_text}' 填入逻辑坑位 ({r_log},{c_log})")
                elif logical_matrix[r_log][c_log] == 0: # OCR失败或无效
                     logical_matrix[r_log][c_log] = 2
    else: # 布局已校准，执行全局重排版
        logger.info(f"会话 {session_id}: 基于证据池({len(obu_evidence_pool)}个OBU)和已校准布局，开始重构最终逻辑矩阵...")

        # a. 先“清空”输出矩阵和文本映射（保留-1的不可用位）
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
        #    使用与 refine_layout_and_map_yolo_to_logical 中为当前帧YOLO锚点进行映射时一致的参数和逻辑
        row_ys_lp = layout_params.get("row_y_estimates", [])
        col_xs_lp_reg = layout_params.get("col_x_estimates_regular", [])
        y_match_thresh_lp = layout_params.get("avg_physical_row_height", 50) * 0.7
        avg_col_spacing_lp = layout_params.get("avg_col_spacing", layout_params.get("avg_obu_w",100))
        x_match_thresh_lp = avg_col_spacing_lp * 0.7
        expected_total_rows_lp = current_config["expected_total_rows"]
        inferred_regular_cols_lp = layout_params.get("inferred_regular_cols", current_config["regular_cols_count"])

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

# --- Main Image Processing Function (V5.1 框架 - 修正耗时计算) ---
def process_image_with_ocr_logic(image_path, current_onnx_session, session_id, current_layout_config, min_area_cfg, max_area_cfg):
    logger = current_app.logger
    logger.info(f"会话 {session_id}: 处理图片 {os.path.basename(image_path)} (V5.1 状态矩阵 - timing fix)")
    timing_profile = {}
    t_start_overall_processing = time.time() # 总处理开始时间

    # 1. Read Image
    t_start_step = time.time()
    original_image = cv2.imread(image_path)
    timing_profile['1_image_reading'] = time.time() - t_start_step
    if original_image is None:
        logger.error(f"错误: 无法读取图片: {image_path}")
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img_h, orig_img_w = original_image.shape[:2]
    logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    # 2. YOLO Detection & Area Filtering
    t_start_step = time.time()
    actual_max_area_threshold_px = None
    if max_area_cfg is not None:
        if isinstance(max_area_cfg, float) and 0 < max_area_cfg <= 1.0: actual_max_area_threshold_px = (orig_img_h * orig_img_w) * max_area_cfg
        elif isinstance(max_area_cfg, (int, float)) and max_area_cfg > 1: actual_max_area_threshold_px = float(max_area_cfg)

    logger.info("--- 开始整图检测 (YOLO) ---")
    input_cfg = current_onnx_session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640)
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int): model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]

    _t = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['2a_yolo_preprocessing'] = time.time() - _t;
    _t = time.time(); outputs_main = current_onnx_session.run(None, {input_name: input_tensor}); timing_profile['2b_yolo_inference'] = time.time() - _t;
    detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_image.shape[:2], (model_input_h_ref, model_input_w_ref), ratio_main, pad_x_main, pad_y_main, num_classes=len(COCO_CLASSES)); timing_profile['2c_yolo_postprocessing'] = time.time() - _t

    aggregated_boxes_xyxy = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]
    aggregated_scores = [d[4] for d in detections_result_list]
    aggregated_class_ids = [d[5] for d in detections_result_list]
    logger.info(f"YOLO检测完成。找到了 {len(aggregated_boxes_xyxy)} 个原始框。")

    _t_area_filter_start = time.time()
    if len(aggregated_boxes_xyxy) > 0 and ((min_area_cfg is not None and min_area_cfg > 0) or actual_max_area_threshold_px is not None):
        filtered_boxes,filtered_scores,filtered_ids=[],[],[]; initial_count=len(aggregated_boxes_xyxy)
        for i_box,box_xyxy in enumerate(aggregated_boxes_xyxy):
            b_w,b_h=box_xyxy[2]-box_xyxy[0],box_xyxy[3]-box_xyxy[1]; area=b_w*b_h; valid=True
            if min_area_cfg is not None and min_area_cfg > 0 and area < min_area_cfg: valid=False
            if actual_max_area_threshold_px is not None and area > actual_max_area_threshold_px: valid=False
            if valid: filtered_boxes.append(box_xyxy); filtered_scores.append(aggregated_scores[i_box]); filtered_ids.append(aggregated_class_ids[i_box])
        aggregated_boxes_xyxy,aggregated_scores,aggregated_class_ids=filtered_boxes,filtered_scores,filtered_ids;
        logger.info(f"面积筛选后剩 {len(aggregated_boxes_xyxy)} 个框.")
    timing_profile['2d_area_filtering'] = time.time() - _t_area_filter_start
    timing_profile['2_yolo_detection_total'] = time.time() - t_start_step

    # 准备YOLO结果给 map_yolo_and_update_layout
    yolo_boxes_for_mapping = []
    for i, yolo_box in enumerate(aggregated_boxes_xyxy):
        cx, cy, w, h = get_box_center_and_dims(yolo_box)
        if cx is not None:
            yolo_boxes_for_mapping.append({
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'box_yolo': yolo_box, 'score': aggregated_scores[i],
                'original_index': i
            })

    # 3. OCR Preprocessing & Task Preparation
    t_start_step = time.time()
    tasks_for_ocr = []
    ocr_input_metadata = [None] * len(aggregated_boxes_xyxy)
    if len(aggregated_boxes_xyxy) > 0:
        logger.info(f"--- 对 {len(aggregated_boxes_xyxy)} 个YOLO框进行OCR预处理 ---")
        for i, yolo_box_coords in enumerate(aggregated_boxes_xyxy):
            class_id = int(aggregated_class_ids[i]); class_name = COCO_CLASSES[class_id]
            x1_y, y1_y, x2_y, y2_y = yolo_box_coords; h_y, w_y = y2_y-y1_y, x2_y-x1_y
            y1_d_ideal = y1_y + int(h_y * DIGIT_ROI_Y_OFFSET_FACTOR); h_d_ideal = int(h_y * DIGIT_ROI_HEIGHT_FACTOR)
            y2_d_ideal = y1_d_ideal + h_d_ideal
            w_d_exp = int(w_y * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_y = x1_y + w_y / 2.0
            x1_d_ideal = int(cx_y - w_d_exp / 2.0); x2_d_ideal = int(cx_y + w_d_exp / 2.0)
            y1_d_clip,y2_d_clip = max(0,y1_d_ideal),min(orig_img_h,y2_d_ideal)
            x1_d_clip,x2_d_clip = max(0,x1_d_ideal),min(orig_img_w,x2_d_ideal)

            # 从 yolo_boxes_for_mapping 获取当前YOLO锚点的详细信息 (cx, cy, w, h, score)
            # yolo_boxes_for_mapping 是在步骤2之后，步骤3之前准备好的
            current_yolo_anchor_details_for_ocr_item = None
            if i < len(yolo_boxes_for_mapping):
                yolo_anchor_data_from_list = yolo_boxes_for_mapping[i]
                # 确保我们取的是正确的锚点 (original_index 应该匹配)
                if yolo_anchor_data_from_list.get('original_index') == i:
                    current_yolo_anchor_details_for_ocr_item = {
                        'cx': yolo_anchor_data_from_list.get('cx'),
                        'cy': yolo_anchor_data_from_list.get('cy'),
                        'w': yolo_anchor_data_from_list.get('w'),
                        'h': yolo_anchor_data_from_list.get('h'),
                        'score': yolo_anchor_data_from_list.get('score')
                    }
                else:
                    logger.warning(f"会话 {session_id}: 在构建ocr_input_metadata时，yolo_boxes_for_mapping[{i}] 的 original_index "
                                f"({yolo_anchor_data_from_list.get('original_index')}) 与当前循环索引 {i} 不匹配。")
            else:
                logger.warning(f"会话 {session_id}: 在构建ocr_input_metadata时，索引 {i} 超出 yolo_boxes_for_mapping 范围。")


            ocr_input_metadata[i] = {
                "original_index": i,
                "roi_index": i + 1,
                "class": class_name,
                "bbox_yolo": yolo_box_coords, # [x1,y1,x2,y2]
                "bbox_digit_ocr_clipped": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip],
                "confidence_yolo": float(aggregated_scores[i]),
                "yolo_anchor_details": current_yolo_anchor_details_for_ocr_item # MODIFIED: 添加YOLO锚点详细信息
            }
            img_for_ocr = None
            if x2_d_clip > x1_d_clip and y2_d_clip > y1_d_clip:
                digit_roi = original_image[y1_d_clip:y2_d_clip, x1_d_clip:x2_d_clip]
                h_roi, w_roi = digit_roi.shape[:2]
                if h_roi > 0 and w_roi > 0:
                    scale = TARGET_OCR_INPUT_HEIGHT / h_roi; target_w = int(w_roi * scale)
                    if target_w <= 0: target_w = 1
                    resized_roi = cv2.resize(digit_roi, (target_w, TARGET_OCR_INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
                    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img_for_ocr = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)

                    if SAVE_PROCESS_PHOTOS and img_for_ocr is not None: # Save OCR Slice
                        ocr_slice_dir = os.path.join(PROCESS_PHOTO_DIR, "ocr_slices", session_id) # Add session_id to subdir
                        if not os.path.exists(ocr_slice_dir): os.makedirs(ocr_slice_dir, exist_ok=True)
                        slice_filename = f"s_idx{i}_roi{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                        slice_output_path = os.path.join(ocr_slice_dir, slice_filename)
                        try: cv2.imwrite(slice_output_path, img_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        except Exception as e_save_slice: logger.error(f"保存OCR切片图失败 {slice_output_path}: {e_save_slice}")
            tasks_for_ocr.append((i, i + 1, img_for_ocr)) # i is original_index
    timing_profile['3_ocr_preprocessing_prep'] = time.time() - t_start_step

    # 4. Parallel OCR Processing & Result Consolidation
    t_start_step = time.time()
    final_ocr_results_list = [None] * len(aggregated_boxes_xyxy)
    ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes_xyxy)

    if tasks_for_ocr:
        logger.info(f"提交 {len(tasks_for_ocr)} 个OCR任务...")
        ocr_results_indexed_from_pool = [None] * len(tasks_for_ocr)
        global ocr_processing_pool, actual_num_ocr_workers
        if actual_num_ocr_workers > 1 and ocr_processing_pool:
            try:
                pool_res = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                for orig_idx_res, res_dict_ocr in pool_res: ocr_results_indexed_from_pool[orig_idx_res] = res_dict_ocr
            except Exception as e_map: logger.error(f"OCR Pool map error: {e_map}");
        else:
            logger.info("OCR串行处理")
            serial_ocr_predictor = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG): raise FileNotFoundError("Serial OCR model dir not found")
                serial_ocr_predictor = pdx.inference.create_predictor(model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG, model_name='PP-OCRv5_server_rec', device='cpu')
                for task_idx_s, task_data_s in enumerate(tasks_for_ocr):
                    orig_idx_s, _, img_data_s = task_data_s
                    if img_data_s is not None:
                        res_gen_s = serial_ocr_predictor.predict([img_data_s]); res_list_s = next(res_gen_s, None)
                        ocr_results_indexed_from_pool[orig_idx_s] = res_list_s[0] if (res_list_s and isinstance(res_list_s, list) and len(res_list_s) > 0) else (res_list_s if isinstance(res_list_s, dict) else {'rec_text': '', 'rec_score': 0.0})
                    else: ocr_results_indexed_from_pool[orig_idx_s] = {'rec_text': 'PREPROC_FAIL_SERIAL', 'rec_score': 0.0}
                if serial_ocr_predictor: del serial_ocr_predictor
            except Exception as e_serial_ocr: logger.error(f"Serial OCR error: {e_serial_ocr}")

        for i, ocr_dict in enumerate(ocr_results_indexed_from_pool):
            full_res_item = {**(ocr_input_metadata[i] or {})}
            if ocr_dict and isinstance(ocr_dict, dict):
                raw_txt = ocr_dict.get('rec_text', ""); score = ocr_dict.get('rec_score', 0.0)
                if raw_txt and raw_txt not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_SERIAL']:
                    digits = "".join(re.findall(r'\d', raw_txt))
                    full_res_item["ocr_final_text"] = digits if digits else "N/A_NO_DIGITS"
                    ocr_texts_for_drawing[i] = digits if digits else "ERR"
                else: full_res_item["ocr_final_text"] = raw_txt; ocr_texts_for_drawing[i] = "ERR"
                full_res_item["ocr_confidence"] = score
            else:
                full_res_item["ocr_final_text"] = "N/A_OCR_FAIL"; ocr_texts_for_drawing[i] = "N/A"; full_res_item["ocr_confidence"] = 0.0
            final_ocr_results_list[i] = full_res_item
    else:
        logger.info("无任务进行OCR")
        for i in range(len(aggregated_boxes_xyxy)): # Populate with N/A if no OCR tasks
            final_ocr_results_list[i] = {**(ocr_input_metadata[i] or {}), "ocr_final_text": "N/A_NO_OCR_TASKS", "ocr_confidence": 0.0}
    timing_profile['4_ocr_processing_total'] = time.time() - t_start_step
    logger.info(f"OCR处理与结果整合完成 ({timing_profile['4_ocr_processing_total']:.3f}s)")

    # 5. 调用新的布局校准与YOLO映射函数
    t_start_step = time.time()
    # MODIFIED: 调用新的 refine_layout_and_map_yolo_to_logical 函数
    # 它需要 image_wh 参数，我们从 original_image.shape[:2] 获取
    image_height_width = original_image.shape[:2]
    current_frame_yolo_logical_map, layout_was_updated = refine_layout_and_map_yolo_to_logical(
        yolo_boxes_for_mapping, session_id, image_height_width, logger
    )
    if layout_was_updated:
        logger.info(f"会话 {session_id}: 布局参数已通过 refine_layout_and_map_yolo_to_logical 更新。")
    timing_profile['5_refine_layout_and_map_yolo'] = time.time() - t_start_step # MODIFIED: timing key name

    # 6. 更新会话的状态矩阵 (使用新的 update_session_state_from_ocr 函数)
    t_start_step = time.time()
    # MODIFIED: 调用新的 update_session_state_from_ocr 函数
    # 它需要 current_frame_yolo_logical_map (来自refine_layout...) 和 final_ocr_results_list
    update_session_state_from_ocr(session_id, current_frame_yolo_logical_map, final_ocr_results_list, logger)
    timing_profile['6_update_session_state_from_ocr'] = time.time() - t_start_step

    # 7. (可选) 保存YOLO标注图 (matrix_viz图由客户端根据JSON绘制)
    if SAVE_PROCESS_PHOTOS and len(aggregated_boxes_xyxy) > 0:
        t_start_step = time.time()
        # Create a fresh copy for drawing to avoid modifying original_image if it's used elsewhere
        image_to_draw_on = original_image.copy()
        annotated_img = draw_detections(image_to_draw_on, np.array(aggregated_boxes_xyxy),
                                        np.array(aggregated_scores), np.array(aggregated_class_ids),
                                        COCO_CLASSES, ocr_texts=ocr_texts_for_drawing,
                                        roi_indices=[item.get('roi_index') for item in final_ocr_results_list if item])
        img_name_base = os.path.splitext(os.path.basename(image_path))[0]
        ts_filename = datetime.now().strftime("%Y%m%d%H%M%S%f") # Unique timestamp for filename
        annotated_path = os.path.join(PROCESS_PHOTO_DIR, f"annotated_{img_name_base}_{ts_filename}.jpg")
        try:
            cv2.imwrite(annotated_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY])
            logger.info(f"YOLO标注图已保存: {annotated_path}")
        except Exception as e_save_ann:
            logger.error(f"保存YOLO标注图失败: {e_save_ann}")
        timing_profile['7_drawing_yolo_annotations'] = time.time() - t_start_step

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    logger.info(f"--- Timing profile for {os.path.basename(image_path)} ({session_id}) ---")
    for stage_key in sorted(timing_profile.keys()): # Sort keys for consistent log order
        logger.info(f"  {stage_key}: {timing_profile[stage_key]:.3f}s")

    # MODIFIED: 从 session_data_store 重新获取最新的会话数据
    current_session_state = session_data_store.get(session_id)
    if not current_session_state:
        logger.error(f"会话 {session_id}: 在 process_image_with_ocr_logic 末尾无法获取会话状态！返回空。")
        # 根据您的函数签名，它期望返回 matrix, texts, timings, warnings
        # 如果会话丢失，我们应该返回一些默认值
        empty_matrix = [[0] * current_layout_config.get("regular_cols_count", 4)
                        for _ in range(current_layout_config.get("expected_total_rows", 13))]
        # (确保 current_layout_config 在此作用域内可用，如果不可用，则需要传递或使用固定默认值)
        # 假设 current_layout_config 在此函数作用域内是可用的 (它是 process_image_with_ocr_logic 的参数)
        return empty_matrix, {}, timing_profile, [{"message": "Error: Session data lost before returning."}]


    final_matrix_to_return = current_session_state.get("logical_matrix", [])
    final_texts_to_return = current_session_state.get("recognized_texts_map", {})

    # 假设 warnings_from_processing 是在此函数内某处收集的，如果不是，需要调整
    # 从您之前的 predict_image_route 代码看，warnings_from_processing 似乎是 process_image_with_ocr_logic 的一个预期返回值
    # 但在 process_image_with_ocr_logic 函数体内并没有看到它的定义或收集。
    # 为了保持接口一致性，暂时返回一个空列表。如果后续需要，我们可以添加警告收集逻辑。
    warnings_for_return = []

    return final_matrix_to_return, final_texts_to_return, timing_profile, warnings_for_return

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
                "recognized_texts_map": {},  # MODIFIED: 键名从 "recognized_texts" 改为 "recognized_texts_map"
                "obu_evidence_pool": {},
                "layout_parameters": {"is_calibrated": False}, # 强制重新校准时，is_calibrated设为False
                "current_layout_config": current_layout_config_for_session,
                "frame_count": 0,
                "last_activity": datetime.now()
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