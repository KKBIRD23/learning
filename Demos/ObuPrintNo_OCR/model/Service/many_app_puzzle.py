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
from typing import List, Tuple, Dict, Any

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

def calibrate_layout_parameters_from_benchmark_frame(benchmark_yolo_boxes, layout_config, image_wh, logger):
    """
    V5.3 P5: 基于基准帧的YOLO锚点，一次性精确计算所有逻辑坑位的预期物理坐标。
    Args:
        benchmark_yolo_boxes (list): 来自高质量基准帧的YOLO锚点列表
                                    [{'cx', 'cy', 'w', 'h', 'score', 'original_index'}, ...]
        layout_config (dict): 包含总行数、列数、特殊行等信息
        image_wh (tuple): (img_w, img_h)
        logger: 日志记录器
    Returns:
        dict: 包含 "ideal_slot_physical_coords" 等校准后参数的字典，如果失败则返回None
    """
    logger.info("开始基于基准帧进行布局参数校准...")
    if not benchmark_yolo_boxes or len(benchmark_yolo_boxes) < MIN_YOLO_ANCHORS_FOR_LAYOUT: # MIN_YOLO_ANCHORS_FOR_LAYOUT
        logger.warning(f"基准帧YOLO锚点数量 ({len(benchmark_yolo_boxes)}) 不足，无法校准。")
        return None

    # 1. 对基准帧YOLO锚点进行行分组
    # (与 interpret_yolo_structure_and_map_to_logical 中类似的行分组逻辑)
    sorted_benchmark_boxes = sorted(benchmark_yolo_boxes, key=lambda b: (b['cy'], b['cx']))
    physical_rows_benchmark = []
    # ... (完整的行分组逻辑，得到 physical_rows_benchmark) ...
    _avg_h_bm = np.mean([a['h'] for a in sorted_benchmark_boxes if a.get('h',0) > 0]) if any(a.get('h',0) > 0 for a in sorted_benchmark_boxes) else 30
    _y_thresh_bm = _avg_h_bm * YOLO_ROW_GROUP_Y_THRESHOLD_FACTOR
    if not sorted_benchmark_boxes: return None
    _curr_row_bm = [sorted_benchmark_boxes[0]]
    for i in range(1, len(sorted_benchmark_boxes)):
        if abs(sorted_benchmark_boxes[i]['cy'] - _curr_row_bm[-1]['cy']) < _y_thresh_bm: _curr_row_bm.append(sorted_benchmark_boxes[i])
        else: physical_rows_benchmark.append(sorted(_curr_row_bm, key=lambda a:a['cx'])); _curr_row_bm = [sorted_benchmark_boxes[i]]
    if _curr_row_bm: physical_rows_benchmark.append(sorted(_curr_row_bm, key=lambda a:a['cx']))
    if not physical_rows_benchmark: logger.warning("基准帧行分组失败。"); return None
    logger.info(f"基准帧YOLO行分组为 {len(physical_rows_benchmark)} 行。")


    # 2. 识别锚定行 (例如，物理底部特殊行 -> 逻辑行12) 和学习初始几何参数
    #    这是最核心的算法，需要非常鲁棒。
    #    初始V1简化版：假设物理底部是特殊行，并从中学习行高、列参数。
    #    (这部分逻辑可以从 interpret_yolo_structure_and_map_to_logical 的V5.1 P3版本中借鉴和强化)

    calibrated_params = {
        "ideal_slot_physical_coords": {}, # {(r,c): {'cx','cy','w','h'}}
        "avg_physical_row_height": 50,    # Default
        "avg_col_spacing": 100,         # Default
        "avg_obu_w": np.mean([a['w'] for a in benchmark_yolo_boxes if a.get('w',0)>0]) or 100,
        "avg_obu_h": np.mean([a['h'] for a in benchmark_yolo_boxes if a.get('h',0)>0]) or 40,
        "special_row_at_logical_top": False # Default
    }

    # 2a. 尝试锚定特殊行 (默认物理底部是逻辑底部特殊行)
    # (与 interpret_yolo_structure_and_map_to_logical V5.1 P3 类似逻辑)
    # ... (识别特殊行，确定其是逻辑顶部还是底部 -> calibrated_params["special_row_at_logical_top"])
    # ... (学习 avg_physical_row_height)
    # ... (学习 avg_start_x_for_reg_cols, avg_col_spacing_for_reg_cols)
    # 这是一个非常复杂的步骤，我们先用一个高度简化的占位符：
    # 假设我们能从 physical_rows_benchmark 中找到最底部的一行作为逻辑第12行，
    # 最顶部的一行作为逻辑第0行（如果行数足够多），然后插值。
    # 列也类似，从最左最右的X坐标均匀分布。这仍然不够好，但作为起点。

    if len(physical_rows_benchmark) >= 2: # 需要至少两行来估算行高和趋势
        # 简化：取所有检测到的物理行的Y坐标范围
        all_phys_row_mean_cy = [np.mean([a['cy'] for a in r]) for r in physical_rows_benchmark if r]
        if not all_phys_row_mean_cy : logger.warning("基准帧无有效行Y坐标"); return None
        min_phys_y, max_phys_y = min(all_phys_row_mean_cy), max(all_phys_row_mean_cy)

        # 假设检测到的行块均匀分布在13个逻辑行中的某一段
        # (这个假设非常强，后续需要用特殊行锚定来优化)
        # 暂时：将检测到的物理行直接对应到逻辑行的底部几行
        num_phys_rows_to_map = len(physical_rows_benchmark)
        start_logical_row = layout_config["expected_total_rows"] - num_phys_rows_to_map
        if start_logical_row < 0: start_logical_row = 0 # 不能小于0

        temp_row_y_estimates = [0.0] * layout_config["expected_total_rows"]
        temp_col_x_estimates_reg = [0.0] * layout_config["regular_cols_count"]

        # Y坐标：基于检测到的行高，从底部向上推
        if num_phys_rows_to_map > 1:
            calibrated_params["avg_physical_row_height"] = (max_phys_y - min_phys_y) / (num_phys_rows_to_map -1)
        else: # 只有一行物理行
            calibrated_params["avg_physical_row_height"] = calibrated_params["avg_obu_h"] * 1.2

        anchor_y = max_phys_y
        anchor_lr = layout_config["expected_total_rows"] - 1
        for lr in range(layout_config["expected_total_rows"]):
            temp_row_y_estimates[lr] = anchor_y - (anchor_lr - lr) * calibrated_params["avg_physical_row_height"]

        # X坐标：基于所有检测到的锚点的X范围，均匀分布4列
        all_phys_cx = [a['cx'] for r in physical_rows_benchmark for a in r]
        if not all_phys_cx: logger.warning("基准帧无有效X坐标"); return None
        min_phys_x, max_phys_x = min(all_phys_cx), max(all_phys_cx)
        if layout_config["regular_cols_count"] > 1:
            x_step = (max_phys_x - min_phys_x) / (layout_config["regular_cols_count"] - 1)
            for lc in range(layout_config["regular_cols_count"]):
                temp_col_x_estimates_reg[lc] = min_phys_x + lc * x_step
        else:
            temp_col_x_estimates_reg[0] = (min_phys_x + max_phys_x) / 2

        calibrated_params["row_y_estimates"] = temp_row_y_estimates
        calibrated_params["col_x_estimates_regular"] = temp_col_x_estimates_reg
        calibrated_params["avg_col_spacing"] = x_step if layout_config["regular_cols_count"] > 1 else calibrated_params["avg_obu_w"]*1.1

        logger.info(f"  基准校准：行Y估算: {[int(y) for y in calibrated_params['row_y_estimates']]}")
        logger.info(f"  基准校准：列X估算: {[int(x) for x in calibrated_params['col_x_estimates_regular']]}")

        # 3. 生成所有13x4逻辑坑位的预期物理坐标
        for r_log in range(layout_config["expected_total_rows"]):
            expected_cy = calibrated_params["row_y_estimates"][r_log]
            is_special = (r_log == layout_config["expected_total_rows"] - 1 and not calibrated_params["special_row_at_logical_top"]) or \
                         (r_log == 0 and calibrated_params["special_row_at_logical_top"])

            cols_this_row = layout_config["special_row_cols_count"] if is_special else layout_config["regular_cols_count"]

            current_row_expected_xs = []
            if is_special and layout_config["special_row_cols_count"] == 2 and layout_config["regular_cols_count"] == 4:
                if len(calibrated_params["col_x_estimates_regular"]) == 4:
                    current_row_expected_xs = [calibrated_params["col_x_estimates_regular"][1], calibrated_params["col_x_estimates_regular"][2]]
            else:
                current_row_expected_xs = calibrated_params["col_x_estimates_regular"][:cols_this_row]

            for c_log_in_row, expected_cx in enumerate(current_row_expected_xs):
                # 将行内列索引转换回全局逻辑列索引
                final_c_log = c_log_in_row
                if is_special and layout_config["special_row_cols_count"] == 2 and layout_config["regular_cols_count"] == 4:
                    final_c_log = c_log_in_row + 1 # 0->1, 1->2

                calibrated_params["ideal_slot_physical_coords"][(r_log, final_c_log)] = {
                    'cx': int(expected_cx), 'cy': int(expected_cy),
                    'w': int(calibrated_params["avg_obu_w"]), 'h': int(calibrated_params["avg_obu_h"])
                }
        logger.info(f"  基准校准：生成了 {len(calibrated_params['ideal_slot_physical_coords'])} 个理想坑位的物理坐标。")
        return calibrated_params # 返回包含 ideal_slot_physical_coords 的参数字典
    else: # 可靠行不足
        logger.warning("基准帧可靠行不足2行，无法进行精确校准。")
        return None

def interpret_yolo_structure_and_map_to_logical(current_yolo_boxes_with_orig_idx, session_id, image_wh, logger):
    """
    V5.3 P5: 如果布局未校准，则调用校准函数；否则，使用已校准的布局参数
              将当前帧YOLO锚点映射到逻辑坐标。
    """
    logger.info(f"会话 {session_id}: (V5.3 P5) 映射当前帧YOLO...")
    session = session_data_store.get(session_id)
    if not session: # Should not happen if predict_image_route handles session creation
        logger.error(f"会话 {session_id}: 严重错误 - interpret_yolo 未找到会话数据。")
        return {}, False

    layout_params = session["layout_parameters"]
    current_config = session["current_layout_config"]
    layout_updated_this_run = False
    current_frame_yolo_to_logical_mapping = {}

    if not current_yolo_boxes_with_orig_idx:
        logger.info(f"会话 {session_id}: 当前帧无YOLO检测框。")
        return {}, False

    # 步骤 1: 检查布局是否已校准，如果未校准，则使用当前帧进行校准
    if not layout_params.get("is_calibrated", False):
        logger.info(f"会话 {session_id}: 布局参数未校准，使用当前帧作为基准进行校准...")
        # 假设当前帧是高质量的基准帧
        calibrated_params_from_benchmark = calibrate_layout_parameters_from_benchmark_frame(
            current_yolo_boxes_with_orig_idx, # 使用当前帧的YOLO作为基准数据
            current_config,
            image_wh,
            logger
        )
        if calibrated_params_from_benchmark:
            session["layout_parameters"] = calibrated_params_from_benchmark # 更新会话的布局参数
            session["layout_parameters"]["is_calibrated"] = True # 标记已校准
            layout_updated_this_run = True
            logger.info(f"会话 {session_id}: 布局参数已通过基准帧成功校准。")
        else:
            logger.error(f"会话 {session_id}: 基准帧布局参数校准失败！无法进行映射。")
            return {}, False # 校准失败，无法继续

    # 步骤 2: 使用已校准的布局参数，为当前帧的YOLO锚点推断逻辑坐标
    # 现在 layout_params 中应该有 "ideal_slot_physical_coords"
    ideal_slots_map = session["layout_parameters"].get("ideal_slot_physical_coords", {})
    if not ideal_slots_map:
        logger.error(f"会话 {session_id}: 布局参数中缺少 ideal_slot_physical_coords，无法映射。")
        return {}, False

    avg_w_from_layout = session["layout_parameters"].get("avg_obu_w", 100)
    avg_h_from_layout = session["layout_parameters"].get("avg_obu_h", 40)
    # 匹配阈值可以基于平均OBU尺寸的一半左右
    x_match_threshold = avg_w_from_layout * 0.5
    y_match_threshold = avg_h_from_layout * 0.5

    for anchor in current_yolo_boxes_with_orig_idx:
        if 'cx' not in anchor or 'cy' not in anchor: continue

        best_r_c_match = None
        min_dist_sq = float('inf')

        for (r_log, c_log), slot_coords in ideal_slots_map.items():
            dist_sq = (anchor['cy'] - slot_coords['cy'])**2 + (anchor['cx'] - slot_coords['cx'])**2
            # 检查是否在阈值内
            if abs(anchor['cy'] - slot_coords['cy']) < y_match_threshold and \
               abs(anchor['cx'] - slot_coords['cx']) < x_match_threshold:
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_r_c_match = (r_log, c_log)

        if best_r_c_match:
            current_frame_yolo_to_logical_mapping[anchor['original_index']] = best_r_c_match

    logger.info(f"会话 {session_id}: (V5.3 P5) 当前帧YOLO映射到固定逻辑坐标 {len(current_frame_yolo_to_logical_mapping)} 个。")
    return current_frame_yolo_to_logical_mapping, layout_updated_this_run

# --- 更新会话状态的核心函数 V5.3 ---
def update_session_state_from_ocr(session_id, current_frame_yolo_logical_map, ocr_results_this_frame, current_yolo_boxes_for_evidence, logger):
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 严重错误 - update_session_state_from_ocr 未找到会话数据。")
        return
    logger.info(f"会话 {session_id}: (V5.3) 更新状态矩阵和OBU证据池...")

    # 使用正确的键名 "logical_status_matrix"
    logical_status_matrix = session["logical_status_matrix"]
    recognized_texts_map = session["recognized_texts_map"]
    obu_evidence_pool = session["obu_evidence_pool"]
    layout_params = session["layout_parameters"]
    current_config = session["current_layout_config"]

    # 1. 将当前帧通过DB校验的OCR结果更新到 obu_evidence_pool
    for ocr_item in ocr_results_this_frame:
        if not ocr_item: continue
        original_yolo_idx = ocr_item.get("original_index")
        ocr_text = ocr_item.get("ocr_final_text", "")
        ocr_confidence = ocr_item.get("ocr_confidence", 0.0)
        is_valid_in_db = ocr_text in VALID_OBU_CODES

        if is_valid_in_db:
            yolo_anchor_info_for_this_ocr = next((box for box in current_yolo_boxes_for_evidence if box.get('original_index') == original_yolo_idx), None)
            if not yolo_anchor_info_for_this_ocr:
                logger.warning(f"会话 {session_id}: 有效OBU '{ocr_text}' 缺少YOLO锚点(idx:{original_yolo_idx})，无法入池。")
                continue
            if ocr_text not in obu_evidence_pool:
                obu_evidence_pool[ocr_text] = {'physical_anchors': [], 'ocr_confidence': 0.0, 'logical_coord_estimate': None, 'last_seen_frame': 0}
            current_anchor_entry = {
                'cx': yolo_anchor_info_for_this_ocr['cx'], 'cy': yolo_anchor_info_for_this_ocr['cy'],
                'w': yolo_anchor_info_for_this_ocr['w'], 'h': yolo_anchor_info_for_this_ocr['h'],
                'score': yolo_anchor_info_for_this_ocr['score'], 'frame_id': session.get("frame_count", 0)}
            obu_evidence_pool[ocr_text]['physical_anchors'] = [current_anchor_entry]
            obu_evidence_pool[ocr_text]['ocr_confidence'] = max(obu_evidence_pool[ocr_text]['ocr_confidence'], ocr_confidence)
            obu_evidence_pool[ocr_text]['last_seen_frame'] = session.get("frame_count", 0)

    # 2. 基于更新后的 obu_evidence_pool 和最新的 layout_params, 重新构建输出矩阵
    if layout_params.get("is_calibrated"):
        logger.info(f"会话 {session_id}: (V5.3) 基于证据池和布局，重构逻辑矩阵...")
        for r in range(len(logical_status_matrix)): # 使用 logical_status_matrix
            for c in range(len(logical_status_matrix[r])):
                if logical_status_matrix[r][c] != -1:
                    logical_status_matrix[r][c] = 0
                    if (r,c) in recognized_texts_map: del recognized_texts_map[(r,c)]

        potential_placements = []
        # ... (与上一版相同的 potential_placements 计算逻辑，使用 layout_params) ...
        row_ys_lp = layout_params.get("row_y_estimates", []); col_xs_lp_reg = layout_params.get("col_x_estimates_regular", [])
        avg_h_lp = layout_params.get("avg_obu_h", 40); avg_w_lp = layout_params.get("avg_obu_w", 100)
        y_match_thresh_lp_final = layout_params.get("avg_physical_row_height", avg_h_lp) * 0.75
        x_match_thresh_lp_final = (layout_params.get("avg_col_spacing", avg_w_lp) or avg_w_lp) * 0.75
        if row_ys_lp and col_xs_lp_reg:
            for obu_text_verified, evidence_data in obu_evidence_pool.items():
                if not evidence_data.get('physical_anchors'): continue
                best_anchor_for_obu = evidence_data['physical_anchors'][-1]
                r_final, c_final = -1, -1; min_dist_sq_final = float('inf')
                for r_idx, r_y_est in enumerate(row_ys_lp):
                    is_special_row_final = (r_idx == current_config["expected_total_rows"] - 1 and not layout_params.get("special_row_at_logical_top", False)) or \
                                           (r_idx == 0 and layout_params.get("special_row_at_logical_top", False))
                    cols_in_this_logical_row_final = current_config["special_row_cols_count"] if is_special_row_final else current_config["regular_cols_count"]
                    current_row_col_xs_to_match_final = []
                    if is_special_row_final and current_config["special_row_cols_count"] == 2 and current_config["regular_cols_count"] == 4:
                        if len(col_xs_lp_reg) == 4: current_row_col_xs_to_match_final = [col_xs_lp_reg[1], col_xs_lp_reg[2]]
                    else: current_row_col_xs_to_match_final = col_xs_lp_reg[:cols_in_this_logical_row_final]
                    if not current_row_col_xs_to_match_final: continue
                    for c_idx_in_row, c_x_est in enumerate(current_row_col_xs_to_match_final):
                        dist_sq = (best_anchor_for_obu['cy'] - r_y_est)**2 + (best_anchor_for_obu['cx'] - c_x_est)**2
                        if dist_sq < min_dist_sq_final and \
                           abs(best_anchor_for_obu['cy'] - r_y_est) < y_match_thresh_lp_final and \
                           abs(best_anchor_for_obu['cx'] - c_x_est) < x_match_thresh_lp_final:
                            min_dist_sq_final = dist_sq; r_final = r_idx
                            if is_special_row_final and current_config["special_row_cols_count"] == 2 and current_config["regular_cols_count"] == 4:
                                c_final = c_idx_in_row + 1
                            else: c_final = c_idx_in_row
                if r_final != -1 and c_final != -1:
                    match_score = evidence_data.get('ocr_confidence', 0.0) + best_anchor_for_obu.get('score',0.0)
                    potential_placements.append((obu_text_verified, evidence_data, r_final, c_final, match_score))
            potential_placements.sort(key=lambda x: x[4], reverse=True)
            for obu_text_p, evi_data_p, r_p, c_p, score_p in potential_placements:
                if 0 <= r_p < len(logical_status_matrix) and 0 <= c_p < len(logical_status_matrix[0]) and logical_status_matrix[r_p][c_p] == 0 : # 使用 logical_status_matrix
                    logical_status_matrix[r_p][c_p] = 1; recognized_texts_map[(r_p, c_p)] = obu_text_p
                    evi_data_p['logical_coord_estimate'] = (r_p, c_p)
                elif 0 <= r_p < len(logical_status_matrix) and 0 <= c_p < len(logical_status_matrix[0]) and recognized_texts_map.get((r_p,c_p)) != obu_text_p :
                     logger.warning(f"  全局重排版: 坑位({r_p},{c_p}) 已被 '{recognized_texts_map.get((r_p,c_p))}' 占据，'{obu_text_p}' 无法放入。")

    # 3. 对于当前帧中YOLO检测到、有逻辑映射、但OCR失败或DB校验失败的，标记为状态2
    for original_yolo_idx_fail, (r_log_fail, c_log_fail) in current_frame_yolo_logical_map.items():
        if 0 <= r_log_fail < len(logical_status_matrix) and 0 <= c_log_fail < len(logical_status_matrix[0]) and logical_status_matrix[r_log_fail][c_log_fail] == 0: # 使用 logical_status_matrix
            ocr_for_this_yolo_fail = next((ocr for ocr in ocr_results_this_frame if ocr and ocr.get("original_index") == original_yolo_idx_fail), None)
            is_ocr_valid_for_this_yolo = False
            if ocr_for_this_yolo_fail:
                ocr_text_fail = ocr_for_this_yolo_fail.get("ocr_final_text","")
                if ocr_text_fail in VALID_OBU_CODES: is_ocr_valid_for_this_yolo = True
            if not is_ocr_valid_for_this_yolo:
                 logical_status_matrix[r_log_fail][c_log_fail] = 2 # 使用 logical_status_matrix
                 logger.info(f"会话 {session_id}: 矩阵[{r_log_fail}][{c_log_fail}] 因YOLO检测到但OCR无效而标记为失败。")
    logger.info(f"会话 {session_id}: (V5.3) 状态矩阵和证据池更新完成。")

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

# --- 新增：YOLO逻辑位置映射调试图绘制函数 ---
def draw_yolo_logical_placement_map(yolo_to_logical_map, current_config, session_id, image_name_base, frame_num_in_session, logger):
    """
    在服务端生成一张图，显示当前帧的YOLO检测框被映射到了哪些逻辑坑位。
    Args:
        yolo_to_logical_map (dict): {original_yolo_index: (logical_row, logical_col)}
        current_config (dict): 当前会话的布局配置 (LAYOUT_CONFIG)
        session_id (str): 当前会话ID
        image_name_base (str): 原始图片的文件名基础部分 (不含扩展名)
        frame_num_in_session (int): 当前图片在会话中的帧序号
        logger: Flask app logger instance
    """
    if not yolo_to_logical_map:
        logger.info(f"会话 {session_id}: 无YOLO到逻辑坐标的映射数据，不生成YOLO位置图。")
        return

    num_rows = current_config["expected_total_rows"]
    num_cols = current_config["regular_cols_count"]

    cell_size = 60  # 单元格像素大小
    padding = 15
    spacing = 5

    img_width = num_cols * cell_size + (num_cols - 1) * spacing + 2 * padding
    img_height = num_rows * cell_size + (num_rows - 1) * spacing + 2 * padding

    canvas = np.full((img_height, img_width, 3), (220, 220, 220), dtype=np.uint8) # 浅灰色背景

    color_yolo_mapped_fill = (100, 100, 255)  # 例如，一种蓝色表示YOLO映射到此
    color_unavailable_fill = (250, 250, 250) # 非常浅的灰色或接近白色，表示不可用
    color_text = (0, 0, 0)      # 黑色文字
    font_scale = 0.4
    font_thickness = 1

    # 创建一个标记矩阵，记录哪些逻辑坑位被YOLO映射了
    placement_marker_matrix = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for original_yolo_idx, (r, c) in yolo_to_logical_map.items():
        if 0 <= r < num_rows and 0 <= c < num_cols:
            # 如果一个坑位被多个YOLO框映射（理论上不应发生，或应由映射算法解决），这里简单覆盖
            placement_marker_matrix[r][c] = str(original_yolo_idx) # 存储YOLO的原始索引

    # 标记特殊行两边的不可用格子
    special_row_idx = current_config["expected_total_rows"] - 1 # 假设特殊行在底部
    if not current_config.get("special_row_at_logical_top", False): # 再次确认特殊行位置假设
        if current_config["expected_total_rows"] > 0 and \
           current_config["regular_cols_count"] == 4 and \
           current_config["special_row_cols_count"] == 2:
            if 0 <= special_row_idx < num_rows:
                if placement_marker_matrix[special_row_idx][0] is None : placement_marker_matrix[special_row_idx][0] = "NA" # 标记为不可用
                if placement_marker_matrix[special_row_idx][3] is None : placement_marker_matrix[special_row_idx][3] = "NA"
    # (如果特殊行可能在顶部，也需要类似逻辑处理)


    for r in range(num_rows):
        for c in range(num_cols):
            cell_x_start = padding + c * (cell_size + spacing)
            cell_y_start = padding + r * (cell_size + spacing)
            center_x = cell_x_start + cell_size // 2
            center_y = cell_y_start + cell_size // 2

            marker = placement_marker_matrix[r][c]
            current_fill_color = (200, 200, 200) # 默认未被映射的格子颜色
            display_text = ""

            if marker == "NA": # 不可用格子
                current_fill_color = color_unavailable_fill
            elif marker is not None: # YOLO映射到此
                current_fill_color = color_yolo_mapped_fill
                display_text = f"Y:{marker}" # 显示YOLO原始索引

            cv2.rectangle(canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          current_fill_color, -1)
            cv2.rectangle(canvas,
                          (cell_x_start, cell_y_start),
                          (cell_x_start + cell_size, cell_y_start + cell_size),
                          (50,50,50), 1) # 边框

            if display_text:
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
                cv2.putText(canvas, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, font_thickness, cv2.LINE_AA)

    # 保存图像
    output_map_dir = os.path.join(PROCESS_PHOTO_DIR, "yolo_logical_maps", session_id)
    if not os.path.exists(output_map_dir):
        os.makedirs(output_map_dir, exist_ok=True)

    map_image_name = f"yolo_map_{image_name_base}_s{session_id[:8]}_f{frame_num_in_session}.png" # 使用PNG以便清晰
    map_output_path = os.path.join(output_map_dir, map_image_name)
    try:
        cv2.imwrite(map_output_path, canvas)
        logger.info(f"YOLO逻辑位置映射图已保存到: {map_output_path}")
    except Exception as e_imwrite_map:
        logger.error(f"保存YOLO逻辑位置映射图失败 {map_output_path}: {e_imwrite_map}")

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

# --- Main Image Processing Function (V5.3 框架) ---
def process_image_with_ocr_logic(image_path, current_onnx_session, session_id, current_layout_config, min_area_cfg, max_area_cfg):
    logger = current_app.logger
    logger.info(f"会话 {session_id}: 处理图片 {os.path.basename(image_path)} (V5.3 DB Check)") # 更新版本号
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"会话 {session_id}: 严重错误 - process_image_with_ocr_logic 未找到会话数据。")
        # 根据您的错误处理策略，这里可能需要返回错误或空结果
        # 为简单起见，我们先假设 predict_image_route 已经确保了 session 的存在
        # 但健壮的做法是这里也处理 session 为 None 的情况
        # 例如: return [], {}, timing_profile (如果函数签名允许)
        # 或者抛出异常，由上层捕获
        # 暂时先假设 session 一定存在，因为 predict_image_route 会创建它
        # 但如果 predict_image_route 的逻辑有变，这里就需要更强的保护
        # 为了确保能运行，如果真的没找到，我们返回一个表示错误的状态
        return [], {}, {"error": "Session not found in process_image_with_ocr_logic"}
    timing_profile = {}
    t_start_overall_processing = time.time() # 总处理开始时间

    # 1. Read Image
    t_start_step = time.time()
    original_image = cv2.imread(image_path)
    timing_profile['1_image_reading'] = time.time() - t_start_step
    if original_image is None:
        logger.error(f"错误: 无法读取图片: {image_path}")
        # 在实际应用中，这里应该向上抛出异常或返回特定错误，而不是直接raise FileNotFoundError导致500
        # 但为了保持与之前逻辑的兼容性，暂时保留
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img_h, orig_img_w = original_image.shape[:2]
    logger.info(f"原始图片: {os.path.basename(image_path)} (H={orig_img_h}, W={orig_img_w})")

    # 2. YOLO Detection & Area Filtering
    t_start_step = time.time()
    actual_max_area_threshold_px = None
    if max_area_cfg is not None: # 使用传入的参数
        if isinstance(max_area_cfg, float) and 0 < max_area_cfg <= 1.0: actual_max_area_threshold_px = (orig_img_h * orig_img_w) * max_area_cfg
        elif isinstance(max_area_cfg, (int, float)) and max_area_cfg > 1: actual_max_area_threshold_px = float(max_area_cfg)

    logger.info("--- 开始整图检测 (YOLO) ---")
    input_cfg = current_onnx_session.get_inputs()[0]; input_name = input_cfg.name; input_shape_onnx = input_cfg.shape
    model_input_h_ref, model_input_w_ref = (640, 640) # 假设的YOLO输入尺寸
    if len(input_shape_onnx) == 4 and isinstance(input_shape_onnx[2], int) and isinstance(input_shape_onnx[3], int):
        model_input_h_ref, model_input_w_ref = input_shape_onnx[2], input_shape_onnx[3]

    _t = time.time(); input_tensor, ratio_main, pad_x_main, pad_y_main = preprocess_onnx_for_main(original_image, (model_input_h_ref, model_input_w_ref)); timing_profile['2a_yolo_preprocessing'] = time.time() - _t;
    _t = time.time(); outputs_main = current_onnx_session.run(None, {input_name: input_tensor}); timing_profile['2b_yolo_inference'] = time.time() - _t;
    # 注意：确保 postprocess_yolo_onnx_for_main 函数定义与这里的调用参数一致
    detections_result_list = postprocess_yolo_onnx_for_main(outputs_main, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
                                                            original_image.shape[:2], (model_input_h_ref, model_input_w_ref),
                                                            ratio_main, pad_x_main, pad_y_main,
                                                            num_classes=len(COCO_CLASSES));
    timing_profile['2c_yolo_postprocessing'] = time.time() - _t

    aggregated_boxes_xyxy = [[d[0], d[1], d[2], d[3]] for d in detections_result_list]
    aggregated_scores = [d[4] for d in detections_result_list]
    aggregated_class_ids = [d[5] for d in detections_result_list]
    logger.info(f"YOLO检测完成。找到了 {len(aggregated_boxes_xyxy)} 个原始框。")

    _t_area_filter_start = time.time()
    if len(aggregated_boxes_xyxy) > 0 and ((min_area_cfg is not None and min_area_cfg > 0) or actual_max_area_threshold_px is not None): # 使用传入的参数
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

    # 3. OCR Preprocessing & Task Preparation
    t_start_step = time.time()
    tasks_for_ocr = []
    # ocr_input_metadata 用于存储每个YOLO框对应的裁剪信息和元数据，方便后续OCR结果与YOLO框关联
    ocr_input_metadata = [None] * len(aggregated_boxes_xyxy)
    if len(aggregated_boxes_xyxy) > 0:
        logger.info(f"--- 对 {len(aggregated_boxes_xyxy)} 个YOLO框进行OCR预处理 ---")
        for i, yolo_box_coords in enumerate(aggregated_boxes_xyxy): # i 即为 original_index
            class_id = int(aggregated_class_ids[i]); class_name = COCO_CLASSES[class_id]
            x1_y, y1_y, x2_y, y2_y = yolo_box_coords; h_y, w_y = y2_y-y1_y, x2_y-x1_y

            # 计算数字ROI的精确裁剪坐标 (与之前版本逻辑一致)
            y1_d_ideal = y1_y + int(h_y * DIGIT_ROI_Y_OFFSET_FACTOR); h_d_ideal = int(h_y * DIGIT_ROI_HEIGHT_FACTOR)
            y2_d_ideal = y1_d_ideal + h_d_ideal
            w_d_exp = int(w_y * DIGIT_ROI_WIDTH_EXPAND_FACTOR); cx_y = x1_y + w_y / 2.0
            x1_d_ideal = int(cx_y - w_d_exp / 2.0); x2_d_ideal = int(cx_y + w_d_exp / 2.0)
            y1_d_clip,y2_d_clip = max(0,y1_d_ideal),min(orig_img_h,y2_d_ideal)
            x1_d_clip,x2_d_clip = max(0,x1_d_ideal),min(orig_img_w,x2_d_ideal)

            ocr_input_metadata[i] = {"original_index": i, "roi_index": i + 1, "class": class_name,
                                     "bbox_yolo": yolo_box_coords,
                                     "bbox_digit_ocr_clipped": [x1_d_clip, y1_d_clip, x2_d_clip, y2_d_clip],
                                     "confidence_yolo": float(aggregated_scores[i])}
            img_for_ocr = None
            if x2_d_clip > x1_d_clip and y2_d_clip > y1_d_clip: # 确保裁剪区域有效
                digit_roi = original_image[y1_d_clip:y2_d_clip, x1_d_clip:x2_d_clip]
                h_roi, w_roi = digit_roi.shape[:2]
                if h_roi > 0 and w_roi > 0:
                    scale = TARGET_OCR_INPUT_HEIGHT / h_roi; target_w = int(w_roi * scale)
                    if target_w <= 0: target_w = 1 # 最小宽度为1
                    resized_roi = cv2.resize(digit_roi, (target_w, TARGET_OCR_INPUT_HEIGHT),
                                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
                    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img_for_ocr = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR) # PaddleOCR通常需要BGR

                    if SAVE_PROCESS_PHOTOS and img_for_ocr is not None:
                        ocr_slice_dir = os.path.join(PROCESS_PHOTO_DIR, "ocr_slices", session_id)
                        if not os.path.exists(ocr_slice_dir): os.makedirs(ocr_slice_dir, exist_ok=True)
                        slice_filename = f"s_idx{i}_roi{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
                        slice_output_path = os.path.join(ocr_slice_dir, slice_filename)
                        try: cv2.imwrite(slice_output_path, img_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        except Exception as e_save_slice: logger.error(f"保存OCR切片图失败 {slice_output_path}: {e_save_slice}")

            tasks_for_ocr.append((i, i + 1, img_for_ocr)) # i is the original_index for this YOLO box
    timing_profile['3_ocr_preprocessing_prep'] = time.time() - t_start_step

    # 4. Parallel OCR Processing & Result Consolidation
    t_start_step = time.time()
    # final_ocr_results_list 将存储每个YOLO框（按original_index顺序）的最终OCR信息
    final_ocr_results_list = [None] * len(aggregated_boxes_xyxy)
    # ocr_texts_for_drawing 用于旧的 annotated_*.jpg 绘图，可以考虑是否还需要
    ocr_texts_for_drawing = ["N/A"] * len(aggregated_boxes_xyxy)

    if tasks_for_ocr:
        logger.info(f"提交 {len(tasks_for_ocr)} 个OCR任务...")
        ocr_results_from_pool_indexed = [None] * len(tasks_for_ocr) # 存储按original_index排序的原始OCR结果

        global ocr_processing_pool, actual_num_ocr_workers # 使用全局的进程池
        if actual_num_ocr_workers > 1 and ocr_processing_pool:
            try:
                # map的输入是 tasks_for_ocr, 每个元素是 (original_index, roi_display_index, image_data)
                # ocr_task_for_worker 返回 (original_index, recognition_dict)
                pool_results = ocr_processing_pool.map(ocr_task_for_worker, tasks_for_ocr)
                for original_idx_from_res, res_dict in pool_results:
                    if 0 <= original_idx_from_res < len(ocr_results_from_pool_indexed):
                        ocr_results_from_pool_indexed[original_idx_from_res] = res_dict
                    else:
                        logger.warning(f"从OCR worker返回的original_index ({original_idx_from_res}) 无效。")
            except Exception as e_map:
                logger.error(f"OCR Pool map error: {e_map}\n{traceback.format_exc()}")
                # 可以在这里加入串行回退逻辑，或者让后续的整合步骤处理None值
        else: # Serial OCR processing
            logger.info("OCR串行处理")
            # ... (此处应包含完整的串行OCR处理逻辑，填充 ocr_results_from_pool_indexed)
            # (与您之前版本中的串行逻辑类似)
            _serial_ocr_predictor = None
            try:
                if not os.path.exists(SERVER_REC_MODEL_DIR_CFG_CONFIG): raise FileNotFoundError("Serial OCR model dir not found")
                _serial_ocr_predictor = pdx.inference.create_predictor(model_dir=SERVER_REC_MODEL_DIR_CFG_CONFIG, model_name='PP-OCRv5_server_rec', device='cpu')
                for task_original_idx, _, img_data_for_serial_ocr in tasks_for_ocr:
                    if img_data_for_serial_ocr is not None:
                        _res_gen_s = _serial_ocr_predictor.predict([img_data_for_serial_ocr])
                        _res_list_s = next(_res_gen_s, None)
                        ocr_results_from_pool_indexed[task_original_idx] = _res_list_s[0] if (_res_list_s and isinstance(_res_list_s, list) and len(_res_list_s) > 0) else (_res_list_s if isinstance(_res_list_s, dict) else {'rec_text': '', 'rec_score': 0.0})
                    else:
                        ocr_results_from_pool_indexed[task_original_idx] = {'rec_text': 'PREPROC_FAIL_SERIAL', 'rec_score': 0.0}
            except Exception as e_s_ocr: logger.error(f"Serial OCR error: {e_s_ocr}")
            finally:
                if _serial_ocr_predictor: del _serial_ocr_predictor


        # 整合OCR结果到 final_ocr_results_list
        for i, raw_ocr_dict in enumerate(ocr_results_from_pool_indexed): # i is original_index
            # 合并YOLO的元数据和OCR结果
            current_metadata = ocr_input_metadata[i] if i < len(ocr_input_metadata) else {}
            full_ocr_item = {**current_metadata} # 包含 bbox_yolo, confidence_yolo 等

            if raw_ocr_dict and isinstance(raw_ocr_dict, dict):
                ocr_text_raw = raw_ocr_dict.get('rec_text', "")
                ocr_score_raw = raw_ocr_dict.get('rec_score', 0.0)

                # 提取纯数字，并进行基本校验
                if ocr_text_raw and ocr_text_raw not in ['INIT_FAIL', 'PREDICT_FAIL', 'PREPROC_FAIL', 'WORKER_INIT_FAIL', 'PREPROC_FAIL_SERIAL']:
                    digits_only = "".join(re.findall(r'\d', ocr_text_raw))
                    # V5.3中，数据库校验将在 update_session_state_from_ocr 中进行
                    # 这里我们先保存提取的数字和原始置信度
                    full_ocr_item["ocr_final_text"] = digits_only if digits_only else "N/A_NO_DIGITS" # 如果过滤后没数字了
                    ocr_texts_for_drawing[i] = digits_only if digits_only else "ERR" # 用于旧的绘图
                else: # OCR失败或错误代码
                    full_ocr_item["ocr_final_text"] = ocr_text_raw # 保留错误码
                    ocr_texts_for_drawing[i] = "ERR"
                full_ocr_item["ocr_confidence"] = ocr_score_raw
            else: # raw_ocr_dict 为 None 或格式不对
                full_ocr_item["ocr_final_text"] = "N/A_OCR_RESULT_INVALID"
                ocr_texts_for_drawing[i] = "N/A"
                full_ocr_item["ocr_confidence"] = 0.0
            final_ocr_results_list[i] = full_ocr_item
    else:
        logger.info("无任务进行OCR")
        # 即使没有OCR任务，也要确保 final_ocr_results_list 与 aggregated_boxes_xyxy 长度一致，并填充N/A
        for i in range(len(aggregated_boxes_xyxy)):
            current_metadata = ocr_input_metadata[i] if i < len(ocr_input_metadata) else {}
            final_ocr_results_list[i] = {**current_metadata, "ocr_final_text": "N/A_NO_OCR_TASKS", "ocr_confidence": 0.0}

    timing_profile['4_ocr_processing_total'] = time.time() - t_start_step
    logger.info(f"OCR处理与结果整合完成 ({timing_profile['4_ocr_processing_total']:.3f}s)")

    # 准备YOLO结果给 refine_layout_and_map_yolo_to_logical
    # yolo_boxes_for_mapping 的每个元素应该是字典，包含 'cx','cy','w','h','box_yolo','score','original_index'
    yolo_boxes_for_mapping = []
    for i, yolo_box_item in enumerate(aggregated_boxes_xyxy): # 使用经过面积筛选的YOLO框
        cx, cy, w, h = get_box_center_and_dims(yolo_box_item)
        if cx is not None:
            yolo_boxes_for_mapping.append({
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'box_yolo': yolo_box_item,
                'score': aggregated_scores[i] if i < len(aggregated_scores) else 0.0,
                'original_index': i # 这个索引对应 aggregated_boxes_xyxy 和 final_ocr_results_list
            })

    # 步骤5: 解读YOLO结构并为当前帧YOLO锚点映射逻辑坐标
    t_start_step = time.time()
    # session 对象已在此函数开头获取
    # interpret_yolo_structure_and_map_to_logical 需要 image_wh 参数是 (宽度, 高度)
    current_frame_yolo_logical_map, layout_was_updated = interpret_yolo_structure_and_map_to_logical(
        yolo_boxes_for_mapping, # 传递包含 original_index 的YOLO框列表
        session_id,
        (orig_img_w, orig_img_h), # 传递图像宽度和高度
        logger
    )
    if layout_was_updated:
        logger.info(f"会话 {session_id}: 布局参数已通过 interpret_yolo_structure_and_map_to_logical 更新。")
    timing_profile['5_interpret_yolo_and_map'] = time.time() - t_start_step # 更新计时键名

    # 新增：如果SAVE_PROCESS_PHOTOS为True，则调用绘制YOLO逻辑位置映射图的函数
    if SAVE_PROCESS_PHOTOS and current_frame_yolo_logical_map: # 确保有映射结果才绘制
        t_start_draw_yolo_map = time.time()
        img_name_base_for_map = os.path.splitext(os.path.basename(image_path))[0]
        # session["frame_count"] 在 predict_image_route 中递增，这里直接使用
        draw_yolo_logical_placement_map(
            current_frame_yolo_logical_map,
            session["current_layout_config"], # 从会话中获取当前布局配置
            session_id,
            img_name_base_for_map,
            session.get("frame_count", 0), # 获取当前帧序号
            logger
        )
        timing_profile['5a_drawing_yolo_logical_map'] = time.time() - t_start_draw_yolo_map
    elif SAVE_PROCESS_PHOTOS:
        logger.info(f"会话 {session_id}: 无YOLO到逻辑的映射结果，不生成YOLO位置图。")

    # 6. 更新会话的状态矩阵和OBU证据池
    t_start_step = time.time()
    # 直接传递 yolo_boxes_for_mapping 作为 current_yolo_boxes_for_evidence
    update_session_state_from_ocr(
        session_id,
        current_frame_yolo_logical_map,
        final_ocr_results_list,
        yolo_boxes_for_mapping, # <--- 作为 current_yolo_boxes_for_evidence 传递
        logger                  # <--- logger 作为最后一个参数
    )
    timing_profile['6_update_session_state'] = time.time() - t_start_step

    # 7. (可选) 保存YOLO标注图 (用于调试YOLO和OCR的对应关系)
    if SAVE_PROCESS_PHOTOS and len(aggregated_boxes_xyxy) > 0:
        t_start_step = time.time()
        image_to_draw_on = original_image.copy()
        # 确保 roi_indices 列表的长度与 boxes, scores, class_ids 一致
        roi_indices_for_drawing = [item.get('roi_index') for item in final_ocr_results_list if item]
        if len(roi_indices_for_drawing) != len(aggregated_boxes_xyxy): # Fallback if lengths mismatch
            roi_indices_for_drawing = list(range(1, len(aggregated_boxes_xyxy) + 1))

        annotated_img = draw_detections(image_to_draw_on, np.array(aggregated_boxes_xyxy),
                                        np.array(aggregated_scores), np.array(aggregated_class_ids),
                                        COCO_CLASSES, ocr_texts=ocr_texts_for_drawing,
                                        roi_indices=roi_indices_for_drawing)
        img_name_base = os.path.splitext(os.path.basename(image_path))[0]
        ts_filename = datetime.now().strftime("%Y%m%d%H%M%S%f")
        annotated_path = os.path.join(PROCESS_PHOTO_DIR, f"annotated_{img_name_base}_{ts_filename}.jpg")
        try:
            cv2.imwrite(annotated_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, PROCESS_PHOTO_JPG_QUALITY])
            logger.info(f"YOLO标注图已保存: {annotated_path}")
        except Exception as e_save_ann:
            logger.error(f"保存YOLO标注图失败: {e_save_ann}")
        timing_profile['7_drawing_yolo_annotations'] = time.time() - t_start_step

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall_processing
    logger.info(f"--- Timing profile for {os.path.basename(image_path)} ({session_id}) ---")
    for stage_key in sorted(timing_profile.keys()):
        logger.info(f"  {stage_key}: {timing_profile[stage_key]:.3f}s")

    # 从会话中获取最终的逻辑矩阵和识别文本
    session_after_update = session_data_store.get(session_id, {}) # 获取最新的会话数据
    final_matrix_to_return = session_after_update.get("logical_status_matrix", [])
    final_texts_to_return = session_after_update.get("recognized_texts_map", {})

    return final_matrix_to_return, final_texts_to_return, timing_profile

# --- Flask Routes (修改 /predict 以返回JSON，并包含正确的会话初始化) ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger
    session_id_from_req = request.form.get('session_id', 'NO_SESSION_ID_IN_REQUEST') # 先获取一下，即使后面会强制

    logger.info(f"--- PREDICT ROUTE START for session_id: {session_id_from_req} ---")
    existing_session_preview = session_data_store.get(session_id_from_req)
    if existing_session_preview:
        logger.info(f"  Preview of existing session data keys: {list(existing_session_preview.keys())}")
    else:
        logger.info(f"  No existing session data found for this ID yet.")

    # 1. 获取 session_id (强制要求客户端提供)
    if 'session_id' not in request.form:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400
    session_id = request.form.get('session_id')

    # 2. 获取当前会话应使用的布局配置
    # (未来扩展: 此处可以根据客户端传递的 box_type 参数来选择不同的 LAYOUT_CONFIG)
    # 目前，我们全局只有一个 LAYOUT_CONFIG
    current_layout_config_for_session = LAYOUT_CONFIG # LAYOUT_CONFIG 是在脚本顶部定义的全局常量

    # 3. 文件检查
    if 'file' not in request.files:
        logger.warning(f"会话 {session_id}: 请求中未找到文件部分。")
        return jsonify({"error": "No file part in the request", "session_id": session_id}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning(f"会话 {session_id}: 未选择文件。")
        return jsonify({"error": "No selected file", "session_id": session_id}), 400

    if not (file and allowed_file(file.filename)): # allowed_file 是您的辅助函数
        logger.warning(f"会话 {session_id}: 文件类型不允许: {file.filename}")
        return jsonify({"error": "File type not allowed", "session_id": session_id}), 400

    original_filename_for_exc = "N/A" # 用于在异常块中记录文件名

    try:
        original_filename = secure_filename(file.filename)
        original_filename_for_exc = original_filename # 更新以便在异常时使用
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename)
        filename = f"{name}_{timestamp}{ext}"

        upload_dir = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        logger.info(f"会话 {session_id}: 文件 '{filename}' 已成功保存到 '{filepath}'")

        # 4. 确保ONNX会话已加载
        global onnx_session
        if onnx_session is None:
            logger.error(f"会话 {session_id}: ONNX session 未初始化!")
            return jsonify({"error": "ONNX session not initialized on server", "session_id": session_id}), 500

        # 5. 获取或初始化会话数据
        session = session_data_store.get(session_id)
        if not session:
            logger.info(f"会话 {session_id}: 新建会话。")
            # 初始化逻辑状态矩阵
        session = session_data_store.get(session_id)
        if not session:
            logger.info(f"会话 {session_id}: 新建会话。")
            # 初始化逻辑状态矩阵
            initial_matrix = [[0] * current_layout_config_for_session["regular_cols_count"]
                              for _ in range(current_layout_config_for_session["expected_total_rows"])]

            special_row_idx = current_layout_config_for_session["expected_total_rows"] - 1
            if current_layout_config_for_session["expected_total_rows"] > 0 and \
               current_layout_config_for_session["regular_cols_count"] == 4 and \
               current_layout_config_for_session["special_row_cols_count"] == 2:
                if 0 <= special_row_idx < len(initial_matrix):
                    initial_matrix[special_row_idx][0] = -1
                    initial_matrix[special_row_idx][3] = -1

            session_data_store[session_id] = {
                "logical_status_matrix": initial_matrix,       # <--- 核心：确保这里被正确初始化
                "recognized_texts_map": {},                  # <--- 确保这里也被正确初始化
                "obu_evidence_pool": {},
                "layout_parameters": {
                    "row_y_estimates": [],
                    "col_x_estimates_regular": [],
                    "avg_obu_w": 100, "avg_obu_h": 40,
                    "is_calibrated": False,
                    "avg_physical_row_height": 50,
                    "avg_col_spacing": 100,
                    "special_row_at_logical_top": False
                },
                "current_layout_config": current_layout_config_for_session,
                "frame_count": 0,
                "last_activity": datetime.now()
            }

        session = session_data_store.get(session_id) # 重新获取，确保拿到的是创建后的session对象
        session["frame_count"] = session.get("frame_count", 0) + 1
        session["last_activity"] = datetime.now()


        # 6. 从 app.config 获取面积筛选参数
        min_area_cfg_val = current_app.config.get('MIN_DETECTION_AREA_CFG', 2000)
        max_area_cfg_val = current_app.config.get('MAX_DETECTION_AREA_CFG', 0.1)

        # 7. 调用核心处理逻辑
        # process_image_with_ocr_logic 现在需要 session 对象来直接修改其内容，或者返回需要更新的部分
        # 为了保持函数接口的清晰，让 process_image_with_ocr_logic 返回更新后的矩阵和文本
        logical_matrix_result, recognized_texts_result, timings = process_image_with_ocr_logic(
            filepath,
            onnx_session,
            session_id, # 传递 session_id 以便 process_image_with_ocr_logic 内部可以获取和更新会话数据
            current_layout_config_for_session, # 传递当前会话的布局配置
            min_area_cfg_val,
            max_area_cfg_val
        )

        # 8. 准备并返回JSON响应
        response_data = {
            "message": "File processed successfully.",
            "session_id": session_id,
            "received_filename": original_filename,
            "obu_status_matrix": logical_matrix_result,
            "obu_texts": {f"{r}_{c}": text for (r,c), text in recognized_texts_result.items()}, # 将元组key转为字符串
            "timing_profile_seconds": timings,
        }

        # 检查会话是否已满 (基于识别出的文本数量)
        num_identified_successfully = 0
        for r_idx in range(len(logical_matrix_result)):
            for c_idx in range(len(logical_matrix_result[r_idx])):
                if logical_matrix_result[r_idx][c_idx] == 1: # 状态1代表成功识别
                    num_identified_successfully +=1

        total_expected_obus = current_layout_config_for_session.get("total_obus", 50) # 从布局配置获取总数
        if num_identified_successfully >= total_expected_obus:
            response_data["session_status"] = "completed"
            logger.info(f"会话 {session_id}: 所有 {total_expected_obus} 个OBU已识别，会话完成。")
        else:
            response_data["session_status"] = "in_progress"
            logger.info(f"会话 {session_id}: 已识别 {num_identified_successfully}/{total_expected_obus} 个OBU。")


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