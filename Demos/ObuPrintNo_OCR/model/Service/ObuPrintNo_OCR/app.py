# app.py
import os
import cv2
import numpy as np
import time
import traceback
import multiprocessing
from datetime import datetime
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import atexit
import logging
from logging.handlers import RotatingFileHandler
import uuid
from typing import List, Dict, Tuple, Any, Optional
import base64 # 用于图像的Base64编码
import re # 导入re模块
from waitress import serve # 导入生产级服务器

# --- 从新模块导入 ---
import config
from image_utils import read_image_cv2, draw_ocr_results_on_image # 导入新的绘图函数
from yolo_handler import YoloHandler
from ocr_handler import OcrHandler
from layout_and_state_manager import LayoutStateManager

# --- 全局变量 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

yolo_predictor: Optional[YoloHandler] = None
ocr_predictor: Optional[OcrHandler] = None
layout_state_mgr: Optional[LayoutStateManager] = None
session_data_store: Dict[str, Any] = {}

# --- 日志设置 ---
def setup_logging(app_instance):
    if not os.path.exists(config.LOG_DIR):
        try: os.makedirs(config.LOG_DIR)
        except OSError as e: print(f"Error creating log directory {config.LOG_DIR}: {e}")
    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == file_handler.baseFilename for h in app_instance.logger.handlers):
        app_instance.logger.addHandler(file_handler)
    app_instance.logger.setLevel(logging.INFO)
    app_instance.logger.info(f"Flask应用日志系统已启动。版本: {config.APP_VERSION}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# --- 最终版专家纠错函数 (带决胜局逻辑) ---
def find_best_match_by_mask(
    ocr_text: str,
    valid_codes: set,
    mask: str,
    threshold: int
) -> Optional[str]:
    """
    使用掩码和汉明距离，在有效OBU码中为OCR结果寻找最可靠的匹配。
    增加了决胜局（Tie-Breaker）逻辑来处理多个候选项。
    """
    if len(ocr_text) != 16 or len(mask) != 16:
        return None

    candidate_matches = []
    min_mask_distance = threshold + 1

    # 步骤1: 找出所有在掩码固定位上足够相似的候选者
    for valid_code in valid_codes:
        # 健壮性检查：确保valid_code也符合掩码的固定位，避免脏数据干扰
        is_candidate = True
        for vc, mc in zip(valid_code, mask):
            if mc != '_' and vc != mc:
                is_candidate = False
                break
        if not is_candidate:
            continue

        distance = 0
        for ocr_char, valid_char, mask_char in zip(ocr_text, valid_code, mask):
            if mask_char != '_' and ocr_char != valid_char:
                distance += 1

        if distance < min_mask_distance:
            min_mask_distance = distance
            candidate_matches = [valid_code]
        elif distance == min_mask_distance:
            candidate_matches.append(valid_code)

    # 步骤2: 根据候选者数量进行决策
    if not candidate_matches or min_mask_distance > threshold:
        return None # 没有找到任何在阈值内的匹配

    if len(candidate_matches) == 1:
        return candidate_matches[0] # 只有一个最佳匹配，直接返回

    # 步骤3: 决胜局！在多个候选项中，选择全局汉明距离最小的那个
    min_global_distance = 17 # 大于16即可
    final_best_match = None
    found_tie_in_global = False

    for code in candidate_matches:
        global_distance = sum(1 for c1, c2 in zip(ocr_text, code) if c1 != c2)
        if global_distance < min_global_distance:
            min_global_distance = global_distance
            final_best_match = code
            found_tie_in_global = False
        elif global_distance == min_global_distance:
            found_tie_in_global = True

    if not found_tie_in_global:
        return final_best_match
    else:
        # 如果在决胜局中依然存在平局，为了绝对安全，放弃纠错
        return None

# --- 核心图像处理逻辑 (V4.0_FINAL_PERFECT_SYSTEM) ---
def process_image_with_ocr_logic(
    image_path: str,
    session_id: str,
    logger: Any,
    mode: str = 'scattered_cumulative_ocr',
) -> Tuple[Optional[List[List[int]]], Optional[Dict[Any, Any]], Dict[str, float], List[Dict[str, str]], Optional[List[Dict[str,Any]]], Optional[str]]:
    log_prefix = f"会话 {session_id} (process_image V{config.APP_VERSION}_ScatterFocus M:{mode}):"
    logger.info(f"{log_prefix} 开始处理图片 {os.path.basename(image_path)}")

    timing_profile = {}
    t_start_overall = time.time()
    warnings_list = []
    scattered_results_list: Optional[List[Dict[str, Any]]] = None
    annotated_image_base64_str: Optional[str] = None

    # --- 初始化默认返回值 ---
    session_obj_for_init = session_data_store.get(session_id)
    current_layout_config_for_init = config
    if session_obj_for_init and "current_layout_config" in session_obj_for_init:
        current_layout_config_for_init = session_obj_for_init["current_layout_config"]
    expected_rows_for_empty = getattr(current_layout_config_for_init, "LAYOUT_EXPECTED_TOTAL_ROWS", config.LAYOUT_EXPECTED_TOTAL_ROWS)
    expected_cols_for_empty = getattr(current_layout_config_for_init, "LAYOUT_REGULAR_COLS_COUNT", config.LAYOUT_REGULAR_COLS_COUNT)
    default_empty_matrix = [[0] * expected_cols_for_empty for _ in range(expected_rows_for_empty)]
    final_matrix: Optional[List[List[int]]] = default_empty_matrix
    final_texts: Optional[Dict[Tuple[int, int], str]] = {}
    # --- 结束返回值初始化 ---

    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"{log_prefix} 严重错误 - 未找到会话数据（在处理中丢失）！")
        if mode == 'full_layout':
            return default_empty_matrix, {}, {"error": "Session data lost"}, [{"message": "会话数据丢失。"}], None, None
        else:
            return None, None, {"error": "Session data lost"}, [{"message": "会话数据丢失。"}], [], None


    global yolo_predictor, ocr_predictor, layout_state_mgr
    if not yolo_predictor or not ocr_predictor:
        logger.critical(f"{log_prefix} YOLO或OCR核心处理器未初始化！")
        return final_matrix, final_texts, {"error": "Server not ready"}, [{"message": "服务内部错误(YOLO/OCR)。"}], None, None
    if mode == 'full_layout' and not layout_state_mgr:
        logger.critical(f"{log_prefix} 整版模式下LayoutStateManager未初始化！")
        return final_matrix, final_texts, {"error": "Server not ready"}, [{"message": "服务内部错误(LSM)。"}], None, None

    try:
        if "frame_count" not in session: session["frame_count"] = 0
        session["frame_count"] += 1
        current_frame_num = session["frame_count"]
        session["last_activity"] = datetime.now()
        if "status_flags" not in session: session["status_flags"] = {}
        session["status_flags"]["frame_skipped_due_to_no_overlap"] = False
        session["status_flags"]["is_first_frame_anchor_failed"] = False


        t_step = time.time()
        original_image = read_image_cv2(image_path)
        img_h, img_w = original_image.shape[:2]
        timing_profile['1_image_reading'] = time.time() - t_step
        logger.info(f"{log_prefix} 原始图片: {os.path.basename(image_path)} (H={img_h}, W={img_w})")

        t_step = time.time()
        yolo_detections = yolo_predictor.detect(original_image)
        timing_profile['2_yolo_detection'] = time.time() - t_step
        logger.info(f"{log_prefix} YOLO检测完成，找到 {len(yolo_detections)} 个有效框。")

        if config.SAVE_PROCESS_PHOTOS and yolo_detections:
            try:
                yolo_raw_img = draw_yolo_detections_on_image(original_image, yolo_detections, None, config.YOLO_COCO_CLASSES)
                img_name_base_raw = os.path.splitext(os.path.basename(image_path))[0]
                ts_filename_raw = datetime.now().strftime("%Y%m%d%H%M%S%f")
                yolo_raw_dir = os.path.join(config.PROCESS_PHOTO_DIR, "yolo_raw")
                if not os.path.exists(yolo_raw_dir): os.makedirs(yolo_raw_dir, exist_ok=True)
                yolo_raw_path = os.path.join(yolo_raw_dir, f"yolo_raw_{img_name_base_raw}_s{session_id[:8]}_f{current_frame_num}_{ts_filename_raw}.jpg")
                cv2.imwrite(yolo_raw_path, yolo_raw_img, [cv2.IMWRITE_JPEG_QUALITY, config.PROCESS_PHOTO_JPG_QUALITY])
            except Exception as e_save_yolo_raw:
                logger.error(f"{log_prefix} 保存纯YOLO检测结果图失败: {e_save_yolo_raw}", exc_info=True)

        t_step = time.time()
        ocr_tasks_for_pool, ocr_input_metadata = ocr_predictor.prepare_ocr_tasks_from_detections(
            original_image, yolo_detections, session_id, current_frame_num, config.SAVE_TRAINING_ROI_IMAGES
        )
        raw_ocr_pool_results = ocr_predictor.recognize_prepared_tasks(ocr_tasks_for_pool)
        final_ocr_results_list = ocr_predictor.consolidate_ocr_results(raw_ocr_pool_results, ocr_input_metadata)
        timing_profile['3_ocr_processing'] = time.time() - t_step
        logger.info(f"{log_prefix} OCR处理完成，得到 {len(final_ocr_results_list)} 条结果。")

        if mode == 'full_layout':
            logger.info(f"{log_prefix} 进入“整版识别”模式处理流程。")
            t_step_layout = time.time()
            stable_layout_params_from_session = session.get("stable_layout_parameters")
            session_config = session["current_layout_config"]

            current_frame_layout_stats, yolo_detections_with_rc = layout_state_mgr.analyze_frame_layout_and_get_params(
                yolo_detections, (img_w, img_h), session_config, session_id
            )

            if not current_frame_layout_stats:
                logger.error(f"{log_prefix} (整版模式) 当前帧布局分析失败。")
                warnings_list.append({"message": "警告：当前帧布局分析失败。", "code": "FRAME_LAYOUT_ANALYSIS_FAILED"}); current_frame_layout_stats = {}
            if not yolo_detections_with_rc and yolo_detections:
                logger.warning(f"{log_prefix} (整版模式) XY聚类未返回带rc的检测结果。"); yolo_detections_with_rc = [dict(d, frame_r=-1, frame_c=-1) for d in yolo_detections]

            if not stable_layout_params_from_session and current_frame_layout_stats:
                logger.info(f"{log_prefix} (整版模式) 会话首次有效帧，使用当前帧统计结果作为初始稳定布局参考。")
                session["stable_layout_parameters"] = current_frame_layout_stats.copy(); session["layout_parameters"]["is_calibrated"] = True
                stable_layout_params_from_session = session["stable_layout_parameters"]

            if not stable_layout_params_from_session:
                logger.error(f"{log_prefix} (整版模式) 无法获取有效的稳定布局参数！")
                stable_layout_params_from_session = {
                    "median_obu_w_stable": 100, "median_obu_h_stable": 40, "avg_physical_row_height_stable": 60,
                    "row_y_means_from_clustering": {}, "col_x_means_from_clustering": {},
                    "special_row_at_logical_top": False, "identified_special_row_frame_r": -1}
                session["stable_layout_parameters"] = stable_layout_params_from_session

            current_frame_verified_obus_for_anchor = []
            map_yolo_idx_to_ocr_item_full = {ocr.get("original_index"): ocr for ocr in final_ocr_results_list if ocr}
            for det_rc_full in yolo_detections_with_rc:
               original_yolo_idx_full = det_rc_full["original_index"]
               ocr_item_full = map_yolo_idx_to_ocr_item_full.get(original_yolo_idx_full)
               if ocr_item_full and ocr_item_full.get("ocr_final_text") in config.VALID_OBU_CODES:
                   yolo_details_full = ocr_item_full.get("yolo_anchor_details")
                   if yolo_details_full and det_rc_full.get("frame_r", -1) != -1 :
                       current_frame_verified_obus_for_anchor.append({
                           "text": ocr_item_full["ocr_final_text"], "physical_anchor": yolo_details_full,
                           "ocr_confidence": ocr_item_full.get("ocr_confidence", 0.0),
                           "original_yolo_idx": original_yolo_idx_full, "frame_r": det_rc_full["frame_r"]})

            y_anchor_info, estimated_row_y_for_drawing = layout_state_mgr.determine_y_axis_anchor(
               current_frame_verified_obus_for_anchor, session["obu_evidence_pool"],
               current_frame_layout_stats, session_id, current_frame_num, session_config)
            session["status_flags"]["y_anchor_info_current_frame"] = y_anchor_info

            if current_frame_num == 1 and y_anchor_info.get("is_first_frame_anchor_failed", False):
                logger.error(f"{log_prefix} (整版模式)【行政规定检查失败】第一帧未能通过特殊行成功锚定！")
                warnings_list.append({
                    "message": "错误(整版模式)：第一帧未能识别到近端特殊行或锚定失败。",
                    "code": "FIRST_FRAME_ANCHOR_FAILED_ADMIN_RULE"
                })
                timing_profile['4_layout_and_state'] = time.time() - t_step_layout
                return final_matrix, final_texts, timing_profile, warnings_list, None, None

            if y_anchor_info.get("is_skipped_due_to_no_overlap"):
                warnings_list.append({ "message": "检测到可能的拍摄跳跃或漏帧(整版模式)。", "code": "FRAME_SKIPPED_NO_OVERLAP"})

            session["layout_parameters"]["row_y_estimates"] = estimated_row_y_for_drawing

            if config.SAVE_PROCESS_PHOTOS:
                try:
                    layout_state_mgr.draw_stable_layout_on_image(
                        yolo_detections_with_rc, (img_w, img_h), session_id,
                        current_frame_num, y_anchor_info, current_frame_layout_stats)
                except Exception as e_draw_layout:
                    logger.critical(f"{log_prefix} (整版模式) CRITICAL_LOG: 绘制单帧布局图时发生严重错误: {e_draw_layout}", exc_info=True)

            if not y_anchor_info.get("is_skipped_due_to_no_overlap", False):
                matrix_updated, texts_updated, warnings_state = layout_state_mgr.update_session_state_with_reference_logic(
                   session, yolo_detections_with_rc, final_ocr_results_list,
                   y_anchor_info, session_id, current_frame_num, session_config)
                if warnings_state: warnings_list.extend(warnings_state)
                final_matrix = matrix_updated; final_texts = texts_updated
            else:
                logger.warning(f"{log_prefix} (整版模式) 因漏帧，跳过核心状态更新。返回当前会话状态。")
                final_matrix = session.get("logical_matrix", final_matrix)
                final_texts = session.get("recognized_texts_map", final_texts)
            timing_profile['4_layout_and_state'] = time.time() - t_step_layout
            logger.info(f"{log_prefix} “整版识别”模式处理流程完成。")

        else: # scattered_cumulative_ocr 模式
            logger.info(f"{log_prefix} 执行“累积式零散识别”流程。")
            t_step_scatter = time.time()

            accumulated_obu_texts_set = set(session.get("accumulated_obu_texts_set", []))

            current_frame_recognized_count = 0
            final_recognized_texts_this_frame = set()

            for ocr_item in final_ocr_results_list:
                raw_text = ocr_item.get("ocr_final_text", "").strip()

                candidate_text = raw_text
                if config.OCR_HEURISTIC_REPLACEMENTS:
                    for char_to_replace, replacement in config.OCR_HEURISTIC_REPLACEMENTS.items():
                        candidate_text = candidate_text.replace(char_to_replace, replacement)

                candidate_text = "".join(re.findall(r'\d', candidate_text))
                if len(candidate_text) != 16:
                    continue

                if candidate_text in config.VALID_OBU_CODES:
                    final_recognized_texts_this_frame.add(candidate_text)
                    ocr_item['final_corrected_text'] = candidate_text
                    continue

                if config.ENABLE_OCR_CORRECTION:
                    corrected_text = find_best_match_by_mask(
                        candidate_text,
                        config.VALID_OBU_CODES,
                        config.OCR_CORRECTION_MASK,
                        config.OCR_CORRECTION_HAMMING_THRESHOLD
                    )

                    if corrected_text:
                        final_recognized_texts_this_frame.add(corrected_text)
                        ocr_item['final_corrected_text'] = corrected_text
                        logger.info(f"掩码纠错成功: 将 '{candidate_text}' (原始: '{raw_text}') 修正为 '{corrected_text}'")

            for text in final_recognized_texts_this_frame:
                if text not in accumulated_obu_texts_set:
                    current_frame_recognized_count += 1
                accumulated_obu_texts_set.add(text)

            session["accumulated_obu_texts_set"] = accumulated_obu_texts_set
            logger.info(f"{log_prefix} “累积式零散识别”：当前帧有效识别 {current_frame_recognized_count} 个, "
                        f"累积总数 {len(accumulated_obu_texts_set)} 个。")

            if config.SAVE_TRAINING_ROI_IMAGES:
                label_file_path = os.path.join(config.PROCESS_PHOTO_DIR, "training_rois", session_id, "label.txt")
                with open(label_file_path, 'a', encoding='utf-8') as f:
                    for ocr_item in final_ocr_results_list:
                        corrected_text = ocr_item.get('final_corrected_text')
                        if corrected_text:
                            roi_filename = f"f{current_frame_num}_yolo{ocr_item['original_index']}_h48_w320.png"
                            f.write(f"{roi_filename}\t{corrected_text}\n")

            if config.SAVE_PROCESS_PHOTOS:
                annotated_img_full_size = draw_ocr_results_on_image(
                    original_image, yolo_detections, final_ocr_results_list, config.VALID_OBU_CODES
                )
                if config.SAVE_SCATTERED_ANNOTATED_IMAGE:
                    annotated_dir = os.path.join(config.PROCESS_PHOTO_DIR, "scattered_annotated")
                    if not os.path.exists(annotated_dir): os.makedirs(annotated_dir, exist_ok=True)
                    annotated_path = os.path.join(annotated_dir, f"annotated_s{session_id[:8]}_f{current_frame_num}.jpg")
                    cv2.imwrite(annotated_path, annotated_img_full_size, [cv2.IMWRITE_JPEG_QUALITY, 85])

                try:
                    target_w = config.SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH
                    orig_h_ann, orig_w_ann = annotated_img_full_size.shape[:2]
                    scale_ann = target_w / orig_w_ann
                    target_h_ann = int(orig_h_ann * scale_ann)
                    resized_annotated_img = cv2.resize(annotated_img_full_size, (target_w, target_h_ann),
                                                       interpolation=cv2.INTER_AREA if scale_ann < 1 else cv2.INTER_LINEAR)
                    retval, buffer = cv2.imencode('.jpg', resized_annotated_img,
                                                  [cv2.IMWRITE_JPEG_QUALITY, config.SCATTERED_MODE_IMAGE_JPG_QUALITY])
                    if retval:
                        annotated_image_base64_str = base64.b64encode(buffer).decode('utf-8')
                except Exception as e_scatter_draw:
                    logger.error(f"{log_prefix} 生成或编码缩小标注图时发生错误: {e_scatter_draw}", exc_info=True)

            scattered_results_list = [{"text": obu} for obu in sorted(list(accumulated_obu_texts_set))]
            final_matrix = None; final_texts = None
            timing_profile['4_scattered_processing'] = time.time() - t_step_scatter
            logger.info(f"{log_prefix} “累积式零散识别”模式处理流程完成。")

        if config.SAVE_PROCESS_PHOTOS and yolo_detections and mode == 'full_layout' and final_texts is not None:
            t_step_draw_yolo_final = time.time()
            ocr_texts_for_final_draw = {
                ocr_item.get("original_index"): ocr_item.get("ocr_final_text")
                for ocr_item in final_ocr_results_list
                if ocr_item and ocr_item.get("ocr_final_text") in config.VALID_OBU_CODES
            }
            annotated_img_final = draw_yolo_detections_on_image(
                original_image, yolo_detections, ocr_texts_for_final_draw, config.YOLO_COCO_CLASSES)
            img_name_base_final = os.path.splitext(os.path.basename(image_path))[0]
            ts_filename_final = datetime.now().strftime("%Y%m%d%H%M%S%f")
            annotated_path_final = os.path.join(config.PROCESS_PHOTO_DIR, f"annotated_{img_name_base_final}_s{session_id[:8]}_f{current_frame_num}_{ts_filename_final}.jpg")
            try:
                cv2.imwrite(annotated_path_final, annotated_img_final, [cv2.IMWRITE_JPEG_QUALITY, config.PROCESS_PHOTO_JPG_QUALITY])
            except Exception as e_save_ann_final: logger.error(f"{log_prefix} 保存最终YOLO标注图失败: {e_save_ann_final}", exc_info=True)
            timing_profile['5_drawing_final_annotation'] = time.time() - t_step_draw_yolo_final

    except Exception as e:
        logger.error(f"{log_prefix} 处理图片时发生未知严重错误: {e}", exc_info=True)
        warnings_list.append({"message": f"服务内部错误，处理失败: {str(e)}", "code": "INTERNAL_SERVER_ERROR"})

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall
    logger.info(f"{log_prefix} --- Timing profile for {os.path.basename(image_path)} ---")
    for key, val in sorted(timing_profile.items()): logger.info(f"  {key}: {val:.3f}s")

    return final_matrix, final_texts, timing_profile, warnings_list, scattered_results_list, annotated_image_base64_str

# app.py (Continued - Part 3 - Final)

# --- Flask 路由 ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger

    session_id = request.form.get('session_id')
    if not session_id:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400

    mode = request.form.get('mode', 'scattered_cumulative_ocr').lower()
    if mode not in ['full_layout', 'scattered_cumulative_ocr']:
        logger.warning(f"会话 {session_id}: 无效的处理模式 '{mode}'，将使用默认 'scattered_cumulative_ocr'。")
        mode = 'scattered_cumulative_ocr'

    force_recalibrate_str = request.form.get('force_recalibrate', 'false').lower()
    force_recalibrate = (force_recalibrate_str == 'true') if mode == 'full_layout' else False

    if 'file' not in request.files:
        logger.warning(f"会话 {session_id}: 请求中未找到文件部分。")
        return jsonify({"error": "No file part in the request", "session_id": session_id}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning(f"会话 {session_id}: 未选择文件。")
        return jsonify({"error": "No selected file", "session_id": session_id}), 400
    if not (file and allowed_file(file.filename)):
        logger.warning(f"会话 {session_id}: 文件类型不允许: {file.filename}")
        return jsonify({"error": "File type not allowed", "session_id": session_id}), 400

    original_filename_for_log = secure_filename(file.filename)

    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename_for_log)

        session = session_data_store.get(session_id)
        is_new_session_or_recalibrate = False

        if not session or (mode == 'full_layout' and force_recalibrate):
            is_new_session_or_recalibrate = True
            log_msg_session = f"会话 {session_id} (模式: {mode}): "
            if mode == 'full_layout' and force_recalibrate and session:
                log_msg_session += "用户触发强制重新校准。"
            else:
                log_msg_session += "新建会话或首次处理。"
            logger.info(log_msg_session)

            current_layout_config_for_session = {
                "expected_total_rows": config.LAYOUT_EXPECTED_TOTAL_ROWS,
                "regular_rows_count": config.LAYOUT_REGULAR_ROWS_COUNT,
                "regular_cols_count": config.LAYOUT_REGULAR_COLS_COUNT,
                "special_row_cols_count": config.LAYOUT_SPECIAL_ROW_COLS_COUNT,
                "total_obus": config.LAYOUT_TOTAL_OBUS_EXPECTED
            }
            initial_matrix = [[0] * current_layout_config_for_session["regular_cols_count"]
                              for _ in range(current_layout_config_for_session["expected_total_rows"])]

            session_data_store[session_id] = {
                "accumulated_obu_texts_set": set(),
                "logical_matrix": initial_matrix,
                "recognized_texts_map": {},
                "obu_evidence_pool": {},
                "layout_parameters": {"is_calibrated": False, "row_y_estimates": [] },
                "stable_layout_parameters": None,
                "current_layout_config": current_layout_config_for_session,
                "frame_count": 0,
                "last_activity": datetime.now(),
                "status_flags": {
                    "frame_skipped_due_to_no_overlap": False,
                    "y_anchor_info_current_frame": None,
                }}
        session = session_data_store[session_id]

        frame_num_for_filename = session.get("frame_count",0) + 1
        filename_on_server = f"s{session_id[:8]}_f{frame_num_for_filename}_{name}_{timestamp}{ext}"
        upload_path = os.path.join(config.UPLOAD_FOLDER, filename_on_server)
        if not os.path.exists(config.UPLOAD_FOLDER):
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        file.save(upload_path)
        logger.info(f"会话 {session_id}: 文件 '{filename_on_server}' 已成功保存到 '{upload_path}'")


        matrix_res, texts_res, timings_res, warnings_res, scattered_res, base64_img_res = \
            process_image_with_ocr_logic(
                upload_path, session_id, logger, mode=mode
            )

        response_data = {
            "message": "File processed successfully.",
            "session_id": session_id,
            "received_filename": original_filename_for_log,
            "mode_processed": mode,
            "timing_profile_seconds": timings_res,
            "warnings": warnings_res
        }

        current_session_status = "unknown"
        session_after_process = session_data_store.get(session_id)

        if mode == 'full_layout':
            response_data["obu_status_matrix"] = matrix_res
            response_data["obu_texts"] = {f"{r}_{c}": text for (r,c), text in texts_res.items()} if texts_res else {}

            num_identified = sum(1 for r_val in matrix_res for status in r_val if status == 1) if matrix_res else 0
            total_expected = config.LAYOUT_TOTAL_OBUS_EXPECTED
            if session_after_process and "current_layout_config" in session_after_process:
                 total_expected = session_after_process["current_layout_config"].get("total_obus", total_expected)

            y_anchor_info_final = session_after_process.get("status_flags",{}).get("y_anchor_info_current_frame") if session_after_process else None
            current_frame_processed_num = session_after_process.get("frame_count", 0) if session_after_process else 0

            if current_frame_processed_num == 1 and y_anchor_info_final and y_anchor_info_final.get("is_first_frame_anchor_failed", False) :
                current_session_status = "first_frame_anchor_error"
            elif num_identified >= total_expected:
                current_session_status = "completed"
            else:
                current_session_status = "in_progress"
            response_data["session_status"] = current_session_status
            logger.info(f"会话 {session_id} (整版模式): 已识别 {num_identified}/{total_expected} 个OBU。状态: {current_session_status}")

        elif mode == 'scattered_cumulative_ocr':
            response_data["accumulated_results"] = scattered_res
            response_data["current_frame_annotated_image_base64"] = base64_img_res
            current_session_status = "scattered_recognition_in_progress"
            response_data["session_status"] = current_session_status
            logger.info(f"会话 {session_id} (累积零散模式): 返回 {len(scattered_res if scattered_res else [])} 个累积OBU。")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_log}' 时发生严重错误: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- 应用初始化与启动 ---
def initialize_global_handlers(app_logger):
    global yolo_predictor, ocr_predictor, layout_state_mgr
    app_logger.info("--- 开始初始化全局处理器 ---")
    try:
        yolo_predictor = YoloHandler(
            model_path=config.ONNX_MODEL_PATH,
            conf_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=config.YOLO_IOU_THRESHOLD,
            min_area_px=config.YOLO_MIN_DETECTION_AREA_PX,
            max_area_factor=config.YOLO_MAX_DETECTION_AREA_FACTOR,
            coco_classes=config.YOLO_COCO_CLASSES,
            logger=app_logger)
        ocr_predictor = OcrHandler(
            onnx_model_path=config.OCR_ONNX_MODEL_PATH,
            keys_path=config.OCR_KEYS_PATH,
            num_workers=config.OCR_NUM_WORKERS,
            target_ocr_input_height=config.OCR_TARGET_INPUT_HEIGHT,
            digit_roi_y_offset_factor=config.OCR_DIGIT_ROI_Y_OFFSET_FACTOR,
            digit_roi_height_factor=config.OCR_DIGIT_ROI_HEIGHT_FACTOR,
            digit_roi_width_expand_factor=config.OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR,
            logger=app_logger)
        layout_state_mgr = LayoutStateManager(
            config_params={
                "LAYOUT_EXPECTED_TOTAL_ROWS": config.LAYOUT_EXPECTED_TOTAL_ROWS,
                "LAYOUT_REGULAR_COLS_COUNT": config.LAYOUT_REGULAR_COLS_COUNT,
                "LAYOUT_SPECIAL_ROW_COLS_COUNT": config.LAYOUT_SPECIAL_ROW_COLS_COUNT,
                "LAYOUT_MIN_CORE_ANCHORS_FOR_STATS": config.LAYOUT_MIN_CORE_ANCHORS_FOR_STATS,
                "LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD": config.LAYOUT_Y_AXIS_GROUPING_PIXEL_THRESHOLD,
                "LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD": config.LAYOUT_X_AXIS_GROUPING_PIXEL_THRESHOLD,
                "PROCESS_PHOTO_DIR": config.PROCESS_PHOTO_DIR,
                "IMAGE_FALLBACK_HEIGHT_FOR_LAYOUT": 5712,
                "avg_obu_h": 40.0,
                "avg_obu_w": 100.0
            }, logger=app_logger)
        app_logger.info("--- 全局处理器初始化完成 ---")
    except Exception as e:
        app_logger.critical(f"全局处理器初始化失败: {e}", exc_info=True)
        raise

def cleanup_ocr_pool_on_exit():
    global ocr_predictor
    if ocr_predictor and hasattr(ocr_predictor, 'close_pool'):
        print("应用退出，正在关闭OCR处理池...")
        ocr_predictor.close_pool()
        print("OCR处理池已关闭。")

if __name__ == '__main__':
    setup_logging(app)
    try:
        initialize_global_handlers(app.logger)
    except Exception as e_init:
        app.logger.critical(f"应用启动失败，无法初始化核心处理器: {e_init}")
        exit(1)
    atexit.register(cleanup_ocr_pool_on_exit)
    if not os.path.exists(config.UPLOAD_FOLDER): os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(config.PROCESS_PHOTO_DIR): os.makedirs(config.PROCESS_PHOTO_DIR, exist_ok=True)
    app.logger.info(f"服务版本 {config.APP_VERSION} 启动中... 监听 0.0.0.0:5000")
    app.logger.info(f"过程图片保存开关 (SAVE_PROCESS_PHOTOS): {config.SAVE_PROCESS_PHOTOS}")
    if config.SAVE_PROCESS_PHOTOS:
        app.logger.info(f"  纯YOLO检测图保存: enabled (to process_photo/yolo_raw/)")
        app.logger.info(f"  单帧逻辑投射图保存 (整版模式): enabled (to process_photo/)")
        app.logger.info(f"  最终标注图保存 (整版模式): enabled (to process_photo/)")
        app.logger.info(f"  缩小标注图Base64返回 (零散模式): enabled")
    if config.SAVE_TRAINING_ROI_IMAGES:
         app.logger.info(f"  训练用ROI切片保存: enabled (to process_photo/training_rois/)")

    serve(app, host='0.0.0.0', port=5000)