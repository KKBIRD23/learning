# app.py
import os
import cv2
import numpy as np
import time
import traceback # 用于更详细的错误日志
import multiprocessing
from datetime import datetime
from flask import Flask, request, jsonify, current_app # 确保 current_app 被导入
from werkzeug.utils import secure_filename
import atexit
import logging
from logging.handlers import RotatingFileHandler
import uuid
from typing import List, Dict, Tuple, Any, Optional

# --- 从新模块导入 ---
import config # 导入所有配置
from image_utils import read_image_cv2, draw_yolo_detections_on_image
from yolo_handler import YoloHandler
from ocr_handler import OcrHandler
from layout_and_state_manager import LayoutStateManager

# --- 全局变量 ---
app = Flask(__name__)
# 将配置加载到app.config (如果Flask路由直接使用，或者通过current_app.config访问)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
# 其他配置项通过导入的 config 模块直接访问，例如 config.SERVER_URL (如果定义了)

# 模型处理器和状态管理器实例 (在应用启动时初始化)
yolo_predictor: Optional[YoloHandler] = None
ocr_predictor: Optional[OcrHandler] = None
layout_state_mgr: Optional[LayoutStateManager] = None

# 会话数据存储 (简单的内存字典，生产环境可能需要更持久化的方案)
session_data_store: Dict[str, Any] = {}

# --- 日志设置 ---
def setup_logging(app_instance):
    """配置应用的日志记录器。"""
    if not os.path.exists(config.LOG_DIR):
        try:
            os.makedirs(config.LOG_DIR)
        except OSError as e:
            # 在应用启动早期，如果日志目录创建失败，可能只能print
            print(f"Error creating log directory {config.LOG_DIR}: {e}")
            # 也可以选择抛出异常，让应用启动失败
            # raise

    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE)

    # 使用RotatingFileHandler进行日志轮转
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8' # 确保UTF-8编码
    )
    # 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(formatter)

    # 清除Flask默认的处理器，以避免重复日志或使用我们自定义的格式
    # (注意：直接清除 app_instance.logger.handlers 可能比较激进，
    #  如果Flask有其他重要的默认处理器，可能会受影响。
    #  更稳妥的方式是检查并移除特定的处理器，或者设置 propagate=False)
    # for handler in list(app_instance.logger.handlers): # 创建副本进行迭代和移除
    #     app_instance.logger.removeHandler(handler)
    # app_instance.logger.propagate = False # 阻止日志向根logger传播

    # 为app.logger添加我们的处理器 (确保只添加一次)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == file_handler.baseFilename for h in app_instance.logger.handlers):
        app_instance.logger.addHandler(file_handler)

    app_instance.logger.setLevel(logging.INFO) # 或从config读取日志级别
    app_instance.logger.info(f"Flask应用日志系统已启动。版本: {config.APP_VERSION}")

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许。"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# process_image_with_ocr_logic 函数将在下一段提供
# app.py (Continued - Part 2 - process_image_with_ocr_logic)

# --- 核心图像处理逻辑 (V3.3_P1_Scatter_Complete_And_Corrected) ---
def process_image_with_ocr_logic(
    image_path: str,
    session_id: str,
    logger: Any, # 通常是 current_app.logger
    mode: str = 'full_layout',
) -> Tuple[Optional[List[List[int]]], Optional[Dict[Any, Any]], Dict[str, float], List[Dict[str, str]], Optional[List[Dict[str,Any]]]]:
    log_prefix = f"会话 {session_id} (process_image V{config.APP_VERSION}_P1Scatter M:{mode}):"
    logger.info(f"{log_prefix} 开始处理图片 {os.path.basename(image_path)}")

    timing_profile = {}
    t_start_overall = time.time()
    warnings_list = []
    scattered_results_list: Optional[List[Dict[str, Any]]] = None

    # --- 初始化默认返回值 ---
    # 尝试从会话获取配置，否则使用全局config的默认值
    session_obj_for_init = session_data_store.get(session_id) # 在try之前获取一次
    current_layout_config_for_init = config # 默认使用全局config
    if session_obj_for_init and "current_layout_config" in session_obj_for_init:
        current_layout_config_for_init = session_obj_for_init["current_layout_config"]

    expected_rows_for_empty = getattr(current_layout_config_for_init, "LAYOUT_EXPECTED_TOTAL_ROWS", config.LAYOUT_EXPECTED_TOTAL_ROWS)
    expected_cols_for_empty = getattr(current_layout_config_for_init, "LAYOUT_REGULAR_COLS_COUNT", config.LAYOUT_REGULAR_COLS_COUNT)

    default_empty_matrix = [[0] * expected_cols_for_empty for _ in range(expected_rows_for_empty)]
    # 特殊行-1标记将在 update_session_state_with_reference_logic 中根据情况处理

    final_matrix: Optional[List[List[int]]] = default_empty_matrix
    final_texts: Optional[Dict[Tuple[int, int], str]] = {}
    # --- 结束返回值初始化 ---

    # 再次获取session，确保是最新的，因为路由中可能已经创建或更新了它
    session = session_data_store.get(session_id)
    if not session and mode == 'full_layout':
        logger.error(f"{log_prefix} 严重错误 - 整版模式下未找到会话数据！")
        return final_matrix, final_texts, {"error": "Session data lost for full_layout mode"}, [{"message": "会话数据丢失，无法处理整版识别。"}], None

    global yolo_predictor, ocr_predictor, layout_state_mgr
    if mode == 'full_layout' and not layout_state_mgr:
        logger.critical(f"{log_prefix} 整版模式下LayoutStateManager未初始化！")
        return final_matrix, final_texts, {"error": "Server not ready"}, [{"message": "服务内部错误(LSM)。"}], None
    elif not yolo_predictor or not ocr_predictor:
        logger.critical(f"{log_prefix} YOLO或OCR核心处理器未初始化！")
        return final_matrix, final_texts, {"error": "Server not ready"}, [{"message": "服务内部错误(YOLO/OCR)。"}], None

    try:
        # 1. 读取图像
        t_step = time.time()
        original_image = read_image_cv2(image_path)
        img_h, img_w = original_image.shape[:2]
        timing_profile['1_image_reading'] = time.time() - t_step
        logger.info(f"{log_prefix} 原始图片: {os.path.basename(image_path)} (H={img_h}, W={img_w})")

        # 2. YOLO检测
        t_step = time.time()
        yolo_detections = yolo_predictor.detect(original_image)
        timing_profile['2_yolo_detection'] = time.time() - t_step
        logger.info(f"{log_prefix} YOLO检测完成，找到 {len(yolo_detections)} 个有效框。")

        # --- 保存纯YOLO检测结果图 ---
        if config.SAVE_PROCESS_PHOTOS and yolo_detections:
            try:
                yolo_raw_img = draw_yolo_detections_on_image(original_image, yolo_detections, None, config.YOLO_COCO_CLASSES)
                img_name_base_raw = os.path.splitext(os.path.basename(image_path))[0]
                ts_filename_raw = datetime.now().strftime("%Y%m%d%H%M%S%f")
                yolo_raw_dir = os.path.join(config.PROCESS_PHOTO_DIR, "yolo_raw")
                if not os.path.exists(yolo_raw_dir): os.makedirs(yolo_raw_dir, exist_ok=True)
                frame_num_for_filename = session.get("frame_count", 0) if session else 0 # 使用当前帧号
                yolo_raw_path = os.path.join(yolo_raw_dir, f"yolo_raw_{img_name_base_raw}_s{session_id[:8]}_f{frame_num_for_filename}_{ts_filename_raw}.jpg")
                cv2.imwrite(yolo_raw_path, yolo_raw_img, [cv2.IMWRITE_JPEG_QUALITY, config.PROCESS_PHOTO_JPG_QUALITY])
                logger.info(f"{log_prefix} 纯YOLO检测结果图已保存: {yolo_raw_path}")
            except Exception as e_save_yolo_raw:
                logger.error(f"{log_prefix} 保存纯YOLO检测结果图失败: {e_save_yolo_raw}", exc_info=True)
        # --- 结束保存 ---

        # 3. OCR处理
        t_step = time.time()
        current_frame_num_for_ocr = session.get("frame_count", 0) if session else 0
        ocr_tasks_for_pool, ocr_input_metadata = ocr_predictor.prepare_ocr_tasks_from_detections(
            original_image, yolo_detections, session_id, current_frame_num_for_ocr, config.SAVE_PROCESS_PHOTOS
        )
        raw_ocr_pool_results = ocr_predictor.recognize_prepared_tasks(ocr_tasks_for_pool)
        final_ocr_results_list = ocr_predictor.consolidate_ocr_results(raw_ocr_pool_results, ocr_input_metadata)
        timing_profile['3_ocr_processing'] = time.time() - t_step
        logger.info(f"{log_prefix} OCR处理完成，得到 {len(final_ocr_results_list)} 条结果。")

        # --- 根据模式进行后续处理 ---
        if mode == 'scattered_ocr':
            logger.info(f"{log_prefix} 进入“零散识别”模式处理流程。")
            scattered_results_list = []
            for ocr_item in final_ocr_results_list:
                if ocr_item and ocr_item.get("ocr_final_text") and \
                   ocr_item.get("ocr_final_text") not in ["N/A_NO_DIGITS", "N/A_EMPTY_RAW"] and \
                   not ocr_item.get("ocr_final_text", "").startswith("OCR_") and \
                   ocr_item.get("ocr_final_text") in config.VALID_OBU_CODES:
                    scattered_results_list.append({
                        "text": ocr_item["ocr_final_text"],
                        "bbox_xyxy": ocr_item.get("bbox_yolo_abs"),
                        "ocr_confidence": ocr_item.get("ocr_confidence"),
                        "yolo_score": ocr_item.get("yolo_anchor_details", {}).get("score")})
            logger.info(f"{log_prefix} “零散识别”模式完成，识别到 {len(scattered_results_list)} 个有效OBU。")
            final_matrix = None; final_texts = None

        elif mode == 'full_layout':
            if not session:
                 logger.error(f"{log_prefix} 严重错误 - 整版模式下session对象丢失！") # 理论上不会发生
                 return default_empty_matrix, {}, {"error": "Session object lost during full_layout processing"}, [{"message": "会话数据丢失。"}], None

            logger.info(f"{log_prefix} 进入“整版识别”模式处理流程。")
            t_step_layout = time.time()
            stable_layout_params_from_session = session.get("stable_layout_parameters")
            current_frame_num_layout = session.get("frame_count", 0)
            session_config = session["current_layout_config"]

            current_frame_layout_stats, yolo_detections_with_rc = layout_state_mgr.analyze_frame_layout_and_get_params(
                yolo_detections, (img_w, img_h), session_config, session_id
            )

            if not current_frame_layout_stats:
                logger.error(f"{log_prefix} 当前帧布局分析失败。")
                warnings_list.append({"message": "警告：当前帧布局分析失败。", "code": "FRAME_LAYOUT_ANALYSIS_FAILED"}); current_frame_layout_stats = {}
            if not yolo_detections_with_rc and yolo_detections:
                logger.warning(f"{log_prefix} XY聚类未返回带rc的检测结果。"); yolo_detections_with_rc = [dict(d, frame_r=-1, frame_c=-1) for d in yolo_detections]

            if not stable_layout_params_from_session and current_frame_layout_stats:
                logger.info(f"{log_prefix} 会话首次有效帧，使用当前帧统计结果作为初始稳定布局参考。")
                session["stable_layout_parameters"] = current_frame_layout_stats.copy(); session["layout_parameters"]["is_calibrated"] = True
                stable_layout_params_from_session = session["stable_layout_parameters"]

            if not stable_layout_params_from_session:
                logger.error(f"{log_prefix} 无法获取有效的稳定布局参数！")
                stable_layout_params_from_session = {
                    "median_obu_w_stable": 100, "median_obu_h_stable": 40, "avg_physical_row_height_stable": 60,
                    "row_y_means_from_clustering": {}, "col_x_means_from_clustering": {},
                    "special_row_at_logical_top": False, "identified_special_row_frame_r": -1}
                session["stable_layout_parameters"] = stable_layout_params_from_session

            current_frame_verified_obus_for_anchor = []
            map_yolo_idx_to_ocr_item = {ocr.get("original_index"): ocr for ocr in final_ocr_results_list if ocr}
            for det_rc in yolo_detections_with_rc:
               original_yolo_idx = det_rc["original_index"]
               ocr_item = map_yolo_idx_to_ocr_item.get(original_yolo_idx)
               if ocr_item and ocr_item.get("ocr_final_text") in config.VALID_OBU_CODES:
                   yolo_details = ocr_item.get("yolo_anchor_details")
                   if yolo_details and det_rc.get("frame_r", -1) != -1 :
                       current_frame_verified_obus_for_anchor.append({
                           "text": ocr_item["ocr_final_text"], "physical_anchor": yolo_details,
                           "ocr_confidence": ocr_item.get("ocr_confidence", 0.0),
                           "original_yolo_idx": original_yolo_idx, "frame_r": det_rc["frame_r"]})

            y_anchor_info, estimated_row_y_for_drawing = layout_state_mgr.determine_y_axis_anchor(
               current_frame_verified_obus_for_anchor, session["obu_evidence_pool"],
               current_frame_layout_stats, session_id, current_frame_num_layout, session_config)
            session["status_flags"]["y_anchor_info_current_frame"] = y_anchor_info

            if current_frame_num_layout == 1 and y_anchor_info.get("is_first_frame_anchor_failed", False):
                logger.error(f"{log_prefix} 【行政规定检查失败】第一帧未能通过特殊行成功锚定！")
                warnings_list.append({
                    "message": "错误：第一帧未能识别到近端特殊行或锚定失败。请确保从近端特殊行开始拍摄并保证其清晰、完整（至少两格）地被拍摄入图。",
                    "code": "FIRST_FRAME_ANCHOR_FAILED_ADMIN_RULE"
                })
                timing_profile['4_layout_and_state'] = time.time() - t_step_layout
                # final_matrix 和 final_texts 保持在try之前的默认空值
                return final_matrix, final_texts, timing_profile, warnings_list, None

            if y_anchor_info.get("is_skipped_due_to_no_overlap"):
                warnings_list.append({ "message": "检测到可能的拍摄跳跃或漏帧。", "code": "FRAME_SKIPPED_NO_OVERLAP"})

            session["layout_parameters"]["row_y_estimates"] = estimated_row_y_for_drawing

            if config.SAVE_PROCESS_PHOTOS:
                logger.critical(f"{log_prefix} CRITICAL_LOG: 即将调用 draw_stable_layout_on_image。")
                try:
                    layout_state_mgr.draw_stable_layout_on_image(
                        yolo_detections_with_rc, (img_w, img_h), session_id,
                        current_frame_num_layout, y_anchor_info, current_frame_layout_stats)
                    logger.critical(f"{log_prefix} CRITICAL_LOG: draw_stable_layout_on_image 调用完成。")
                except Exception as e_draw_layout:
                    logger.critical(f"{log_prefix} CRITICAL_LOG: 绘制单帧布局图时发生严重错误: {e_draw_layout}", exc_info=True)

            if not y_anchor_info.get("is_skipped_due_to_no_overlap", False):
                matrix_updated, texts_updated, warnings_state = layout_state_mgr.update_session_state_with_reference_logic(
                   session, yolo_detections_with_rc, final_ocr_results_list,
                   y_anchor_info, session_id, current_frame_num_layout, session_config)
                if warnings_state: warnings_list.extend(warnings_state)
                final_matrix = matrix_updated; final_texts = texts_updated
            else:
                logger.warning(f"{log_prefix} 因漏帧，跳过核心状态更新。返回当前会话状态。")
                final_matrix = session.get("logical_matrix", final_matrix)
                final_texts = session.get("recognized_texts_map", final_texts)
            timing_profile['4_layout_and_state'] = time.time() - t_step_layout
            logger.info(f"{log_prefix} “整版识别”模式处理流程完成。")
        else:
            logger.error(f"{log_prefix} 未知的处理模式: {mode}")
            warnings_list.append({"message": f"错误：未知的处理模式 '{mode}'。", "code": "UNKNOWN_MODE"})
            final_matrix = None; final_texts = None; scattered_results_list = []

        # 保存最终的YOLO标注图 (带OCR文本)
        if config.SAVE_PROCESS_PHOTOS and yolo_detections:
            # (这个绘制逻辑对于零散模式可能意义不大，或者需要不同的绘制方式，暂时只在整版模式下绘制)
            # if mode == 'full_layout' or (mode == 'scattered_ocr' and scattered_results_list): # 零散模式也画一个？
            if mode == 'full_layout' and final_texts is not None: # 只在整版且有结果时画
                t_step_draw_yolo = time.time()
                ocr_texts_for_draw_map = {}
                # 从 final_texts (这是 (r,c)->text 格式) 转换回 yolo_original_index -> text 格式
                # 这比较困难，因为我们不知道哪个yolo_original_index对应哪个(r,c)了
                # 更简单的方式是直接用 final_ocr_results_list 来构建这个map
                if final_ocr_results_list:
                    for ocr_item in final_ocr_results_list:
                        if ocr_item and isinstance(ocr_item, dict) and "original_index" in ocr_item:
                            # 只标注那些最终被成功填充到矩阵中的文本（如果能获取到的话）
                            # 或者，简单地标注所有OCR识别出的文本（即使它最终没被用）
                            ocr_text_to_draw = ocr_item.get("ocr_final_text", "ERR")
                            if ocr_text_to_draw in config.VALID_OBU_CODES : # 只画有效的
                                ocr_texts_for_draw_map[ocr_item.get("original_index")] = ocr_text_to_draw

                annotated_img = draw_yolo_detections_on_image(
                    original_image, yolo_detections, ocr_texts_for_draw_map, config.YOLO_COCO_CLASSES
                )
                img_name_base = os.path.splitext(os.path.basename(image_path))[0]
                ts_filename = datetime.now().strftime("%Y%m%d%H%M%S%f")
                process_photo_main_dir = config.PROCESS_PHOTO_DIR
                if not os.path.exists(process_photo_main_dir): os.makedirs(process_photo_main_dir, exist_ok=True)
                annotated_path = os.path.join(process_photo_main_dir, f"annotated_{img_name_base}_s{session_id[:8]}_f{current_frame_num_for_ocr}_{ts_filename}.jpg")
                try:
                    save_success = cv2.imwrite(annotated_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, config.PROCESS_PHOTO_JPG_QUALITY])
                    if save_success: logger.info(f"{log_prefix} YOLO标注图已保存: {annotated_path}")
                    else: logger.error(f"{log_prefix} cv2.imwrite未能保存YOLO标注图到 {annotated_path} (返回False)")
                except Exception as e_save_ann: logger.error(f"{log_prefix} 保存YOLO标注图失败: {e_save_ann}", exc_info=True)
                timing_profile['5_drawing_annotations'] = time.time() - t_step_draw_yolo

    except FileNotFoundError as e_fnf:
        logger.error(f"{log_prefix} 处理图片时文件未找到: {e_fnf}", exc_info=True)
        warnings_list.append({"message": f"错误: 图片文件处理失败 ({os.path.basename(image_path)}).", "code": "FILE_PROCESSING_ERROR"})
    except Exception as e:
        logger.error(f"{log_prefix} 处理图片时发生未知严重错误: {e}", exc_info=True)
        warnings_list.append({"message": f"服务内部错误，处理失败: {str(e)}", "code": "INTERNAL_SERVER_ERROR"})

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall
    logger.info(f"{log_prefix} --- Timing profile for {os.path.basename(image_path)} ---")
    for key, val in sorted(timing_profile.items()): logger.info(f"  {key}: {val:.3f}s")

    return final_matrix, final_texts, timing_profile, warnings_list, scattered_results_list

# --- Flask 路由 (predict_image_route 需要修改以接收 mode 参数并调整返回) ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger

    session_id = request.form.get('session_id')
    if not session_id:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400

    mode = request.form.get('mode', 'full_layout').lower()
    if mode not in ['full_layout', 'scattered_ocr']:
        logger.warning(f"会话 {session_id}: 无效的处理模式 '{mode}'，将使用默认 'full_layout'。")
        mode = 'full_layout'

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

        # 获取当前帧号用于文件名 (如果会话已存在)
        session_obj_for_filename = session_data_store.get(session_id)
        frame_num_for_filename = 0
        if session_obj_for_filename and mode == 'full_layout': # 整版模式才递增和使用帧号
            frame_num_for_filename = session_obj_for_filename.get("frame_count", 0) + 1
        elif mode == 'scattered_ocr': # 零散模式可以用一个通用标记或时间戳
            frame_num_for_filename = 0 # 或者用一个随机数/时间戳避免冲突，但通常单次请求
        else: # 新会话的整版模式第一帧
            frame_num_for_filename = 1

        filename_on_server = f"s{session_id[:8]}_f{frame_num_for_filename}_{name}_{timestamp}{ext}"

        upload_path = os.path.join(config.UPLOAD_FOLDER, filename_on_server)
        if not os.path.exists(config.UPLOAD_FOLDER):
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        file.save(upload_path)
        logger.info(f"会话 {session_id}: 文件 '{filename_on_server}' 已成功保存到 '{upload_path}'")

        # --- 会话初始化或获取 ---
        if mode == 'full_layout':
            session = session_data_store.get(session_id)
            if not session or force_recalibrate:
                log_msg = f"会话 {session_id} (整版模式): " + \
                          ("用户触发强制重新校准。" if force_recalibrate and session else "新建会话或首次处理。")
                logger.info(log_msg)

                # 从全局config初始化会话特定的布局配置
                current_layout_config_for_session = {
                    "expected_total_rows": config.LAYOUT_EXPECTED_TOTAL_ROWS,
                    "regular_rows_count": config.LAYOUT_REGULAR_ROWS_COUNT, # 虽然可能不直接用，但保留
                    "regular_cols_count": config.LAYOUT_REGULAR_COLS_COUNT,
                    "special_row_cols_count": config.LAYOUT_SPECIAL_ROW_COLS_COUNT,
                    "total_obus": config.LAYOUT_TOTAL_OBUS_EXPECTED
                }
                initial_matrix = [[0] * current_layout_config_for_session["regular_cols_count"]
                                  for _ in range(current_layout_config_for_session["expected_total_rows"])]
                # 初始时，我们先不标记特殊行的-1，让 update_session_state_with_reference_logic 在首次校准后去做

                session_data_store[session_id] = {
                    "logical_matrix": initial_matrix, "recognized_texts_map": {}, "obu_evidence_pool": {},
                    "layout_parameters": {"is_calibrated": False, "row_y_estimates": [] }, # 动态参数
                    "stable_layout_parameters": None, # 首次分析后填充当前帧的统计作为参考
                    "current_layout_config": current_layout_config_for_session,
                    "frame_count": 0, # 将在这里递增
                    "last_activity": datetime.now(),
                    "status_flags": {
                        "frame_skipped_due_to_no_overlap": False,
                        "y_anchor_info_current_frame": None, # y_anchor_info 会包含 is_first_frame_anchor_failed
                    }}
            session = session_data_store[session_id]
            session["frame_count"] += 1 # 递增帧计数器
            session["last_activity"] = datetime.now()
            # 重置上一帧的锚定失败标记 (如果存在的话)
            if "y_anchor_info_current_frame" in session["status_flags"] and \
               session["status_flags"]["y_anchor_info_current_frame"] is not None:
                session["status_flags"]["y_anchor_info_current_frame"]["is_first_frame_anchor_failed"] = False
            session["status_flags"]["frame_skipped_due_to_no_overlap"] = False

        # 调用核心处理逻辑
        matrix_res, texts_res, timings_res, warnings_res, scattered_res = process_image_with_ocr_logic(
            upload_path, session_id, logger, mode=mode
        )

        # --- 根据模式准备响应 ---
        response_data = {
            "message": "File processed successfully.",
            "session_id": session_id,
            "received_filename": original_filename_for_log,
            "mode_processed": mode,
            "timing_profile_seconds": timings_res,
            "warnings": warnings_res
        }

        current_session_status = "unknown" # 默认状态

        if mode == 'full_layout':
            response_data["obu_status_matrix"] = matrix_res
            response_data["obu_texts"] = {f"{r}_{c}": text for (r,c), text in texts_res.items()} if texts_res else {}

            num_identified = sum(1 for r_val in matrix_res for status in r_val if status == 1) if matrix_res else 0
            # 从会话中获取 total_expected，如果会话不存在（例如，首帧锚定失败提前返回），则用config默认值
            session_for_status = session_data_store.get(session_id) # 再次获取最新的会话状态
            total_expected = config.LAYOUT_TOTAL_OBUS_EXPECTED # 默认值
            current_frame_being_processed = 0 # 默认值，以防 session_for_status 为 None

            if session_for_status: # 只有当会话存在时，才尝试从中获取信息
                 current_layout_config = session_for_status.get("current_layout_config", {})
                 total_expected = current_layout_config.get("total_obus", total_expected)
                 current_frame_being_processed = session_for_status.get("frame_count", 0)

                 y_anchor_info_final = session_for_status.get("status_flags",{}).get("y_anchor_info_current_frame")

                 if current_frame_being_processed == 1 and \
                    y_anchor_info_final and \
                    y_anchor_info_final.get("is_first_frame_anchor_failed", False):
                     current_session_status = "first_frame_anchor_error"
                 elif num_identified >= total_expected:
                     current_session_status = "completed"
                 else:
                     current_session_status = "in_progress"
            else: # 如果会话意外丢失 (理论上不应发生，因为前面有检查)
                logger.error(f"会话 {session_id}: 在准备响应时发现会话数据丢失！")
                current_session_status = "error_session_lost" # 定义一个新的错误状态
            logger.info(f"会话 {session_id} (整版模式): 已识别 {num_identified}/{total_expected} 个OBU。状态: {current_session_status}")

        elif mode == 'scattered_ocr':
            response_data["scattered_results"] = scattered_res
            current_session_status = "scattered_ocr_completed"
            response_data["session_status"] = current_session_status
            logger.info(f"会话 {session_id} (零散模式): 识别到 {len(scattered_res if scattered_res else [])} 个OBU。")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_log}' 时发生严重错误: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- 应用初始化与启动 (与上一版相同) ---
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
            rec_model_dir=config.OCR_REC_MODEL_DIR,
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
                "IMAGE_FALLBACK_HEIGHT_FOR_LAYOUT": 5712, # 可以提供一个默认的图像高度参考
                "avg_obu_h": 40.0, # Fallback median_obu_h for LayoutStateManager
                "avg_obu_w": 100.0 # Fallback median_obu_w for LayoutStateManager
            }, logger=app_logger)
        app_logger.info("--- 全局处理器初始化完成 ---")
    except Exception as e:
        app_logger.critical(f"全局处理器初始化失败: {e}", exc_info=True)
        raise

def cleanup_ocr_pool_on_exit():
    global ocr_predictor
    if ocr_predictor and hasattr(ocr_predictor, 'close_pool'):
        # 使用 current_app.logger 可能在 atexit 时遇到问题，直接用 print 或标准 logging
        print("应用退出，正在关闭OCR处理池...")
        ocr_predictor.close_pool()
        print("OCR处理池已关闭。")

if __name__ == '__main__':
    # setup_logging 应该在 app 创建后，但在任何 app.logger 使用前调用
    # 但由于 initialize_global_handlers 也用 app.logger，所以需要先 setup
    # 为了确保，我们可以在这里再次调用（它内部有防止重复添加的逻辑）
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
    app.logger.info(f"过程图片保存开关: {config.SAVE_PROCESS_PHOTOS}")
    if config.SAVE_PROCESS_PHOTOS:
        app.logger.info(f"  纯YOLO检测图保存: enabled (to process_photo/yolo_raw/)")
        app.logger.info(f"  单帧逻辑投射图保存: enabled (to process_photo/)")
        app.logger.info(f"  最终标注图保存: enabled (to process_photo/)")
    if config.SAVE_TRAINING_ROI_IMAGES:
         app.logger.info(f"  训练用ROI切片保存: enabled (to process_photo/training_rois/)")

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)