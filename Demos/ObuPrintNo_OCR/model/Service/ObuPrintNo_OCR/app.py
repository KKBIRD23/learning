# app.py
import os
import cv2 # 保留，因为 process_image_with_ocr_logic 中可能仍直接使用（尽管大部分已移到utils）
import numpy as np # 同上
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

# --- 从新模块导入 ---
import config # 导入所有配置
from image_utils import read_image_cv2, draw_yolo_detections_on_image # 按需导入
from yolo_handler import YoloHandler
from ocr_handler import OcrHandler
from layout_and_state_manager import LayoutStateManager

# --- 全局变量 ---
app = Flask(__name__)
# 将配置加载到app.config (如果Flask路由直接使用)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
# 其他配置项通过导入的 config 模块直接访问

# 模型处理器和状态管理器实例 (在应用启动时初始化)
yolo_predictor: Optional[YoloHandler] = None
ocr_predictor: Optional[OcrHandler] = None
layout_state_mgr: Optional[LayoutStateManager] = None

# 会话数据存储
session_data_store: Dict[str, Any] = {}

# --- 日志设置 ---
def setup_logging(app_instance):
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)

    # 清除Flask默认的处理器，避免重复日志
    # app_instance.logger.handlers.clear() # 如果Flask版本较低，可能没有这个
    # 或者直接设置 propagate = False
    # app_instance.logger.propagate = False

    # 为app.logger添加我们的处理器
    # 确保只添加一次，或者在开发模式下每次重新添加前清除
    if not any(isinstance(h, RotatingFileHandler) for h in app_instance.logger.handlers):
        app_instance.logger.addHandler(file_handler)

    app_instance.logger.setLevel(logging.INFO) # 或从config读取
    app_instance.logger.info(f"Flask应用日志系统已启动。版本: {config.APP_VERSION}")

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许。"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# --- 核心图像处理逻辑 (已重构以使用新模块) ---
def process_image_with_ocr_logic(
    image_path: str,
    session_id: str,
    logger: Any # Flask app logger
) -> Tuple[List[List[int]], Dict[Tuple[int, int], str], Dict[str, float], List[Dict[str, str]]]:
    """
    重构后的核心图像处理流程。
    """
    log_prefix = f"会话 {session_id} (process_image V{config.APP_VERSION}):"
    logger.info(f"{log_prefix} 开始处理图片 {os.path.basename(image_path)}")

    timing_profile = {}
    t_start_overall = time.time()
    warnings_list = []

    # 获取会话数据 (必须存在，由路由确保)
    session = session_data_store.get(session_id)
    if not session: # 双重保险
        logger.error(f"{log_prefix} 严重错误 - 未找到会话数据！")
        # 返回一个表示错误的状态
        empty_matrix = [[0] * config.LAYOUT_REGULAR_COLS_COUNT for _ in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
        return empty_matrix, {}, {"error": "Session data lost"}, [{"message": "会话数据丢失，无法处理。"}]

    # 确保模型处理器已初始化 (在app启动时完成)
    global yolo_predictor, ocr_predictor, layout_state_mgr
    if not all([yolo_predictor, ocr_predictor, layout_state_mgr]):
        logger.critical(f"{log_prefix} 一个或多个核心处理器未初始化！服务配置错误。")
        empty_matrix = [[0] * config.LAYOUT_REGULAR_COLS_COUNT for _ in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
        return empty_matrix, {}, {"error": "Server not ready"}, [{"message": "服务内部错误，处理器未就绪。"}]

    try:
        # 1. 读取图像
        t_step = time.time()
        original_image = read_image_cv2(image_path)
        img_h, img_w = original_image.shape[:2]
        timing_profile['1_image_reading'] = time.time() - t_step
        logger.info(f"{log_prefix} 原始图片: {os.path.basename(image_path)} (H={img_h}, W={img_w})")

        # 2. YOLO检测
        t_step = time.time()
        yolo_detections = yolo_predictor.detect(original_image) # 返回带original_index的列表
        timing_profile['2_yolo_detection'] = time.time() - t_step
        logger.info(f"{log_prefix} YOLO检测完成，找到 {len(yolo_detections)} 个有效框。")

        # 3. OCR处理
        t_step = time.time()
        # 3a. 准备OCR任务 (ROI提取和预处理)
        ocr_tasks_for_pool, ocr_input_metadata = ocr_predictor.prepare_ocr_tasks_from_detections(
            original_image, yolo_detections,
            session_id_for_saving=session_id,
            save_ocr_slices=config.SAVE_PROCESS_PHOTOS,
            process_photo_dir_base=config.PROCESS_PHOTO_DIR
        )
        # 3b. 执行OCR识别
        raw_ocr_pool_results = ocr_predictor.recognize_prepared_tasks(ocr_tasks_for_pool)
        # 3c. 整合OCR结果
        final_ocr_results_list = ocr_predictor.consolidate_ocr_results(raw_ocr_pool_results, ocr_input_metadata)
        timing_profile['3_ocr_processing'] = time.time() - t_step
        logger.info(f"{log_prefix} OCR处理完成，得到 {len(final_ocr_results_list)} 条结果。")

        # 4. 布局学习与状态更新 (核心逻辑)
        t_step = time.time()
        stable_layout_params = session.get("stable_layout_parameters")
        current_frame_num = session.get("frame_count", 0) # 已在路由中递增

        # 4a. 首次校准 (如果需要)
        if not stable_layout_params:
            logger.info(f"{log_prefix} 会话首次有效帧，尝试学习初始稳定布局参数...")
            # 使用当前帧的YOLO检测结果进行校准
            # 注意：yolo_detections 已经是筛选和处理后的结果
            stable_layout_params = layout_state_mgr.learn_initial_stable_layout(
                yolo_detections, (img_w, img_h), session["current_layout_config"], session_id
            )
            if stable_layout_params:
                session["stable_layout_parameters"] = stable_layout_params
                session["layout_parameters"]["is_calibrated"] = True # 标记动态参数也已基于稳定版校准
                logger.info(f"{log_prefix} 初始稳定布局参数学习成功并已保存。")
            else:
                logger.error(f"{log_prefix} 初始稳定布局参数学习失败！后续定位可能不准确。")
                # 即使学习失败，也继续尝试，但结果可能很差
                # 或者在这里决定是否要中断处理并返回错误
                warnings_list.append({"message": "警告：首次布局校准失败，识别结果可能不准确。"})
                # 创建一个空的稳定参数结构，避免后续代码出错
                stable_layout_params = {
                    "avg_physical_row_height": 50, "col_x_estimates_regular": [100,200,300,400],
                    "avg_obu_w": 100, "avg_obu_h": 40, "special_row_at_logical_top": False,
                    "row_y_estimates_initial_guess": [y*50 for y in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
                }
                session["stable_layout_parameters"] = stable_layout_params # 保存一个fallback

        # 4b. Y轴锚定 (后续帧)
        y_anchor_info = None
        current_dynamic_row_y_estimates = list(stable_layout_params.get("row_y_estimates_initial_guess", [])) # 默认值

        # 准备当前帧已通过DB校验的OBU列表 (用于Y轴锚定和核心状态更新)
        current_frame_verified_obus_for_state_update = []
        for ocr_res_item in final_ocr_results_list:
            if ocr_res_item.get("ocr_final_text") in config.VALID_OBU_CODES:
                # 确保 yolo_anchor_details 存在且包含必要字段
                yolo_details = ocr_res_item.get("yolo_anchor_details")
                if yolo_details and all(k in yolo_details for k in ['cx', 'cy', 'w', 'h']):
                    current_frame_verified_obus_for_state_update.append({
                        "text": ocr_res_item["ocr_final_text"],
                        "physical_anchor": yolo_details,
                        "ocr_confidence": ocr_res_item.get("ocr_confidence", 0.0),
                        "original_yolo_idx": ocr_res_item.get("original_index") # 这个索引对应yolo_detections
                    })

        if current_frame_num > 1: # 非首帧才进行Y轴锚定
            y_anchor_info, current_dynamic_row_y_estimates, is_skipped = layout_state_mgr.determine_y_axis_anchor(
                current_frame_verified_obus_for_state_update,
                session["obu_evidence_pool"],
                stable_layout_params, # 传递稳定参数
                session_id,
                current_frame_num
            )
            session["status_flags"]["y_anchor_info_current_frame"] = y_anchor_info
            session["status_flags"]["frame_skipped_due_to_no_overlap"] = is_skipped
            if is_skipped:
                warnings_list.append({
                    "message": "检测到可能的拍摄跳跃或漏帧，当前图像结果未被有效处理。",
                    "code": "FRAME_SKIPPED_NO_OVERLAP"
                })
                # 如果漏帧，则不进行后续的状态更新，直接使用上一帧的矩阵结果
                logger.warning(f"{log_prefix} 因漏帧，跳过本帧核心状态更新。")
                # timing_profile['4_layout_and_state'] = time.time() - t_step (在try块外统一计算)
                # return session["logical_matrix"], session["recognized_texts_map"], timing_profile, warnings_list
                # (或者，我们让它继续走下去，但update_session_state_with_reference_logic内部会处理这个flag)

        # 更新会话中的动态行Y估算 (即使是首帧，也用初始猜测填充一下)
        session["layout_parameters"]["row_y_estimates"] = current_dynamic_row_y_estimates

        # 4c. 核心状态更新 (参照物优先等)
        # 注意：这里直接修改 session 中的 "logical_matrix" 和 "recognized_texts_map"
        if not session["status_flags"].get("frame_skipped_due_to_no_overlap", False):
            _, _, warnings_from_state_update = layout_state_mgr.update_session_state_with_reference_logic(
                session, # 传递整个session字典，让manager直接修改
                current_frame_verified_obus_for_state_update,
                yolo_detections, # 传递原始的YOLO检测结果 (带original_index)
                final_ocr_results_list, # 传递完整的OCR结果 (带original_index)
                y_anchor_info,
                current_dynamic_row_y_estimates,
                stable_layout_params,
                session_id,
                current_frame_num
            )
            if warnings_from_state_update: warnings_list.extend(warnings_from_state_update)

        timing_profile['4_layout_and_state'] = time.time() - t_step
        logger.info(f"{log_prefix} 布局分析与状态更新完成。")

        # 5. (可选) 保存YOLO调试中标注图
        if config.SAVE_PROCESS_PHOTOS and yolo_detections:
            t_step = time.time()
            # 创建一个 ocr_texts_map 用于绘制
            ocr_texts_for_draw_map = {
                ocr_item.get("original_index"): ocr_item.get("ocr_final_text", "ERR")
                for ocr_item in final_ocr_results_list if ocr_item
            }
            annotated_img = draw_yolo_detections_on_image(
                original_image, yolo_detections, ocr_texts_for_draw_map, config.YOLO_COCO_CLASSES
            )
            img_name_base = os.path.splitext(os.path.basename(image_path))[0]
            ts_filename = datetime.now().strftime("%Y%m%d%H%M%S%f")
            annotated_path = os.path.join(config.PROCESS_PHOTO_DIR, f"annotated_{img_name_base}_{ts_filename}.jpg")
            try:
                cv2.imwrite(annotated_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, config.PROCESS_PHOTO_JPG_QUALITY])
                logger.info(f"{log_prefix} YOLO标注图已保存: {annotated_path}")
            except Exception as e_save_ann:
                logger.error(f"{log_prefix} 保存YOLO标注图失败: {e_save_ann}")
            timing_profile['5_drawing_annotations'] = time.time() - t_step

        final_matrix = session.get("logical_matrix", [])
        final_texts = session.get("recognized_texts_map", {})

    except FileNotFoundError as e_fnf: # 特别处理文件未找到
        logger.error(f"{log_prefix} 处理图片时文件未找到: {e_fnf}", exc_info=True)
        warnings_list.append({"message": f"错误: 图片文件处理失败 ({os.path.basename(image_path)}).", "code": "FILE_PROCESSING_ERROR"})
        empty_matrix = [[0] * config.LAYOUT_REGULAR_COLS_COUNT for _ in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
        final_matrix, final_texts = empty_matrix, {} # 返回空结果
    except Exception as e:
        logger.error(f"{log_prefix} 处理图片时发生未知严重错误: {e}", exc_info=True)
        warnings_list.append({"message": f"服务内部错误，处理失败: {str(e)}", "code": "INTERNAL_SERVER_ERROR"})
        empty_matrix = [[0] * config.LAYOUT_REGULAR_COLS_COUNT for _ in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
        final_matrix, final_texts = empty_matrix, {} # 返回空结果

    timing_profile['0_total_processing_function'] = time.time() - t_start_overall
    logger.info(f"{log_prefix} --- Timing profile for {os.path.basename(image_path)} ---")
    for key, val in sorted(timing_profile.items()):
        logger.info(f"  {key}: {val:.3f}s")

    return final_matrix, final_texts, timing_profile, warnings_list


# --- Flask 路由 ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger # 使用Flask的logger实例

    if 'session_id' not in request.form:
        logger.error("请求中缺少 'session_id'。")
        return jsonify({"error": "session_id is required"}), 400
    session_id = request.form.get('session_id')

    force_recalibrate_str = request.form.get('force_recalibrate', 'false').lower()
    force_recalibrate = (force_recalibrate_str == 'true')

    # 文件检查 (与您之前代码一致)
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
        filename_on_server = f"{name}_{timestamp}{ext}"

        upload_path = os.path.join(config.UPLOAD_FOLDER, filename_on_server)
        if not os.path.exists(config.UPLOAD_FOLDER):
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        file.save(upload_path)
        logger.info(f"会话 {session_id}: 文件 '{filename_on_server}' 已成功保存到 '{upload_path}'")

        # 获取或初始化会话数据
        session = session_data_store.get(session_id)
        if not session or force_recalibrate:
            log_msg = f"会话 {session_id}: " + ("用户触发强制重新校准。" if force_recalibrate and session else "新建会话或首次处理。")
            logger.info(log_msg)

            # 初始化逻辑矩阵 (状态0:未知, -1:不可用)
            initial_matrix = [[0] * config.LAYOUT_REGULAR_COLS_COUNT
                              for _ in range(config.LAYOUT_EXPECTED_TOTAL_ROWS)]
            # 标记特殊行两边的不可用格子
            # (假设特殊行在底部，且为4列常规布局中的中间2列)
            if config.LAYOUT_EXPECTED_TOTAL_ROWS > 0 and \
               config.LAYOUT_REGULAR_COLS_COUNT == 4 and \
               config.LAYOUT_SPECIAL_ROW_COLS_COUNT == 2:
                special_row_idx = config.LAYOUT_EXPECTED_TOTAL_ROWS - 1 # 底部行索引
                if 0 <= special_row_idx < len(initial_matrix):
                    initial_matrix[special_row_idx][0] = -1
                    initial_matrix[special_row_idx][3] = -1

            session_data_store[session_id] = {
                "logical_matrix": initial_matrix,
                "recognized_texts_map": {},
                "obu_evidence_pool": {},
                "layout_parameters": { # 动态参数，会被Y轴锚定等更新
                    "is_calibrated": False,
                    "row_y_estimates": []
                },
                "stable_layout_parameters": None, # 首次校准后填充
                "current_layout_config": { # 从全局config中提取当前会话的布局配置
                    "expected_total_rows": config.LAYOUT_EXPECTED_TOTAL_ROWS,
                    "regular_rows_count": config.LAYOUT_REGULAR_ROWS_COUNT,
                    "regular_cols_count": config.LAYOUT_REGULAR_COLS_COUNT,
                    "special_row_cols_count": config.LAYOUT_SPECIAL_ROW_COLS_COUNT,
                    "total_obus": config.LAYOUT_TOTAL_OBUS_EXPECTED
                },
                "frame_count": 0, # 将在这里递增
                "last_activity": datetime.now(),
                "status_flags": {
                    "frame_skipped_due_to_no_overlap": False,
                    "y_anchor_info_current_frame": None
                }
            }

        session = session_data_store[session_id] # 确保获取到的是最新的
        session["frame_count"] += 1
        session["last_activity"] = datetime.now()
        session["status_flags"]["frame_skipped_due_to_no_overlap"] = False # 重置漏帧标记

        # 调用核心处理逻辑
        matrix_res, texts_res, timings_res, warnings_res = process_image_with_ocr_logic(
            upload_path, session_id, logger
        )

        # 准备响应
        response_data = {
            "message": "File processed successfully.",
            "session_id": session_id,
            "received_filename": original_filename_for_log,
            "obu_status_matrix": matrix_res,
            "obu_texts": {f"{r}_{c}": text for (r,c), text in texts_res.items()}, # 字典key转字符串
            "timing_profile_seconds": timings_res,
            "warnings": warnings_res
        }

        num_identified = sum(1 for r in matrix_res for status in r if status == 1)
        total_expected = session["current_layout_config"].get("total_obus", 50)
        response_data["session_status"] = "completed" if num_identified >= total_expected else "in_progress"
        logger.info(f"会话 {session_id}: 已识别 {num_identified}/{total_expected} 个OBU。状态: {response_data['session_status']}")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_log}' 时发生严重错误: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- 应用初始化与启动 ---
def initialize_global_handlers(app_logger):
    """初始化全局的模型处理器。"""
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
            logger=app_logger
        )
        ocr_predictor = OcrHandler(
            rec_model_dir=config.OCR_REC_MODEL_DIR,
            num_workers=config.OCR_NUM_WORKERS,
            target_ocr_input_height=config.OCR_TARGET_INPUT_HEIGHT,
            digit_roi_y_offset_factor=config.OCR_DIGIT_ROI_Y_OFFSET_FACTOR,
            digit_roi_height_factor=config.OCR_DIGIT_ROI_HEIGHT_FACTOR,
            digit_roi_width_expand_factor=config.OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR,
            logger=app_logger
        )
        layout_state_mgr = LayoutStateManager(
            config={ # 将需要的配置项传递给状态管理器
                "LAYOUT_EXPECTED_TOTAL_ROWS": config.LAYOUT_EXPECTED_TOTAL_ROWS,
                "LAYOUT_REGULAR_COLS_COUNT": config.LAYOUT_REGULAR_COLS_COUNT,
                "LAYOUT_SPECIAL_ROW_COLS_COUNT": config.LAYOUT_SPECIAL_ROW_COLS_COUNT,
                "LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING": config.LAYOUT_MIN_CORE_ANCHORS_FOR_LEARNING,
                "LAYOUT_MIN_VALID_ROWS_FOR_LEARNING": config.LAYOUT_MIN_VALID_ROWS_FOR_LEARNING,
                "LAYOUT_MIN_ANCHORS_PER_RELIABLE_ROW": config.LAYOUT_MIN_ANCHORS_PER_RELIABLE_ROW,
                "LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR": config.LAYOUT_ROW_GROUP_Y_THRESHOLD_FACTOR,
                "LAYOUT_Y_MATCH_THRESHOLD_FACTOR": config.LAYOUT_Y_MATCH_THRESHOLD_FACTOR,
                "LAYOUT_X_MATCH_THRESHOLD_FACTOR": config.LAYOUT_X_MATCH_THRESHOLD_FACTOR
            },
            logger=app_logger
        )
        app_logger.info("--- 全局处理器初始化完成 ---")
    except Exception as e:
        app_logger.critical(f"全局处理器初始化失败: {e}", exc_info=True)
        # 这种情况下服务可能无法正常工作，可以考虑是否要退出或进入维护模式
        raise # 重新抛出异常，让应用启动失败

def cleanup_ocr_pool_on_exit():
    """在应用退出时清理OCR处理池。"""
    global ocr_predictor
    if ocr_predictor and hasattr(ocr_predictor, 'close_pool'):
        current_app.logger.info("应用退出，正在关闭OCR处理池...")
        ocr_predictor.close_pool()

if __name__ == '__main__':
    setup_logging(app) # 初始化日志系统

    try:
        initialize_global_handlers(app.logger) # 初始化模型处理器
    except Exception as e_init:
        app.logger.critical(f"应用启动失败，无法初始化核心处理器: {e_init}")
        # 退出应用，因为核心功能无法工作
        exit(1)

    atexit.register(cleanup_ocr_pool_on_exit) # 注册退出时清理函数

    # 确保必要的目录存在
    if not os.path.exists(config.UPLOAD_FOLDER): os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(config.PROCESS_PHOTO_DIR): os.makedirs(config.PROCESS_PHOTO_DIR, exist_ok=True)

    app.logger.info(f"服务版本 {config.APP_VERSION} 启动中... 监听 0.0.0.0:5000")
    app.logger.info(f"过程图片保存开关: {config.SAVE_PROCESS_PHOTOS}")
    if config.SAVE_PROCESS_PHOTOS:
        app.logger.info(f"过程图片保存目录: {config.PROCESS_PHOTO_DIR}")
        app.logger.info(f"过程图片JPG质量: {config.PROCESS_PHOTO_JPG_QUALITY}")

    # debug=True 会导致Flask为每个请求重新加载代码，这对于有状态的全局变量（如模型实例）
    # 和多进程池（如OCR池）可能会产生问题。在生产或稳定测试时应设为False。
    # use_reloader=False 可以防止由代码更改引起的自动重启，这对于调试多进程池有时是必要的。
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)