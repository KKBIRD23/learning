# app.py (V19.0_Final_Streamlined)
import os
import cv2
import numpy as np
import time
import traceback
import multiprocessing
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import atexit
import logging
from logging.handlers import RotatingFileHandler
import uuid
from typing import List, Dict, Tuple, Any, Optional, Set
import base64
import re
from waitress import serve
import threading
from itertools import groupby
import oracledb
import platform

# --- 从新模块导入 ---
import config
from image_utils import read_image_cv2, draw_ocr_results_on_image
from yolo_handler import YoloHandler
from ocr_handler import OcrHandler
from database_handler import DatabaseHandler
# 移除 layout_and_state_manager 的导入，因为它已被废弃
# from layout_and_state_manager import LayoutStateManager

# --- 全局变量 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# 核心处理器
yolo_predictor: Optional[YoloHandler] = None
ocr_predictor: Optional[OcrHandler] = None
db_handler: Optional[DatabaseHandler] = None

# 核心数据缓存
session_data_store: Dict[str, Any] = {}
VALID_OBU_CODES_CACHE: Set[str] = set()
CACHE_LOCK = threading.Lock()
SESSION_CLEANUP_INTERVAL = timedelta(hours=config.SESSION_CLEANUP_HOURS)

# --- 日志设置 ---
def setup_logging(app_instance):
    if not os.path.exists(config.LOG_DIR):
        try: os.makedirs(config.LOG_DIR)
        except OSError as e: print(f"Error creating log directory {config.LOG_DIR}: {e}")
    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT, encoding='utf-8')

    # --- 核心修正：定义一个精确到毫秒的、24小时制的专业日志格式 ---
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    log_level_from_config = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == file_handler.baseFilename for h in app_instance.logger.handlers):
        app_instance.logger.addHandler(file_handler)

    app_instance.logger.setLevel(log_level_from_config)
    app_instance.logger.info(f"日志级别已设置为: {config.LOG_LEVEL}")
    app_instance.logger.info(f"Flask应用日志系统已启动。版本: {config.APP_VERSION}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# --- 核心裁决引擎辅助函数 (V18.1) ---
def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2): return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def analyze_evidence_pool(evidence_pool: Dict[str, Any]) -> Dict[str, Any]:
    analysis = {"total_count": len(evidence_pool), "segments": [], "is_chaotic": False, "is_pure_and_full": False}
    if analysis["total_count"] < config.MIN_SEGMENT_MEMBERS: return analysis
    try: sorted_obus = sorted([int(obu) for obu in evidence_pool.keys()])
    except (ValueError, TypeError): return analysis
    segments = []
    if sorted_obus:
        current_segment = [sorted_obus[0]]
        for i in range(1, len(sorted_obus)):
            if sorted_obus[i] - sorted_obus[i-1] <= config.SEGMENT_GAP_THRESHOLD:
                current_segment.append(sorted_obus[i])
            else:
                if len(current_segment) >= config.MIN_SEGMENT_MEMBERS: segments.append(current_segment)
                current_segment = [sorted_obus[i]]
        if len(current_segment) >= config.MIN_SEGMENT_MEMBERS: segments.append(current_segment)
    analysis["segments"] = segments
    if not segments or len(segments) > config.MAX_SEGMENTS_THRESHOLD: analysis["is_chaotic"] = True
    if analysis["total_count"] > config.PURITY_CHECK_THRESHOLD:
        for seg in segments:
            if len(seg) >= config.PURITY_CHECK_THRESHOLD and (seg[-1] - seg[0] == len(seg) - 1):
                analysis["is_pure_and_full"] = True
                analysis["pure_segment"] = seg
                break
    return analysis

def adjudicate_candidate(candidate: str, analysis_context: Dict[str, Any], logger: Any) -> bool:
    log_prefix = f"裁决 '{candidate}':"
    if analysis_context["is_pure_and_full"]:
        is_in_pure_segment = int(candidate) in analysis_context["pure_segment"]
        if not is_in_pure_segment: logger.debug(f"{log_prefix} [拒绝] - 未通过“满溢纯净”规则。")
        else: logger.debug(f"{log_prefix} [通过] - 符合“满溢纯净”规则。")
        return is_in_pure_segment
    if not config.ENABLE_HAMMING_CHECK or analysis_context["is_chaotic"]:
        logger.debug(f"{log_prefix} [跳过汉明] - 系统处于混沌模式或已禁用汉明检查。")
        return True
    if not analysis_context["segments"]:
        logger.debug(f"{log_prefix} [跳过汉明] - 无有效号段可供比对，默认通过。")
        return True
    for i, segment in enumerate(analysis_context["segments"]):
        min_dist = float('inf')
        for member_int in segment:
            member_str = f"{member_int:016d}"
            if len(candidate) == len(member_str):
                dist = hamming_distance(candidate, member_str)
                if dist < min_dist: min_dist = dist
        logger.debug(f"{log_prefix} 与号段{i+1} (共{len(segment)}个) 的最小汉明距离为 {min_dist}。")
        if min_dist <= config.HAMMING_THRESHOLD:
            logger.debug(f"{log_prefix} [通过] - 汉明距离 {min_dist} <= 阈值 {config.HAMMING_THRESHOLD}。")
            return True
    logger.debug(f"{log_prefix} [拒绝] - 与所有已知号段的汉明距离都过大。")
    return False

def extract_and_correct_candidates(raw_text: str, logger: Any) -> List[str]:
    if not raw_text: return []
    pattern = r'[A-Z0-9-]{16,20}'
    initial_candidates = re.findall(pattern, raw_text)
    if not initial_candidates: return []
    corrected_candidates = []
    for cand in initial_candidates:
        temp_cand = cand
        if config.ENABLE_HEADER_CORRECTION and not temp_cand.startswith(config.CORRECTION_HEADER_PREFIX):
            if temp_cand.startswith(('S', '6', '8', 'B')):
                 temp_cand = config.CORRECTION_HEADER_PREFIX + temp_cand[len(config.CORRECTION_HEADER_PREFIX):]
        if config.ENABLE_OCR_CORRECTION and config.OCR_HEURISTIC_REPLACEMENTS:
            for char_to_replace, replacement in config.OCR_HEURISTIC_REPLACEMENTS.items():
                temp_cand = temp_cand.replace(char_to_replace, replacement)
        temp_cand = temp_cand.replace('-', '')
        if len(temp_cand) == 16 and temp_cand.isdigit():
            corrected_candidates.append(temp_cand)
        else:
            logger.debug(f"候选 '{cand}' 修正后为 '{temp_cand}'，因格式不符被抛弃。")
    return corrected_candidates

# --- 核心图像处理逻辑 (V19.0_Streamlined) ---
def process_image_with_ocr_logic(
    image_path: str,
    session_id: str,
    logger: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float], List[Dict[str, str]], Optional[str]]:
    log_prefix = f"会话 {session_id} (process_image V{config.APP_VERSION}):"
    logger.info(f"{log_prefix} 开始处理图片 {os.path.basename(image_path)}")
    timing_profile, warnings_list, annotated_image_base64_str = {}, [], None
    t_start_overall = time.time()
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"{log_prefix} 严重错误 - 未找到会话数据！")
        return [], [], {"error": "Session data lost"}, [{"message": "会话数据丢失。"}], None
    global yolo_predictor, ocr_predictor, VALID_OBU_CODES_CACHE, CACHE_LOCK
    if not yolo_predictor or not ocr_predictor:
        logger.critical(f"{log_prefix} YOLO或OCR核心处理器未初始化！")
        return [], [], {"error": "Server not ready"}, [{"message": "服务内部错误(YOLO/OCR)。"}], None
    try:
        session["frame_count"] += 1
        current_frame_num = session["frame_count"]
        session["last_activity"] = datetime.now()
        t_step = time.time()
        original_image = read_image_cv2(image_path)
        timing_profile['1_image_reading'] = time.time() - t_step
        t_step = time.time()
        yolo_detections = yolo_predictor.detect(original_image)
        timing_profile['2_yolo_detection'] = time.time() - t_step
        t_step = time.time()
        ocr_tasks_for_pool, ocr_input_metadata = ocr_predictor.prepare_ocr_tasks_from_detections(
            original_image, yolo_detections, session_id, current_frame_num, config.SAVE_TRAINING_ROI_IMAGES
        )
        raw_ocr_pool_results = ocr_predictor.recognize_prepared_tasks(ocr_tasks_for_pool)
        final_ocr_results_list = ocr_predictor.consolidate_ocr_results(raw_ocr_pool_results, ocr_input_metadata)
        timing_profile['3_ocr_processing'] = time.time() - t_step
        logger.info(f"{log_prefix} 执行V19.0最终裁决与证据累积流程。")
        t_step_adjudication = time.time()
        evidence_pool = session.get("evidence_pool", {})
        with CACHE_LOCK: local_valid_codes = VALID_OBU_CODES_CACHE.copy()
        analysis_context = analyze_evidence_pool(evidence_pool)
        logger.info(f"{log_prefix} 情景分析: 混沌模式={analysis_context['is_chaotic']}, "
                    f"纯净满溢={analysis_context['is_pure_and_full']}, "
                    f"识别号段数={len(analysis_context['segments'])}")
        for ocr_item in final_ocr_results_list:
            if not ocr_item: continue
            raw_text = ocr_item.get("ocr_final_text", "").strip()
            corrected_candidates = extract_and_correct_candidates(raw_text, logger)
            if not corrected_candidates:
                ocr_item['status'] = 'failed_extraction'
                continue
            final_candidate = None
            for cand in corrected_candidates:
                if cand not in local_valid_codes: continue
                if adjudicate_candidate(cand, analysis_context, logger):
                    final_candidate = cand
                    break
            if final_candidate:
                if final_candidate not in evidence_pool:
                    evidence_pool[final_candidate] = {"count": 0, "first_seen_frame": current_frame_num}
                evidence_pool[final_candidate]["count"] += 1
                evidence_pool[final_candidate]["last_seen_frame"] = current_frame_num
                ocr_item['status'] = 'pending'
                if evidence_pool[final_candidate]["count"] >= config.PROMOTION_THRESHOLD:
                    ocr_item['status'] = 'confirmed'
                ocr_item['final_corrected_text'] = final_candidate
            else:
                ocr_item['status'] = 'failed_adjudication'
        session["evidence_pool"] = evidence_pool
        confirmed_results_list, pending_results_list = [], []
        map_code_to_ocr_item = {item['final_corrected_text']: item for item in final_ocr_results_list if 'final_corrected_text' in item}
        for obu_code, evidence in sorted(evidence_pool.items()):
            item_to_add = {"text": obu_code, "count": evidence["count"]}
            if evidence["count"] >= config.PROMOTION_THRESHOLD:
                confirmed_results_list.append(item_to_add)
            else:
                ocr_item_ref = map_code_to_ocr_item.get(obu_code)
                if ocr_item_ref: item_to_add['box'] = ocr_item_ref.get('bbox_yolo_abs')
                pending_results_list.append(item_to_add)
        logger.info(f"{log_prefix} 证据累积完成。确信: {len(confirmed_results_list)} 个, 待定: {len(pending_results_list)} 个。")
        if config.SAVE_PROCESS_PHOTOS:
            annotated_img_full_size = draw_ocr_results_on_image(original_image, yolo_detections, final_ocr_results_list)
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
                resized_annotated_img = cv2.resize(annotated_img_full_size, (target_w, target_h_ann), interpolation=cv2.INTER_AREA if scale_ann < 1 else cv2.INTER_LINEAR)
                retval, buffer = cv2.imencode('.jpg', resized_annotated_img, [cv2.IMWRITE_JPEG_QUALITY, config.SCATTERED_MODE_IMAGE_JPG_QUALITY])
                if retval: annotated_image_base64_str = base64.b64encode(buffer).decode('utf-8')
            except Exception as e_scatter_draw: logger.error(f"{log_prefix} 生成或编码缩小标注图时发生错误: {e_scatter_draw}", exc_info=True)
        if config.SAVE_TRAINING_ROI_IMAGES:
            session_training_roi_dir = os.path.join(config.PROCESS_PHOTO_DIR, "training_rois", session_id)
            if not os.path.exists(session_training_roi_dir): os.makedirs(session_training_roi_dir, exist_ok=True)
            label_file_path = os.path.join(session_training_roi_dir, "label.txt")
            with open(label_file_path, 'a', encoding='utf-8') as f:
                for ocr_item in final_ocr_results_list:
                    corrected_text = ocr_item.get('final_corrected_text')
                    if corrected_text:
                        roi_filename = f"f{current_frame_num}_yolo{ocr_item['original_index']}_h{config.OCR_TARGET_INPUT_HEIGHT}_w320.png"
                        f.write(f"{roi_filename}\t{corrected_text}\n")
        timing_profile['4_adjudication_processing'] = time.time() - t_step_adjudication
    except Exception as e:
        logger.error(f"{log_prefix} 处理图片时发生未知严重错误: {e}", exc_info=True)
        warnings_list.append({"message": f"服务内部错误，处理失败: {str(e)}", "code": "INTERNAL_SERVER_ERROR"})
    timing_profile['0_total_processing_function'] = time.time() - t_start_overall
    logger.info(f"{log_prefix} --- Timing profile for {os.path.basename(image_path)} ---")
    for key, val in sorted(timing_profile.items()): logger.info(f"  {key}: {val:.3f}s")
    return confirmed_results_list, pending_results_list, timing_profile, warnings_list, annotated_image_base64_str

# --- 新增：会话清理线程 ---
def cleanup_expired_sessions():
    while True:
        time.sleep(SESSION_CLEANUP_INTERVAL.total_seconds())
        with CACHE_LOCK:
            now = datetime.now()
            expired_sessions = [sid for sid, sdata in session_data_store.items() if now - sdata.get("last_activity", now) > SESSION_CLEANUP_INTERVAL]
            if expired_sessions:
                app.logger.info(f"会话清理：准备移除 {len(expired_sessions)} 个过期会话。")
                for sid in expired_sessions: del session_data_store[sid]
                app.logger.info(f"会话清理：完成。当前活动会话数: {len(session_data_store)}")

# --- 新增：健康检查接口 ---
@app.route('/health', methods=['GET'])
def health_check_route():
    logger = current_app.logger
    logger.info("接收到 /health 健康检查请求。")
    status_code = 200
    response = {"status": "ok", "checks": {}}
    # 1. 检查数据库连接池
    if db_handler and db_handler.pool:
        try:
            with db_handler.pool.acquire() as conn:
                response["checks"]["database_pool"] = "ok"
        except Exception as e:
            status_code = 503
            response["checks"]["database_pool"] = f"error: {str(e)}"
            logger.error("健康检查：数据库连接池获取连接失败。")
    else:
        status_code = 503
        response["checks"]["database_pool"] = "error: not initialized"
        logger.error("健康检查：数据库连接池未初始化。")
    # 2. 检查内存缓存
    with CACHE_LOCK:
        if VALID_OBU_CODES_CACHE:
            response["checks"]["memory_cache"] = f"ok, {len(VALID_OBU_CODES_CACHE)} items"
        else:
            status_code = 503
            response["checks"]["memory_cache"] = "error: empty or not loaded"
            logger.error("健康检查：内存OBU码缓存为空。")
    if status_code != 200:
        response["status"] = "error"
    return jsonify(response), status_code

# --- 新增：数据缓存刷新接口 ---
@app.route('/refresh-cache', methods=['POST'])
def refresh_cache_route():
    logger = current_app.logger
    logger.info("接收到 /refresh-cache 缓存刷新请求。")

    provided_key = request.headers.get('X-API-KEY')
    if provided_key != config.REFRESH_API_KEY:
        return jsonify({"error": "Invalid or missing API Key"}), 403
    logger.info("接收到缓存刷新请求...")
    global VALID_OBU_CODES_CACHE, db_handler
    if not db_handler:
        return jsonify({"error": "Database handler not initialized"}), 500
    new_data = db_handler.load_valid_obus()
    if new_data is not None:
        with CACHE_LOCK: VALID_OBU_CODES_CACHE = new_data
        logger.info(f"缓存刷新成功，新的OBU码数量: {len(VALID_OBU_CODES_CACHE)}")
        return jsonify({"message": "Cache refreshed successfully", "count": len(VALID_OBU_CODES_CACHE)}), 200
    else:
        return jsonify({"error": "Failed to load data from database"}), 500

# --- 新增：会话终审接口 ---
@app.route('/session/finalize', methods=['POST'])
def finalize_session_route():
    logger = current_app.logger
    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({"error": "session_id is required in JSON body"}), 400
    session_id = data['session_id']
    logger.info(f"接收到 /session/finalize 终审请求, 会话ID: {session_id}")
    with CACHE_LOCK: session = session_data_store.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    evidence_pool = session.get("evidence_pool", {})
    # 在终审时，我们将所有被目击过的OBU（无论次数）都视为最终结果
    final_results = [{"text": obu, "count": evi["count"]} for obu, evi in evidence_pool.items()]
    final_results_sorted = sorted(final_results, key=lambda x: x['text'])
    with CACHE_LOCK:
        if session_id in session_data_store:
            del session_data_store[session_id]
            logger.info(f"会话 {session_id}: 终审完成并已清理。")
    response_data = {"message": "Session finalized successfully.", "session_id": session_id, "total_count": len(final_results_sorted), "final_results": final_results_sorted}
    return jsonify(response_data), 200

# --- Flask 路由 (核心识别接口) ---
@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger

    # --- 新增：在所有逻辑开始前，先记录接口被调用 ---
    session_id_from_form = request.form.get('session_id', 'N/A')
    filename_from_form = request.files.get('file').filename if 'file' in request.files else 'N/A'
    logger.info(f"接收到 /predict 请求, 会话ID: {session_id_from_form}, 文件名: {filename_from_form}")

    # --- 新增：终极诊断日志 ---
    # 打印出请求中所有的表单数据和文件信息，让我们看看到底收到了什么
    logger.info(f"收到的表单数据 (request.form): {request.form}")
    logger.info(f"收到的文件信息 (request.files): {request.files}")
    # ---------------------------

    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file"}), 400

    original_filename_for_log = secure_filename(file.filename)
    upload_path = None  # 初始化 upload_path

    try:
        # --- 线程安全修改：将所有与 session_data_store 相关的操作放入锁内 ---
        with CACHE_LOCK:
            if session_id not in session_data_store:
                logger.info(f"会话 {session_id}: 新建会话。")
                session_data_store[session_id] = {
                    "evidence_pool": {},
                    "frame_count": 0,
                    "last_activity": datetime.now()
                }
            # 在锁内安全地获取和更新会话信息
            session = session_data_store[session_id]
            frame_num_for_filename = session.get("frame_count", 0) + 1
            # 注意：frame_count 的递增现在已移至 process_image_with_ocr_logic 内部
            # 但文件名可以预先使用这个数字，因为 process_image_with_ocr_logic 会正式增加它

        # --- 文件保存操作移出锁，减少锁的持有时间 ---
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        name, ext = os.path.splitext(original_filename_for_log)
        filename_on_server = f"s{session_id[:8]}_f{frame_num_for_filename}_{name}_{timestamp}{ext}"

        # 确保上传目录存在
        if not os.path.exists(config.UPLOAD_FOLDER):
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        upload_path = os.path.join(config.UPLOAD_FOLDER, filename_on_server)
        file.save(upload_path)

        # --- 核心处理逻辑 ---
        # process_image_with_ocr_logic 内部已经有对 session 的读写，
        # 但由于它是在单个请求线程中被调用的，且我们已在外部确保了会话的创建是安全的，
        # 其内部对 session 的修改（如 frame_count++, evidence_pool.update）
        # 也是安全的，因为它们都发生在同一个线程的上下文中。
        # 真正的并发风险在于多个线程同时尝试修改同一个 session_id 的数据，
        # 外层的锁已经防止了这种情况。
        confirmed_res, pending_res, timings_res, warnings_res, base64_img_res = \
            process_image_with_ocr_logic(upload_path, session_id, logger)

        response_data = {
            "message": "File processed successfully.", "session_id": session_id,
            "received_filename": original_filename_for_log, "timing_profile_seconds": timings_res,
            "warnings": warnings_res, "confirmed_results": confirmed_res,
            "pending_results": pending_res, "current_frame_annotated_image_base64": base64_img_res,
            "session_status": "in_progress"
        }
        logger.info(f"会话 {session_id}: 处理完成。确信: {len(confirmed_res or [])}, 待定: {len(pending_res or [])}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_log}' 时发生严重错误: {e}", exc_info=True)
        # 考虑在这里清理可能已保存的上传文件
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                logger.info(f"已清理因错误而未处理的上传文件: {upload_path}")
            except OSError as e_remove:
                logger.error(f"清理上传文件 {upload_path} 时失败: {e_remove}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- 应用初始化与启动 ---
def initialize_global_handlers(app_logger):
    # 这个函数现在只负责初始化，不再负责启动，所以保持原样是安全的
    global yolo_predictor, ocr_predictor, db_handler, VALID_OBU_CODES_CACHE
    app_logger.info("--- 开始初始化全局处理器 ---")
    try:
        db_handler = DatabaseHandler(logger=app_logger)
        initial_obus = db_handler.load_valid_obus()
        if initial_obus is not None:
            with CACHE_LOCK: VALID_OBU_CODES_CACHE = initial_obus
        else: raise RuntimeError("Failed to load initial OBU codes from database.")
        yolo_predictor = YoloHandler(model_path=config.ONNX_MODEL_PATH, conf_threshold=config.YOLO_CONFIDENCE_THRESHOLD, iou_threshold=config.YOLO_IOU_THRESHOLD, min_area_px=config.YOLO_MIN_DETECTION_AREA_PX, max_area_factor=config.YOLO_MAX_DETECTION_AREA_FACTOR, coco_classes=config.YOLO_COCO_CLASSES, logger=app_logger)
        ocr_predictor = OcrHandler(onnx_model_path=config.OCR_ONNX_MODEL_PATH, keys_path=config.OCR_KEYS_PATH, num_workers=config.OCR_NUM_WORKERS, target_ocr_input_height=config.OCR_TARGET_INPUT_HEIGHT, digit_roi_y_offset_factor=config.OCR_DIGIT_ROI_Y_OFFSET_FACTOR, digit_roi_height_factor=config.OCR_DIGIT_ROI_HEIGHT_FACTOR, digit_roi_width_expand_factor=config.OCR_DIGIT_ROI_WIDTH_EXPAND_FACTOR, logger=app_logger)
        app_logger.info("--- 全局处理器初始化完成 ---")
    except Exception as e:
        app_logger.critical(f"全局处理器初始化失败: {e}", exc_info=True)
        raise

def cleanup_on_exit():
    # 这个函数也保持原样
    global ocr_predictor, db_handler
    if ocr_predictor and hasattr(ocr_predictor, 'close_pool'):
        print("应用退出，正在关闭OCR处理池...")
        ocr_predictor.close_pool()
    if db_handler and hasattr(db_handler, 'close_pool'):
        print("应用退出，正在关闭数据库连接池...")
        db_handler.close_pool()

# -----------------------------------------------------------------------------------
# 【核心修正】将所有“执行”逻辑，全部放回 if __name__ == '__main__' 的保护之下
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. 初始化Oracle客户端 (保持不变)
    try:
        if platform.system() != "Windows":
            oracledb.init_oracle_client(lib_dir="/opt/oracle/instantclient_21_13")
            print("Oracle Client (Thick Mode) initialized for Linux/Docker.")
        else:
            print("Running on Windows, using default 'Thin Mode' for Oracle connection.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Oracle Client: {e}")
        exit(1)

    # 2. 设置日志系统 (保持不变)
    setup_logging(app)

    # 3. 初始化所有核心处理器 (保持不变)
    try:
        initialize_global_handlers(app.logger)
    except Exception as e_init:
        app.logger.critical(f"应用启动失败，无法初始化核心处理器: {e_init}")
        exit(1)

    # 4. 注册清理和后台任务 (保持不变)
    atexit.register(cleanup_on_exit)
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    app.logger.info("后台会话清理线程已启动。")

    # 5. 检查目录 (保持不变)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    if not os.path.exists(config.PROCESS_PHOTO_DIR):
        os.makedirs(config.PROCESS_PHOTO_DIR, exist_ok=True)

    # -----------------------------------------------------------------------------------
    # 【核心修正】将Waitress的日志与Flask日志系统集成，以获得接入日志
    # -----------------------------------------------------------------------------------
    waitress_logger = logging.getLogger('waitress')
    waitress_logger.setLevel(logging.INFO)
    for handler in app.logger.handlers:
        waitress_logger.addHandler(handler)
    # -----------------------------------------------------------------------------------

    # 6. 最后，以生产模式启动 Waitress 服务器
    app.logger.info(f"服务版本 {config.APP_VERSION} 启动中... 使用高性能生产服务器 Waitress。")
    serve(app, host='0.0.0.0', port=5000)