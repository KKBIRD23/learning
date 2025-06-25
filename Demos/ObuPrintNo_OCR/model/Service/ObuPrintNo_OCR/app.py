# app.py (V21.2 - "Libra")
import os
import cv2
import numpy as np
import time
import traceback
import multiprocessing
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, current_app, g, has_request_context
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

# --- 全局变量 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

yolo_predictor: Optional[YoloHandler] = None
ocr_predictor: Optional[OcrHandler] = None
db_handler: Optional[DatabaseHandler] = None

session_data_store: Dict[str, Any] = {}
VALID_OBU_CODES_CACHE: Set[str] = set()
CACHE_LOCK = threading.Lock()
SESSION_CLEANUP_INTERVAL = timedelta(hours=config.SESSION_CLEANUP_HOURS)

# --- 【核心修正】日志系统优化，解决上下文错误 ---
class RequestContextFilter(logging.Filter):
    """
    一个自定义的日志过滤器，用于向每条日志记录中注入请求相关的上下文信息。
    """
    def filter(self, record):
        # 【核心修正】在尝试访问g对象前，先判断是否存在请求上下文
        if has_request_context():
            # 如果在请求上下文中，正常从g获取req_id
            record.req_id = g.get('req_id', '-------')
        else:
            # 如果不在请求上下文中（如应用启动时），赋予一个特殊的ID
            record.req_id = 'SYSTEM'
        return True

def setup_logging(app_instance):
    """
    配置应用的日志系统，包括文件轮转、格式化和我们新增的上下文过滤器。
    """
    if not os.path.exists(config.LOG_DIR):
        try: os.makedirs(config.LOG_DIR)
        except OSError as e: print(f"Error creating log directory {config.LOG_DIR}: {e}")

    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILE)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT, encoding='utf-8')

    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d %(levelname)s [%(req_id)s]: %(message)s [in %(pathname)s:%(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestContextFilter())

    log_level_from_config = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == file_handler.baseFilename for h in app_instance.logger.handlers):
        app_instance.logger.addHandler(file_handler)

    app_instance.logger.setLevel(log_level_from_config)
    # 注意：这里的日志记录现在是安全的，因为过滤器会处理没有请求上下文的情况
    app_instance.logger.info(f"日志级别已设置为: {config.LOG_LEVEL}")
    app_instance.logger.info(f"Flask应用日志系统已启动。版本: {config.APP_VERSION}")

# --- 文件辅助函数 ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

# --- 核心裁决引擎辅助函数 ---
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

def adjudicate_candidate(candidate: str, analysis_context: Dict[str, Any], logger: Any, locked_segment: Optional[List[int]] = None) -> bool:
    log_prefix = f"裁决 '{candidate}':"
    if locked_segment:
        try:
            candidate_int = int(candidate)
            if locked_segment[0] - config.GUESS_RANGE <= candidate_int <= locked_segment[-1] + config.GUESS_RANGE:
                 logger.debug(f"{log_prefix} [通过] - 符合锁定的号段规则。")
                 return True
            else:
                 logger.debug(f"{log_prefix} [拒绝] - 不在锁定的号段范围内。")
                 return False
        except (ValueError, TypeError):
            logger.debug(f"{log_prefix} [拒绝] - 候选码无法转为整数以进行号段锁定比较。")
            return False
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

def process_image_with_ocr_logic(
    image_path: str,
    session_id: str,
    logger: Any
) -> Dict[str, Any]:
    log_prefix = f"会话 {session_id}:"
    logger.info(f"{log_prefix} 开始处理图片 {os.path.basename(image_path)}")
    timing_profile, warnings_list = {}, []
    t_start_overall = time.time()
    session = session_data_store.get(session_id)
    if not session:
        logger.error(f"{log_prefix} 严重错误 - 未找到会话数据！")
        warnings_list.append({"message": "会话数据丢失，请重新开始", "code": "SESSION_LOST"})
        return {"status": "error", "data": {"warnings": warnings_list}}
    global yolo_predictor, ocr_predictor, VALID_OBU_CODES_CACHE, CACHE_LOCK
    if not yolo_predictor or not ocr_predictor:
        logger.critical(f"{log_prefix} YOLO或OCR核心处理器未初始化！")
        warnings_list.append({"message": "服务内部错误(YOLO/OCR)", "code": "HANDLER_NOT_READY"})
        return {"status": "error", "data": {"warnings": warnings_list}}
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
        evidence_pool = session.get("evidence_pool", {})
        with CACHE_LOCK: local_valid_codes = VALID_OBU_CODES_CACHE.copy()

        if session.get("mode") == config.RECOGNITION_MODE_FULL_PLATE and current_frame_num == 1:
            logger.info(f"{log_prefix} [整版识别-首帧] 启动预选引擎...")
            first_frame_evidence = {}
            for ocr_item in final_ocr_results_list:
                raw_text = ocr_item.get("ocr_final_text", "").strip()
                corrected_candidates = extract_and_correct_candidates(raw_text, logger)
                for cand in corrected_candidates:
                    if cand in local_valid_codes:
                        first_frame_evidence.setdefault(cand, {"count": 0})
                        first_frame_evidence[cand]["count"] += 1

            analysis_context = analyze_evidence_pool(first_frame_evidence)
            segments = analysis_context.get("segments", [])
            dominant_segment = sorted(segments, key=len, reverse=True)[0] if segments else None

            candidate_segments = []
            if dominant_segment:
                center_num = dominant_segment[len(dominant_segment) // 2]
                prefix = str(center_num)[:-2]
                candidate_start_01 = int(prefix + "01")
                candidate_start_51 = int(prefix + "51")
                votes = {candidate_start_01: 0, candidate_start_51: 0}
                voters = {int(k) for k in first_frame_evidence.keys()}
                for start_num in votes.keys():
                    for i in range(config.EXPECTED_OBU_COUNT):
                        if (start_num + i) in voters:
                            votes[start_num] += 1
                sorted_candidates = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                candidate_segments = [str(k) for k, v in sorted_candidates]
                logger.info(f"{log_prefix} 预选完成。候选号段及票数: {sorted_candidates}")
            else:
                logger.warning(f"{log_prefix} 首帧未找到任何有效号段，无法生成候选。")

            session["status"] = "awaiting_confirmation"
            session["evidence_pool"] = first_frame_evidence

            # 将首帧的所有必要信息存入会话，供确认后使用
            session["first_frame_cache"] = {
                "image_path": image_path,
                "yolo_detections": yolo_detections,
                "final_ocr_results_list": final_ocr_results_list
            }

            # 在等待确认时，不返回任何标注图，只返回候选号段
            return {"status": "awaiting_confirmation", "data": {"candidate_segments": candidate_segments}}

        # --- 后续帧或零散模式的处理逻辑 ---
        locked_segment_info = session.get("locked_segment_info")
        analysis_context = analyze_evidence_pool(evidence_pool)
        newly_added_codes = []
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
                if adjudicate_candidate(cand, analysis_context, logger, locked_segment_info):
                    final_candidate = cand
                    break
            if final_candidate:
                if final_candidate not in evidence_pool: newly_added_codes.append(final_candidate)
                evidence_pool.setdefault(final_candidate, {"count": 0, "first_seen_frame": current_frame_num})
                evidence_pool[final_candidate]["count"] += 1
                evidence_pool[final_candidate]["last_seen_frame"] = current_frame_num
                ocr_item['status'] = 'pending'
                if evidence_pool[final_candidate]["count"] >= config.PROMOTION_THRESHOLD:
                    ocr_item['status'] = 'confirmed'
                ocr_item['final_corrected_text'] = final_candidate
            else:
                ocr_item['status'] = 'failed_adjudication'
        session["evidence_pool"] = evidence_pool

        confirmed_results_list, pending_results_list = _get_results_from_pool(evidence_pool)

        if newly_added_codes:
            logger.info(f"{log_prefix} [法证日志] 本帧新增OBU码({len(newly_added_codes)}个): {sorted(newly_added_codes)}")

        annotated_image_base64_str = None
        if config.SAVE_PROCESS_PHOTOS:
            annotated_img_full_size = draw_ocr_results_on_image(original_image, yolo_detections, final_ocr_results_list)
            try:
                target_w = config.SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH
                orig_h_ann, orig_w_ann = annotated_img_full_size.shape[:2]
                scale_ann = target_w / orig_w_ann
                target_h_ann = int(orig_h_ann * scale_ann)
                resized_annotated_img = cv2.resize(annotated_img_full_size, (target_w, target_h_ann))
                retval, buffer = cv2.imencode('.jpg', resized_annotated_img, [cv2.IMWRITE_JPEG_QUALITY, config.SCATTERED_MODE_IMAGE_JPG_QUALITY])
                if retval: annotated_image_base64_str = base64.b64encode(buffer).decode('utf-8')
            except Exception as e_draw: logger.error(f"{log_prefix} 生成标注图时发生错误: {e_draw}", exc_info=True)

        timing_profile['0_total_processing_function'] = time.time() - t_start_overall
        logger.info(f"{log_prefix} --- Timing profile for {os.path.basename(image_path)} ---")
        for key, val in sorted(timing_profile.items()): logger.info(f"  {key}: {val:.3f}s")

        response_data = {
            "confirmed_results": confirmed_results_list,
            "pending_results": pending_results_list,
            "current_frame_annotated_image_base64": annotated_image_base64_str,
            "locked_segment_info": str(locked_segment_info[0]) if locked_segment_info else None,
            "timing_profile_seconds": timing_profile,
            "warnings": warnings_list
        }
        return {"status": session["status"], "data": response_data}
    except Exception as e:
        logger.error(f"{log_prefix} 处理图片时发生未知严重错误: {e}", exc_info=True)
        warnings_list.append({"message": f"服务内部错误: {str(e)}", "code": "INTERNAL_SERVER_ERROR"})
        return {"status": "error", "data": {"warnings": warnings_list}}

# --- 后台任务与辅助接口 ---
def cleanup_expired_sessions():
    while True:
        time.sleep(SESSION_CLEANUP_INTERVAL.total_seconds())
        with CACHE_LOCK:
            now = datetime.now()
            expired_sessions = [sid for sid, sdata in session_data_store.items() if now - sdata.get("last_activity", now) > SESSION_CLEANUP_INTERVAL]
            if expired_sessions:
                # 使用应用上下文来安全地记录日志
                with app.app_context():
                    app.logger.info(f"会话清理：准备移除 {len(expired_sessions)} 个过期会话。")
                    for sid in expired_sessions: del session_data_store[sid]
                    app.logger.info(f"会话清理：完成。当前活动会话数: {len(session_data_store)}")

# --- 【核心重构】使用Flask的请求钩子来管理日志上下文 ---
@app.before_request
def before_request_logging():
    """在每次请求开始前执行，注入请求ID并打印开始日志。"""
    g.req_id = uuid.uuid4().hex[:8]
    logger = current_app.logger
    separator = "=" * 20
    logger.info(f"{separator} [{g.req_id}] {request.method} {request.path} START {separator}")

@app.after_request
def after_request_logging(response):
    """在每次请求结束后执行，打印结束日志。"""
    logger = current_app.logger
    separator = "=" * 20
    logger.info(f"{separator} [{g.req_id}] {request.method} {request.path} END ({response.status}) {separator}\n")
    return response

# --- API 路由实现 ---
@app.route('/health', methods=['GET'])
def health_check_route():
    logger = current_app.logger
    logger.info("接收到 /health 健康检查请求。")
    status_code = 200
    response = {"status": "ok", "checks": {}}
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

@app.route('/refresh-cache', methods=['POST'])
def refresh_cache_route():
    logger = current_app.logger
    provided_key = request.headers.get('X-API-KEY')
    if provided_key != config.REFRESH_API_KEY:
        logger.warning(f"接收到无效的缓存刷新请求，API Key不匹配。")
        return jsonify({"error": "Invalid or missing API Key"}), 403
    logger.info("接收到合法的缓存刷新请求...")
    global VALID_OBU_CODES_CACHE, db_handler
    if not db_handler:
        logger.error("缓存刷新失败：数据库处理器未初始化。")
        return jsonify({"error": "Database handler not initialized"}), 500
    new_data = db_handler.load_valid_obus()
    if new_data is not None:
        with CACHE_LOCK: VALID_OBU_CODES_CACHE = new_data
        logger.info(f"缓存刷新成功，新的OBU码数量: {len(VALID_OBU_CODES_CACHE)}")
        return jsonify({"message": "Cache refreshed successfully", "count": len(VALID_OBU_CODES_CACHE)}), 200
    else:
        logger.error("缓存刷新失败：从数据库加载数据时发生错误。")
        return jsonify({"error": "Failed to load data from database"}), 500

def _get_results_from_pool(evidence_pool: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    一个辅助函数，根据给定的证据池，生成确信和待定列表。
    """
    confirmed_results_list, pending_results_list = [], []
    for obu_code, evidence in sorted(evidence_pool.items()):
        item_to_add = {"text": obu_code, "count": evidence["count"]}
        if evidence["count"] >= config.PROMOTION_THRESHOLD:
            confirmed_results_list.append(item_to_add)
        else:
            pending_results_list.append(item_to_add)
    return confirmed_results_list, pending_results_list

@app.route('/session/confirm_segment', methods=['POST'])
# app.py (V21.4 - "Libra") - 修正部分

@app.route('/session/confirm_segment', methods=['POST'])
# app.py (V21.4 - "Libra") - 修正部分

@app.route('/session/confirm_segment', methods=['POST'])
def confirm_segment_route():
    logger = current_app.logger # 获取当前应用的logger实例
    data = request.get_json()
    if not data or 'session_id' not in data or 'chosen_segment_key' not in data:
        return jsonify({"error": "session_id and chosen_segment_key are required"}), 400

    session_id = data['session_id']
    chosen_key = data['chosen_segment_key']
    logger.info(f"会话 {session_id}: 接收到号段确认请求，选择的key为 '{chosen_key}'")

    with CACHE_LOCK:
        session = session_data_store.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        if session.get("status") != "awaiting_confirmation":
            return jsonify({"error": "Session is not awaiting confirmation"}), 400

        first_frame_cache = session.get("first_frame_cache")
        if not first_frame_cache:
            logger.error(f"会话 {session_id}: 严重错误 - 在确认号段时，首帧缓存丢失。")
            return jsonify({"error": "First frame cache is missing, cannot process confirmation."}), 500

        try:
            start_num = int(chosen_key)
            locked_segment = [start_num + i for i in range(config.EXPECTED_OBU_COUNT)]

            session["status"] = "locked"
            session["locked_segment_info"] = locked_segment

            logger.info(f"会话 {session_id}: 号段已由用户确认并锁定，起始码: {start_num}")

            # 从首帧缓存的原始证据池中，根据新锁定的号段进行过滤
            original_first_frame_evidence = session.get("evidence_pool", {}) # 这是首帧处理时生成的
            filtered_evidence_pool_for_first_frame = {}
            for obu_str, evidence_data in original_first_frame_evidence.items():
                try:
                    obu_int = int(obu_str)
                    if locked_segment[0] <= obu_int <= locked_segment[-1]:
                        filtered_evidence_pool_for_first_frame[obu_str] = evidence_data
                except (ValueError, TypeError):
                    continue

            session["evidence_pool"] = filtered_evidence_pool_for_first_frame
            confirmed, pending = _get_results_from_pool(filtered_evidence_pool_for_first_frame)

            # 为绘图准备一个带有正确状态的全新列表
            drawing_results_list = []
            cached_ocr_results_raw = first_frame_cache.get("final_ocr_results_list", [])

            for ocr_item_raw in cached_ocr_results_raw:
                item_for_drawing = ocr_item_raw.copy()

                # 【关键修正】对从缓存中取出的原始OCR结果，进行修正和提取
                raw_text_from_cache = item_for_drawing.get("ocr_final_text", "").strip()
                corrected_candidates_for_this_box = extract_and_correct_candidates(raw_text_from_cache, logger)

                # 通常我们只关心第一个最可能的修正结果用于状态判断
                cand_for_status = None
                if corrected_candidates_for_this_box:
                    cand_for_status = corrected_candidates_for_this_box[0]

                # 用修正后的候选码 (cand_for_status) 来判断状态
                if cand_for_status and cand_for_status in filtered_evidence_pool_for_first_frame:
                    count = filtered_evidence_pool_for_first_frame[cand_for_status].get("count", 0)
                    if count >= config.PROMOTION_THRESHOLD:
                        item_for_drawing['status'] = 'confirmed'
                    else:
                        item_for_drawing['status'] = 'pending'
                    item_for_drawing['final_corrected_text'] = cand_for_status # 确保绘图函数能用到
                else:
                    item_for_drawing['status'] = 'failed_adjudication'
                drawing_results_list.append(item_for_drawing)

            base64_img = None
            original_image_cv2 = read_image_cv2(first_frame_cache["image_path"])
            annotated_img_full_size = draw_ocr_results_on_image(
                original_image_cv2,
                first_frame_cache.get("yolo_detections", []),
                drawing_results_list
            )

            try:
                target_w = config.SCATTERED_MODE_ANNOTATED_IMAGE_WIDTH
                orig_h_ann, orig_w_ann = annotated_img_full_size.shape[:2]
                if orig_w_ann > 0:
                    scale_ann = target_w / orig_w_ann
                    target_h_ann = int(orig_h_ann * scale_ann)
                    resized_annotated_img = cv2.resize(annotated_img_full_size, (target_w, target_h_ann))
                    retval, buffer = cv2.imencode('.jpg', resized_annotated_img, [cv2.IMWRITE_JPEG_QUALITY, config.SCATTERED_MODE_IMAGE_JPG_QUALITY])
                    if retval:
                        base64_img = base64.b64encode(buffer).decode('utf-8')
                    else:
                        logger.error(f"会话 {session_id}: 在confirm_segment中编码缩放后的标注图失败。")
                else:
                    logger.error(f"会话 {session_id}: 在confirm_segment中，原标注图宽度为0，无法缩放。")
            except Exception as e_draw:
                logger.error(f"会话 {session_id}: 在confirm_segment中生成或编码缩放标注图时发生错误: {e_draw}", exc_info=True)

            logger.info(f"会话 {session_id}: 首帧确认后，返回 {len(confirmed)} 个确信，{len(pending)} 个待定结果。")

            session.pop("ambiguous_segments_cache", None)
            session.pop("first_frame_cache", None)

            return jsonify({
                "message": "Segment confirmed and locked successfully.",
                "session_id": session_id,
                "session_status": "locked",
                "received_filename": os.path.basename(first_frame_cache["image_path"]),
                "confirmed_results": confirmed,
                "pending_results": pending,
                "current_frame_annotated_image_base64": base64_img,
                "locked_segment_info": chosen_key,
                "timing_profile_seconds": {},
                "warnings": []
            }), 200
        except Exception as e:
            logger.error(f"会话 {session_id}: 在确认号段时发生严重错误: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred during confirmation: {str(e)}"}), 500

@app.route('/session/finalize', methods=['POST'])
def finalize_session_route():
    logger = current_app.logger
    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({"error": "session_id is required in JSON body"}), 400
    session_id = data['session_id']
    logger.info(f"接收到 /session/finalize 终审请求, 会话ID: {session_id}")

    with CACHE_LOCK:
        session = session_data_store.get(session_id)
        if not session:
            return jsonify({"error": "Session not found or already finalized"}), 404

        evidence_pool = session.get("evidence_pool", {})
        final_results = []

        if session.get("mode") == config.RECOGNITION_MODE_FULL_PLATE:
            logger.info(f"会话 {session_id}: 以“整版识别”模式进入终审。")
            locked_segment = session.get("locked_segment_info")
            if not locked_segment:
                logger.error(f"会话 {session_id}: 处于整版模式，但无锁定号段，无法终审。")
                return jsonify({"error": "Cannot finalize a 'full_plate' session without a confirmed segment."}), 400

            for obu_int in locked_segment:
                obu_str = f"{obu_int:016d}"
                count = evidence_pool.get(obu_str, {}).get("count", 0)
                if count == 0:
                    logger.warning(f"终审：号码 {obu_str} 是基于锁定的号段推断得出的，从未在图像中被直接识别。")
                final_results.append({"text": obu_str, "count": count})
        else:
            logger.info(f"会话 {session_id}: 以“零散识别”模式进入终审。")
            for obu, evi in evidence_pool.items():
                if evi.get("count", 0) >= config.PROMOTION_THRESHOLD:
                    final_results.append({"text": obu, "count": evi["count"]})
            final_results = sorted(final_results, key=lambda x: x['text'])

        if session_id in session_data_store:
            del session_data_store[session_id]
            logger.info(f"会话 {session_id}: 终审完成并已清理。")

    response_data = {
        "message": "Session finalized successfully.",
        "session_id": session_id,
        "total_count": len(final_results),
        "final_results": final_results
    }
    return jsonify(response_data), 200

@app.route('/predict', methods=['POST'])
def predict_image_route():
    logger = current_app.logger
    session_id_from_form = request.form.get('session_id', 'N/A')
    filename_from_form = request.files.get('file').filename if 'file' in request.files else 'N/A'
    recognition_mode = request.form.get('recognition_mode', config.DEFAULT_RECOGNITION_MODE)
    logger.info(f"接收到 /predict 请求。会话ID: {session_id_from_form}, 文件名: {filename_from_form}, 模式: {recognition_mode}")
    logger.debug(f"收到的表单数据 (request.form): {request.form}")
    logger.debug(f"收到的文件信息 (request.files): {request.files}")
    session_id = request.form.get('session_id')
    if not session_id: return jsonify({"error": "session_id is required"}), 400
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"error": "Invalid file"}), 400
    original_filename_for_log = secure_filename(file.filename)
    upload_path = None
    try:
        with CACHE_LOCK:
            if session_id not in session_data_store:
                logger.info(f"会话 {session_id}: 新建会话。")
                session_data_store[session_id] = {
                    "evidence_pool": {}, "frame_count": 0, "last_activity": datetime.now(),
                    "mode": recognition_mode, "status": "new", "locked_segment_info": None
                }
            session = session_data_store[session_id]
            if session.get("status") == "awaiting_confirmation":
                logger.warning(f"会话 {session_id}: 正在等待用户确认号段，拒绝处理新图片。")
                return jsonify({
                    "error": "Session is awaiting segment confirmation from the user.",
                    "session_status": "awaiting_confirmation",
                    "candidate_segments": session.get("ambiguous_segments_cache", [])
                }), 409
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename_on_server = f"s{session_id[:8]}_f{session.get('frame_count', 0) + 1}_{timestamp}_{original_filename_for_log}"
        if not os.path.exists(config.UPLOAD_FOLDER): os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        upload_path = os.path.join(config.UPLOAD_FOLDER, filename_on_server)
        file.save(upload_path)
        result = process_image_with_ocr_logic(upload_path, session_id, logger)
        response_data = {
            "session_id": session_id,
            "session_status": result.get("status"),
            "received_filename": original_filename_for_log,
            **result.get("data", {})
        }
        logger.info(f"会话 {session_id}: 处理完成，状态: {result.get('status')}")
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"会话 {session_id}: 处理图片 '{original_filename_for_log}' 时发生严重错误: {e}", exc_info=True)
        if upload_path and os.path.exists(upload_path):
            try: os.remove(upload_path)
            except OSError as e_remove: logger.error(f"清理上传文件失败: {e_remove}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "session_id": session_id}), 500

# --- 应用初始化与启动 ---
def initialize_global_handlers(app_logger):
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
    global ocr_predictor, db_handler
    if ocr_predictor and hasattr(ocr_predictor, 'close_pool'):
        print("应用退出，正在关闭OCR处理池...")
        ocr_predictor.close_pool()
    if db_handler and hasattr(db_handler, 'close_pool'):
        print("应用退出，正在关闭数据库连接池...")
        db_handler.close_pool()

if __name__ == '__main__':
    try:
        if platform.system() != "Windows":
            oracledb.init_oracle_client(lib_dir="/opt/oracle/instantclient_21_13")
            print("Oracle Client (Thick Mode) initialized for Linux/Docker.")
        else:
            print("Running on Windows, using default 'Thin Mode' for Oracle connection.")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Oracle Client: {e}")
        exit(1)
    setup_logging(app)
    try:
        initialize_global_handlers(app.logger)
    except Exception as e_init:
        app.logger.critical(f"应用启动失败，无法初始化核心处理器: {e_init}")
        exit(1)
    atexit.register(cleanup_on_exit)
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    app.logger.info("后台会话清理线程已启动。")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    if not os.path.exists(config.PROCESS_PHOTO_DIR):
        os.makedirs(config.PROCESS_PHOTO_DIR, exist_ok=True)
    waitress_logger = logging.getLogger('waitress')
    waitress_logger.setLevel(logging.INFO)
    for handler in app.logger.handlers:
        waitress_logger.addHandler(handler)
    app.logger.info(f"服务版本 {config.APP_VERSION} 启动中... 使用高性能生产服务器 Waitress。")
    serve(app, host='0.0.0.0', port=5000)