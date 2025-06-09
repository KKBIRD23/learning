# ocr_handler.py
import cv2
import numpy as np
import paddlex as pdx
import re
import multiprocessing
import os
import time
from typing import List, Dict, Tuple, Any

# 从 image_utils 导入必要的函数
from image_utils import crop_roi_from_image, preprocess_roi_for_ocr
# 从 config 导入新配置
from config import SAVE_TRAINING_ROI_IMAGES, PROCESS_PHOTO_DIR # 确保 PROCESS_PHOTO_DIR 被导入

# --- OCR Worker Initialization and Task ---
_worker_ocr_predictor_instance = None

def init_ocr_worker_process(ocr_model_dir_path: str, model_name_cfg: str = 'PP-OCRv5_server_rec'):
    """OCR工作进程的初始化函数。"""
    global _worker_ocr_predictor_instance
    worker_pid = os.getpid()
    # 使用 print 而不是 logger，因为在多进程的 worker 中配置和使用主进程的 logger 比较复杂
    print(f"[OCR Worker PID {worker_pid}] Initializing OCR predictor from: {ocr_model_dir_path}")
    try:
        _worker_ocr_predictor_instance = pdx.inference.create_predictor(
            model_dir=ocr_model_dir_path,
            model_name=model_name_cfg,
            device='cpu'
        )
        print(f"[OCR Worker PID {worker_pid}] OCR predictor initialized successfully.")
    except Exception as e:
        print(f"[OCR Worker PID {worker_pid}] OCR predictor initialization FAILED: {e}")
        _worker_ocr_predictor_instance = None

def ocr_task_for_worker_process(task_data_tuple: Tuple[int, np.ndarray, Tuple[int, int]]) -> Tuple[int, Dict[str, Any]]:
    """
    在工作进程中执行单个OCR任务。
    task_data_tuple 包含 (original_yolo_idx, roi_image_for_ocr, original_roi_shape_hw_before_our_preprocess)
    """
    global _worker_ocr_predictor_instance
    original_yolo_idx, roi_image_for_ocr, original_roi_shape_hw = task_data_tuple
    pid = os.getpid()
    start_time = time.time()

    # --- 日志：打印送入OCR模型的图像尺寸 ---
    if roi_image_for_ocr is not None:
        # 在worker中，我们通常用print，或者需要单独配置worker的日志
        print(f"[OCR Worker PID {pid}] DEBUG: original_yolo_idx={original_yolo_idx}, "
              f"Original ROI (H,W) before our preprocess: {original_roi_shape_hw}, "
              f"ROI送入PaddleX predictor前尺寸 (H,W): {roi_image_for_ocr.shape[:2]}")
    else:
        print(f"[OCR Worker PID {pid}] DEBUG: original_yolo_idx={original_yolo_idx}, ROI图像为None，无法送入模型。")
    # --- 结束日志 ---

    if roi_image_for_ocr is None: # 如果预处理后的图像是None
        return original_yolo_idx, {'rec_text': 'OCR_PREPROC_FAIL_IMG_NONE', 'rec_score': 0.0,
                                   'pid': pid, 'duration': time.time() - start_time}

    if _worker_ocr_predictor_instance is None:
        return original_yolo_idx, {'rec_text': 'OCR_WORKER_INIT_FAIL', 'rec_score': 0.0,
                                   'pid': pid, 'duration': time.time() - start_time}

    try:
        # PaddleOCR predict期望一个图像列表
        recognition_results_generator = _worker_ocr_predictor_instance.predict([roi_image_for_ocr])
        # 获取生成器的第一个（也是唯一一个）结果
        recognition_result_list = next(recognition_results_generator, None)

        final_recognition_dict = {'rec_text': '', 'rec_score': 0.0} # 默认值
        if recognition_result_list and isinstance(recognition_result_list, list) and len(recognition_result_list) > 0:
            # PaddleOCR PP-OCRv3/v4 server rec 模型通常返回 [{ 'rec_text': '...', 'rec_score': ...}]
            final_recognition_dict = recognition_result_list[0]
        elif recognition_result_list and isinstance(recognition_result_list, dict): # 兼容直接返回字典的情况
            final_recognition_dict = recognition_result_list

        return original_yolo_idx, {
            **final_recognition_dict, # 包含 rec_text, rec_score
            'pid': pid,
            'duration': time.time() - start_time
        }
    except Exception as e:
        print(f"[OCR Worker PID {pid}] OCR prediction FAILED for yolo_idx {original_yolo_idx}: {e}")
        return original_yolo_idx, {'rec_text': 'OCR_PREDICT_FAIL_IN_WORKER', 'rec_score': 0.0,
                                   'pid': pid, 'duration': time.time() - start_time, 'error_msg': str(e)}

class OcrHandler:
    def __init__(self,
                 rec_model_dir: str,
                 num_workers: int,
                 target_ocr_input_height: int,
                 digit_roi_y_offset_factor: float,
                 digit_roi_height_factor: float,
                 digit_roi_width_expand_factor: float,
                 logger: Any): # logger 是主进程的logger实例
        self.rec_model_dir = rec_model_dir
        self.num_workers = num_workers
        self.target_ocr_input_height = target_ocr_input_height
        self.digit_roi_y_offset_factor = digit_roi_y_offset_factor
        self.digit_roi_height_factor = digit_roi_height_factor
        self.digit_roi_width_expand_factor = digit_roi_width_expand_factor
        self.logger = logger # 主进程的logger

        self.ocr_processing_pool = None
        self.serial_ocr_predictor = None # 用于串行处理

        if self.num_workers > 1:
            self._initialize_ocr_pool()
        else: # 串行或单进程
            self._initialize_serial_ocr_predictor()

    def _initialize_ocr_pool(self):
        """初始化OCR并行处理池。"""
        if not os.path.exists(self.rec_model_dir):
            self.logger.critical(f"OCR Handler: Server OCR model directory for workers not found: {self.rec_model_dir}. OCR will fallback to serial or fail.")
            self.num_workers = 1 # Fallback to serial if model dir missing for pool
            self._initialize_serial_ocr_predictor() # 尝试初始化串行模式
            return

        self.logger.info(f"OCR Handler: Initializing global OCR processing pool with {self.num_workers} workers...")
        try:
            # 确保使用 'spawn' 启动方法，特别是在Windows上
            # 在类Unix系统上，'fork' 通常是默认且高效的，但在Windows上 'fork' 不可用
            start_method = multiprocessing.get_start_method(allow_none=True)
            if os.name == 'nt' and start_method != 'spawn': # 如果是Windows且当前不是spawn
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                    self.logger.info(f"OCR Handler: Multiprocessing start method set to 'spawn' for Windows.")
                except RuntimeError as e_mp_start:
                     self.logger.warning(f"OCR Handler: Could not set multiprocessing start method to 'spawn': {e_mp_start}. Using default: {start_method}")
            elif start_method is None and os.name == 'nt': # 如果没有设置过，Windows默认可能是forkserver或spawn
                 multiprocessing.set_start_method('spawn', force=True) # 强制spawn
                 self.logger.info(f"OCR Handler: Default start method was None, set to 'spawn' for Windows.")


            self.ocr_processing_pool = multiprocessing.Pool(
                processes=self.num_workers,
                initializer=init_ocr_worker_process,
                initargs=(self.rec_model_dir,)
            )
            self.logger.info("OCR Handler: Global OCR processing pool initialized.")
        except Exception as e_pool_create:
            self.logger.critical(f"OCR Handler: Failed to create global OCR processing pool: {e_pool_create}", exc_info=True)
            self.ocr_processing_pool = None
            self.num_workers = 1 # Fallback to serial
            self._initialize_serial_ocr_predictor() # 尝试初始化串行模式

    def _initialize_serial_ocr_predictor(self):
        """初始化用于串行处理的OCR预测器。"""
        self.logger.info("OCR Handler: Initializing OCR for serial processing...")
        if not os.path.exists(self.rec_model_dir):
            self.logger.error(f"OCR Handler: Serial OCR model directory not found: {self.rec_model_dir}. OCR will fail.")
            return
        try:
            # 对于串行模式，我们直接在主进程中创建和使用一个predictor实例
            # 而不是依赖全局的 _worker_ocr_predictor_instance (那是给worker用的)
            self.serial_ocr_predictor = pdx.inference.create_predictor(
                model_dir=self.rec_model_dir,
                model_name='PP-OCRv5_server_rec', # 与worker保持一致
                device='cpu'
            )
            self.logger.info("OCR Handler: Serial OCR predictor (for main process) initialized.")
        except Exception as e:
            self.logger.error(f"OCR Handler: Serial OCR predictor (for main process) initialization FAILED: {e}", exc_info=True)
            self.serial_ocr_predictor = None

    def prepare_ocr_tasks_from_detections(self,
                                          original_image_cv2: np.ndarray,
                                          yolo_detections: List[Dict[str, Any]],
                                          session_id_for_saving: str = None,
                                          frame_num_for_saving: int = 0,
                                          save_ocr_slices_debug: bool = False, # 改名以区分
                                          # process_photo_dir_base: str, # 从全局config获取
                                          ) -> Tuple[List[Tuple[int, np.ndarray, Tuple[int,int]]], List[Dict[str, Any]]]:
        ocr_tasks_for_pool = []
        ocr_input_metadata_list = [None] * len(yolo_detections) # 预分配列表以保持顺序
        log_prefix_roi = f"会话 {session_id_for_saving if session_id_for_saving else 'N/A'} (prepare_ocr_tasks F{frame_num_for_saving}):"

        # 训练用ROI的保存目录 (从全局config获取基础路径)
        # PROCESS_PHOTO_DIR 应该在全局 config.py 中定义
        # from config import PROCESS_PHOTO_DIR # 或者在类初始化时传入

        # 确保 PROCESS_PHOTO_DIR 在 self.config_params 中 (如果 LayoutStateManager 初始化时传入了)
        # 或者直接从全局 config 模块导入
        # 为了解耦，最好是在 OcrHandler 初始化时也传入 PROCESS_PHOTO_DIR
        # 暂时先假设能从全局 config 模块获取
        try:
            # 尝试从全局config导入，如果OcrHandler没有自己的config_params
            from config import PROCESS_PHOTO_DIR as GLOBAL_PROCESS_PHOTO_DIR
            process_photo_dir_base_for_saving = GLOBAL_PROCESS_PHOTO_DIR
        except ImportError:
            self.logger.warning(f"{log_prefix_roi} 未能从全局config导入PROCESS_PHOTO_DIR，将使用默认'process_photo'")
            process_photo_dir_base_for_saving = "process_photo"


        training_roi_base_dir = os.path.join(process_photo_dir_base_for_saving, "training_rois")
        if SAVE_TRAINING_ROI_IMAGES and session_id_for_saving:
            session_training_roi_dir = os.path.join(training_roi_base_dir, session_id_for_saving)
            if not os.path.exists(session_training_roi_dir):
                try:
                    os.makedirs(session_training_roi_dir, exist_ok=True)
                except OSError as e_mkdir:
                    self.logger.error(f"{log_prefix_roi} 创建训练ROI目录 {session_training_roi_dir} 失败: {e_mkdir}")
                    session_training_roi_dir = None # 标记无法保存
        else:
            session_training_roi_dir = None

        for i, det in enumerate(yolo_detections):
            original_yolo_idx = det['original_index']
            yolo_box_coords = det['box_yolo_xyxy']
            x1_y, y1_y, x2_y, y2_y = yolo_box_coords
            h_y, w_y = y2_y - y1_y, x2_y - x1_y

            y1_d_ideal = y1_y + int(h_y * self.digit_roi_y_offset_factor)
            h_d_ideal = int(h_y * self.digit_roi_height_factor)
            y2_d_ideal = y1_d_ideal + h_d_ideal
            w_d_exp = int(w_y * self.digit_roi_width_expand_factor)
            cx_y_img = x1_y + w_y / 2.0
            x1_d_ideal = int(cx_y_img - w_d_exp / 2.0)
            x2_d_ideal = int(cx_y_img + w_d_exp / 2.0)

            roi_for_ocr_raw = crop_roi_from_image(original_image_cv2, [x1_d_ideal, y1_d_ideal, x2_d_ideal, y2_d_ideal])
            original_roi_shape_hw_before_our_preprocess = roi_for_ocr_raw.shape[:2] if roi_for_ocr_raw is not None else (0,0)
            processed_roi_for_ocr = preprocess_roi_for_ocr(roi_for_ocr_raw, self.target_ocr_input_height)

            if processed_roi_for_ocr is not None:
                self.logger.debug(f"{log_prefix_roi} YOLO原始索引 {original_yolo_idx}: "
                                 f"原始数字ROI(H,W): {original_roi_shape_hw_before_our_preprocess}, "
                                 f"我们预处理后送OCR任务的ROI(H,W): {processed_roi_for_ocr.shape[:2]}")

                if SAVE_TRAINING_ROI_IMAGES and session_training_roi_dir:
                    try:
                        train_roi_filename = f"f{frame_num_for_saving}_yolo{original_yolo_idx}_h{processed_roi_for_ocr.shape[0]}_w{processed_roi_for_ocr.shape[1]}.png"
                        train_roi_path = os.path.join(session_training_roi_dir, train_roi_filename)
                        cv2.imwrite(train_roi_path, processed_roi_for_ocr)
                    except Exception as e_save_train_roi:
                        self.logger.error(f"{log_prefix_roi} 保存训练用ROI失败 {train_roi_path}: {e_save_train_roi}")
            else:
                self.logger.warning(f"{log_prefix_roi} YOLO原始索引 {original_yolo_idx}: 预处理后ROI为None。")

            # 确保即使 processed_roi_for_ocr 为 None，也填充 metadata_list 以保持长度一致
            ocr_input_metadata_list[i] = {
                "original_index": original_yolo_idx,
                "bbox_yolo_abs": yolo_box_coords,
                "yolo_anchor_details": {
                    'cx': det['cx'], 'cy': det['cy'], 'w': det['w'], 'h': det['h'], 'score': det['score']
                }}
            ocr_tasks_for_pool.append((original_yolo_idx, processed_roi_for_ocr, original_roi_shape_hw_before_our_preprocess))

            if save_ocr_slices_debug and processed_roi_for_ocr is not None and session_id_for_saving:
                try:
                    ocr_slice_debug_dir = os.path.join(process_photo_dir_base_for_saving, "ocr_slices_debug", session_id_for_saving)
                    if not os.path.exists(ocr_slice_debug_dir):
                        os.makedirs(ocr_slice_debug_dir, exist_ok=True)
                    slice_filename_debug = f"f{frame_num_for_saving}_s_idx{original_yolo_idx}_roi{i+1}.jpg"
                    slice_output_path_debug = os.path.join(ocr_slice_debug_dir, slice_filename_debug)
                    cv2.imwrite(slice_output_path_debug, processed_roi_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception as e_save_slice_debug:
                    self.logger.error(f"OCR Handler: 保存调试OCR切片图失败 {slice_output_path_debug}: {e_save_slice_debug}")

        return ocr_tasks_for_pool, ocr_input_metadata_list

    def recognize_prepared_tasks(self, ocr_tasks_for_pool: List[Tuple[int, np.ndarray, Tuple[int,int]]]) -> List[Tuple[int, Dict[str, Any]]]:
        if not ocr_tasks_for_pool:
            return []
        raw_ocr_pool_results = []
        if self.num_workers > 1 and self.ocr_processing_pool:
            self.logger.info(f"OCR Handler: 提交 {len(ocr_tasks_for_pool)} 个OCR任务到处理池...")
            try:
                raw_ocr_pool_results = self.ocr_processing_pool.map(ocr_task_for_worker_process, ocr_tasks_for_pool)
            except Exception as e_map: # 更具体的异常捕获
                self.logger.error(f"OCR Handler: OCR Pool map error: {e_map}", exc_info=True)
                # 可以考虑如果pool失败，是否尝试串行执行作为fallback
        else:
            self.logger.info(f"OCR Handler: 串行处理 {len(ocr_tasks_for_pool)} 个OCR任务...")
            if self.serial_ocr_predictor is None: # 检查类成员的串行predictor
                self.logger.error("OCR Handler: 串行OCR预测器 (类成员) 未初始化，无法处理任务。")
                return [(task_idx, {'rec_text': 'OCR_SERIAL_PREDICTOR_FAIL', 'rec_score': 0.0})
                        for task_idx, _, _ in ocr_tasks_for_pool]

            for task_idx, roi_img, orig_roi_shape in ocr_tasks_for_pool:
                # 串行时，直接使用 self.serial_ocr_predictor
                try:
                    if roi_img is None:
                         result_dict = {'rec_text': 'OCR_PREPROC_FAIL_IMG_NONE_SERIAL', 'rec_score': 0.0}
                    else:
                        rec_gen = self.serial_ocr_predictor.predict([roi_img])
                        rec_list = next(rec_gen, None)
                        result_dict = {'rec_text': '', 'rec_score': 0.0}
                        if rec_list and isinstance(rec_list, list) and len(rec_list) > 0:
                            result_dict = rec_list[0]
                        elif rec_list and isinstance(rec_list, dict):
                            result_dict = rec_list
                    raw_ocr_pool_results.append((task_idx, result_dict))
                except Exception as e_serial_pred:
                    self.logger.error(f"OCR Handler: 串行OCR预测失败 for yolo_idx {task_idx}: {e_serial_pred}")
                    raw_ocr_pool_results.append((task_idx, {'rec_text': 'OCR_PREDICT_FAIL_SERIAL', 'rec_score': 0.0}))

        return raw_ocr_pool_results

    def consolidate_ocr_results(self,
                                raw_ocr_pool_results: List[Tuple[int, Dict[str, Any]]],
                                ocr_input_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_ocr_results_list = [None] * len(ocr_input_metadata_list) # 与metadata长度一致
        map_idx_to_raw_ocr_result = {idx: res_dict for idx, res_dict in raw_ocr_pool_results}

        for i, metadata_item in enumerate(ocr_input_metadata_list):
            if metadata_item is None:
                # 这种情况理论上不应发生，如果prepare_ocr_tasks_from_detections的逻辑正确
                self.logger.error(f"OCR Handler: consolidate_ocr_results发现metadata_list中索引 {i} 为None！")
                final_ocr_results_list[i] = {"ocr_final_text": "N/A_METADATA_MISSING_AT_IDX_{i}", "ocr_confidence": 0.0}
                continue

            original_yolo_idx = metadata_item["original_index"]
            raw_ocr_dict = map_idx_to_raw_ocr_result.get(original_yolo_idx)

            full_ocr_item = {**metadata_item}
            if raw_ocr_dict and isinstance(raw_ocr_dict, dict):
                ocr_text_raw = raw_ocr_dict.get('rec_text', "")
                ocr_score_raw = raw_ocr_dict.get('rec_score', 0.0)

                if ocr_text_raw and not ocr_text_raw.startswith("OCR_"):
                    digits_only = "".join(re.findall(r'\d', ocr_text_raw))
                    full_ocr_item["ocr_final_text"] = digits_only if digits_only else "N/A_NO_DIGITS"
                else: # 保留错误码或空（如果无数字）
                    full_ocr_item["ocr_final_text"] = ocr_text_raw if ocr_text_raw else "N/A_EMPTY_RAW"
                full_ocr_item["ocr_confidence"] = float(ocr_score_raw)
            else:
                full_ocr_item["ocr_final_text"] = "N/A_OCR_RAW_RESULT_INVALID"
                full_ocr_item["ocr_confidence"] = 0.0

            final_ocr_results_list[i] = full_ocr_item # 替换None占位符

        return final_ocr_results_list

    def close_pool(self):
        if self.ocr_processing_pool:
            self.logger.info("OCR Handler: Closing OCR processing pool...")
            try:
                self.ocr_processing_pool.close()
                self.ocr_processing_pool.join() # 等待所有worker结束
                self.logger.info("OCR Handler: OCR processing pool closed.")
            except Exception as e:
                self.logger.error(f"OCR Handler: Error closing OCR pool: {e}", exc_info=True)
            self.ocr_processing_pool = None

        # 清理串行预测器（如果已创建）
        if self.serial_ocr_predictor:
            del self.serial_ocr_predictor # 显式删除
            self.serial_ocr_predictor = None
            self.logger.info("OCR Handler: Serial OCR predictor (main process) released.")

        # 对于worker中的全局实例 _worker_ocr_predictor_instance，
        # 它们会随worker进程的结束而自然销毁，通常不需要在主进程中显式处理。
        # 但如果 _initialize_serial_ocr_predictor 修改了全局的 _worker_ocr_predictor_instance，
        # 并且 num_workers <=1，那么在这里也清理一下可能是个好主意。
        global _worker_ocr_predictor_instance
        if self.num_workers <=1 and _worker_ocr_predictor_instance is not None:
             # self.logger.info("OCR Handler: Releasing global OCR predictor instance (used by serial or single worker mode).")
             # del _worker_ocr_predictor_instance # 直接del全局变量可能不安全或无效
             _worker_ocr_predictor_instance = None # 将其设为None，以便下次可以重新初始化

# END OF OcrHandler CLASS