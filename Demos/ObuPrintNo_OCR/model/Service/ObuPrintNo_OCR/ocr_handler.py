# ocr_handler.py
import cv2
import numpy as np
import paddlex as pdx
import re
import multiprocessing
import os
import time
from typing import List, Dict, Tuple, Any

from image_utils import crop_roi_from_image, preprocess_roi_for_ocr # 假设preprocess_roi_for_ocr已移至image_utils

# --- OCR Worker Initialization and Task ---
_worker_ocr_predictor_instance = None

def init_ocr_worker_process(ocr_model_dir_path: str, model_name_cfg: str = 'PP-OCRv5_server_rec'):
    global _worker_ocr_predictor_instance
    worker_pid = os.getpid()
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
    # task_data_tuple 现在包含 (original_yolo_idx, roi_image_for_ocr, original_roi_shape_hw_before_our_preprocess)
    global _worker_ocr_predictor_instance
    original_yolo_idx, roi_image_for_ocr, original_roi_shape_hw = task_data_tuple
    pid = os.getpid()
    start_time = time.time()

    # --- 新增日志：打印送入OCR模型的图像尺寸 ---
    if roi_image_for_ocr is not None:
        print(f"[OCR Worker PID {pid}] DEBUG: original_yolo_idx={original_yolo_idx}, "
              f"Original ROI (H,W) before our preprocess: {original_roi_shape_hw}, "
              f"ROI送入PaddleX predictor前尺寸 (H,W): {roi_image_for_ocr.shape[:2]}")
    else:
        print(f"[OCR Worker PID {pid}] DEBUG: original_yolo_idx={original_yolo_idx}, ROI图像为None，无法送入模型。")
    # --- 结束新增 ---

    if roi_image_for_ocr is None:
        return original_yolo_idx, {'rec_text': 'OCR_PREPROC_FAIL_IMG_NONE', 'rec_score': 0.0,
                                   'pid': pid, 'duration': time.time() - start_time}

    if _worker_ocr_predictor_instance is None:
        return original_yolo_idx, {'rec_text': 'OCR_WORKER_INIT_FAIL', 'rec_score': 0.0,
                                   'pid': pid, 'duration': time.time() - start_time}

    try:
        recognition_results_generator = _worker_ocr_predictor_instance.predict([roi_image_for_ocr])
        recognition_result_list = next(recognition_results_generator, None)

        final_recognition_dict = {'rec_text': '', 'rec_score': 0.0}
        if recognition_result_list and isinstance(recognition_result_list, list) and len(recognition_result_list) > 0:
            final_recognition_dict = recognition_result_list[0]
        elif recognition_result_list and isinstance(recognition_result_list, dict):
            final_recognition_dict = recognition_result_list

        return original_yolo_idx, {
            **final_recognition_dict,
            'pid': pid,
            'duration': time.time() - start_time
        }
    except Exception as e:
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
                 logger: Any):
        self.rec_model_dir = rec_model_dir
        self.num_workers = num_workers
        self.target_ocr_input_height = target_ocr_input_height
        self.digit_roi_y_offset_factor = digit_roi_y_offset_factor
        self.digit_roi_height_factor = digit_roi_height_factor
        self.digit_roi_width_expand_factor = digit_roi_width_expand_factor
        self.logger = logger

        self.ocr_processing_pool = None
        self.serial_ocr_predictor = None

        if self.num_workers > 1:
            self._initialize_ocr_pool()
        else:
            self._initialize_serial_ocr_predictor()

    def _initialize_ocr_pool(self):
        if not os.path.exists(self.rec_model_dir):
            self.logger.critical(f"OCR Handler: Server OCR model directory for workers not found: {self.rec_model_dir}. OCR will fallback to serial or fail.")
            self.num_workers = 1
            self._initialize_serial_ocr_predictor()
            return
        self.logger.info(f"OCR Handler: Initializing global OCR processing pool with {self.num_workers} workers...")
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                    self.logger.info(f"OCR Handler: Multiprocessing start method set to 'spawn'.")
                except RuntimeError as e_mp_start:
                     self.logger.warning(f"OCR Handler: Could not set multiprocessing start method to 'spawn': {e_mp_start}. Using default: {multiprocessing.get_start_method(allow_none=True)}")
            self.ocr_processing_pool = multiprocessing.Pool(
                processes=self.num_workers,
                initializer=init_ocr_worker_process,
                initargs=(self.rec_model_dir,)
            )
            self.logger.info("OCR Handler: Global OCR processing pool initialized.")
        except Exception as e_pool_create:
            self.logger.critical(f"OCR Handler: Failed to create global OCR processing pool: {e_pool_create}", exc_info=True)
            self.ocr_processing_pool = None
            self.num_workers = 1
            self._initialize_serial_ocr_predictor()

    def _initialize_serial_ocr_predictor(self):
        self.logger.info("OCR Handler: Initializing OCR for serial processing...")
        if not os.path.exists(self.rec_model_dir):
            self.logger.error(f"OCR Handler: Serial OCR model directory not found: {self.rec_model_dir}. OCR will fail.")
            return
        try:
            # 在 init_ocr_worker_process 中已经有了全局的 _worker_ocr_predictor_instance
            # 串行时，我们直接调用 init_ocr_worker_process 来初始化这个全局实例
            init_ocr_worker_process(self.rec_model_dir)
            if _worker_ocr_predictor_instance is not None: # 检查是否成功初始化
                self.serial_ocr_predictor = _worker_ocr_predictor_instance # 赋予一个类成员引用，但实际还是用全局的
                self.logger.info("OCR Handler: Serial OCR predictor (global instance) initialized.")
            else:
                self.logger.error("OCR Handler: Serial OCR predictor (global instance) FAILED to initialize.")
                self.serial_ocr_predictor = None
        except Exception as e:
            self.logger.error(f"OCR Handler: Serial OCR predictor initialization FAILED: {e}", exc_info=True)
            self.serial_ocr_predictor = None

    def prepare_ocr_tasks_from_detections(self,
                                          original_image_cv2: np.ndarray,
                                          yolo_detections: List[Dict[str, Any]],
                                          session_id_for_saving: str = None,
                                          save_ocr_slices: bool = False,
                                          process_photo_dir_base: str = "process_photo"
                                          ) -> Tuple[List[Tuple[int, np.ndarray, Tuple[int,int]]], List[Dict[str, Any]]]:
        ocr_tasks_for_pool = []
        ocr_input_metadata_list = [None] * len(yolo_detections)
        img_h_orig, img_w_orig = original_image_cv2.shape[:2]
        log_prefix_roi = f"会话 {session_id_for_saving if session_id_for_saving else 'N/A'} (prepare_ocr_tasks):"

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

            # --- 新增日志：打印我们预处理后的ROI尺寸 ---
            if processed_roi_for_ocr is not None:
                self.logger.debug(f"{log_prefix_roi} YOLO原始索引 {original_yolo_idx}: "
                                 f"原始数字ROI(H,W): {original_roi_shape_hw_before_our_preprocess}, "
                                 f"送入OCR任务的ROI(H,W): {processed_roi_for_ocr.shape[:2]}")
            else:
                self.logger.warning(f"{log_prefix_roi} YOLO原始索引 {original_yolo_idx}: 预处理后ROI为None。")
            # --- 结束新增 ---

            ocr_input_metadata_list[i] = {
                "original_index": original_yolo_idx,
                "bbox_yolo_abs": yolo_box_coords,
                "yolo_anchor_details": {
                    'cx': det['cx'], 'cy': det['cy'],
                    'w': det['w'], 'h': det['h'],
                    'score': det['score']
                }
            }
            # 将原始ROI尺寸也传递给worker，用于日志打印
            ocr_tasks_for_pool.append((original_yolo_idx, processed_roi_for_ocr, original_roi_shape_hw_before_our_preprocess))


            if save_ocr_slices and processed_roi_for_ocr is not None and session_id_for_saving:
                try:
                    ocr_slice_dir = os.path.join(process_photo_dir_base, "ocr_slices", session_id_for_saving)
                    if not os.path.exists(ocr_slice_dir):
                        os.makedirs(ocr_slice_dir, exist_ok=True)
                    slice_filename = f"s_idx{original_yolo_idx}_roi{i+1}.jpg"
                    slice_output_path = os.path.join(ocr_slice_dir, slice_filename)
                    cv2.imwrite(slice_output_path, processed_roi_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception as e_save_slice:
                    self.logger.error(f"OCR Handler: 保存OCR切片图失败 {slice_output_path}: {e_save_slice}")

        return ocr_tasks_for_pool, ocr_input_metadata_list

    def recognize_prepared_tasks(self, ocr_tasks_for_pool: List[Tuple[int, np.ndarray, Tuple[int,int]]]) -> List[Tuple[int, Dict[str, Any]]]:
        if not ocr_tasks_for_pool:
            return []
        raw_ocr_pool_results = []
        if self.num_workers > 1 and self.ocr_processing_pool:
            self.logger.info(f"OCR Handler: 提交 {len(ocr_tasks_for_pool)} 个OCR任务到处理池...")
            try:
                raw_ocr_pool_results = self.ocr_processing_pool.map(ocr_task_for_worker_process, ocr_tasks_for_pool)
            except Exception as e_map:
                self.logger.error(f"OCR Handler: OCR Pool map error: {e_map}", exc_info=True)
        else:
            self.logger.info(f"OCR Handler: 串行处理 {len(ocr_tasks_for_pool)} 个OCR任务...")
            # 确保串行时 _worker_ocr_predictor_instance 已被 _initialize_serial_ocr_predictor 初始化
            if _worker_ocr_predictor_instance is None: # 检查全局实例
                self.logger.error("OCR Handler: 串行OCR预测器 (全局实例) 未初始化，无法处理任务。")
                return [(task_idx, {'rec_text': 'OCR_SERIAL_PREDICTOR_FAIL', 'rec_score': 0.0})
                        for task_idx, _, _ in ocr_tasks_for_pool] # Adjusted for new tuple format

            for task_idx, roi_img, orig_roi_shape in ocr_tasks_for_pool: # Adjusted for new tuple format
                _, result_dict = ocr_task_for_worker_process((task_idx, roi_img, orig_roi_shape))
                raw_ocr_pool_results.append((task_idx, result_dict))

        return raw_ocr_pool_results

    def consolidate_ocr_results(self,
                                raw_ocr_pool_results: List[Tuple[int, Dict[str, Any]]],
                                ocr_input_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_ocr_results_list = [None] * len(ocr_input_metadata_list)
        map_idx_to_raw_ocr_result = {idx: res_dict for idx, res_dict in raw_ocr_pool_results}

        for i, metadata_item in enumerate(ocr_input_metadata_list):
            if metadata_item is None:
                final_ocr_results_list[i] = {"ocr_final_text": "N/A_METADATA_MISSING", "ocr_confidence": 0.0}
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
                else:
                    full_ocr_item["ocr_final_text"] = ocr_text_raw
                full_ocr_item["ocr_confidence"] = float(ocr_score_raw)
            else:
                full_ocr_item["ocr_final_text"] = "N/A_OCR_RAW_RESULT_INVALID"
                full_ocr_item["ocr_confidence"] = 0.0
            final_ocr_results_list[i] = full_ocr_item
        return final_ocr_results_list

    def close_pool(self):
        if self.ocr_processing_pool:
            self.logger.info("OCR Handler: Closing OCR processing pool...")
            try:
                self.ocr_processing_pool.close()
                self.ocr_processing_pool.join()
                self.logger.info("OCR Handler: OCR processing pool closed.")
            except Exception as e:
                self.logger.error(f"OCR Handler: Error closing OCR pool: {e}", exc_info=True)
            self.ocr_processing_pool = None

        # 对于串行模式，_worker_ocr_predictor_instance 是全局的，
        # 其生命周期随worker进程或主进程（如果串行）结束而结束，通常不需要显式删除。
        # 如果需要更严格的清理，可能要在应用退出时处理。
        global _worker_ocr_predictor_instance
        if _worker_ocr_predictor_instance and self.num_workers <=1 : # 如果是串行模式下初始化的
             del _worker_ocr_predictor_instance
             _worker_ocr_predictor_instance = None
             self.logger.info("OCR Handler: Global OCR predictor instance (for serial) released.")
        self.serial_ocr_predictor = None # 清理类成员引用