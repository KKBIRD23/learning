# ocr_handler.py
import cv2
import numpy as np
import paddlex as pdx
import re
import multiprocessing
import os # For worker PID
import time # For OCR task duration
from typing import List, Dict, Tuple, Any

# 从 image_utils 导入必要的函数
from image_utils import crop_roi_from_image, preprocess_roi_for_ocr

# --- OCR Worker Initialization and Task ---
# 这部分逻辑与您原代码中的 worker 初始化和任务函数基本一致
_worker_ocr_predictor_instance = None # 每个worker进程持有一个实例

def init_ocr_worker_process(ocr_model_dir_path: str, model_name_cfg: str = 'PP-OCRv5_server_rec'):
    """OCR工作进程的初始化函数。"""
    global _worker_ocr_predictor_instance
    worker_pid = os.getpid()
    # print(f"[OCR Worker PID {worker_pid}] Initializing OCR predictor from: {ocr_model_dir_path}")
    # current_app.logger.info(...) # 在worker进程中直接用current_app.logger可能不行
    # 可以考虑传递一个简单的print或logging配置给worker
    try:
        _worker_ocr_predictor_instance = pdx.inference.create_predictor(
            model_dir=ocr_model_dir_path,
            model_name=model_name_cfg, # 可以考虑也作为配置传入
            device='cpu' # 假设CPU，可配置
        )
        print(f"[OCR Worker PID {worker_pid}] OCR predictor initialized successfully.")
    except Exception as e:
        print(f"[OCR Worker PID {worker_pid}] OCR predictor initialization FAILED: {e}")
        _worker_ocr_predictor_instance = None

def ocr_task_for_worker_process(task_data_tuple: Tuple[int, np.ndarray]) -> Tuple[int, Dict[str, Any]]:
    """在工作进程中执行单个OCR任务。"""
    global _worker_ocr_predictor_instance
    original_yolo_idx, roi_image_for_ocr = task_data_tuple
    pid = os.getpid()
    start_time = time.time()

    if roi_image_for_ocr is None:
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

        final_recognition_dict = {'rec_text': '', 'rec_score': 0.0}
        if recognition_result_list and isinstance(recognition_result_list, list) and len(recognition_result_list) > 0:
            # PaddleOCR PP-OCRv3/v4 server rec 模型通常返回 [{ 'rec_text': '...', 'rec_score': ...}]
            final_recognition_dict = recognition_result_list[0]
        elif recognition_result_list and isinstance(recognition_result_list, dict): # 兼容直接返回字典的情况
            final_recognition_dict = recognition_result_list

        return original_yolo_idx, {
            **final_recognition_dict,
            'pid': pid,
            'duration': time.time() - start_time
        }
    except Exception as e:
        # print(f"[OCR Worker PID {pid}] OCR prediction FAILED for yolo_idx {original_yolo_idx}: {e}")
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
        """
        初始化OCR处理器。
        Args:
            rec_model_dir (str): PaddleOCR识别模型目录。
            num_workers (int): 并行OCR工作进程数 (0或1表示串行)。
            target_ocr_input_height (int): OCR模型期望的输入图像高度。
            digit_roi_y_offset_factor (float): 数字ROI Y偏移因子。
            digit_roi_height_factor (float): 数字ROI 高度因子。
            digit_roi_width_expand_factor (float): 数字ROI 宽度扩展因子。
            logger: 日志记录器实例。
        """
        self.rec_model_dir = rec_model_dir
        self.num_workers = num_workers
        self.target_ocr_input_height = target_ocr_input_height
        self.digit_roi_y_offset_factor = digit_roi_y_offset_factor
        self.digit_roi_height_factor = digit_roi_height_factor
        self.digit_roi_width_expand_factor = digit_roi_width_expand_factor
        self.logger = logger

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
            self._initialize_serial_ocr_predictor()
            return

        self.logger.info(f"OCR Handler: Initializing global OCR processing pool with {self.num_workers} workers...")
        try:
            # 确保使用 'spawn' 启动方法，特别是在Windows上
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
            self.num_workers = 1 # Fallback to serial
            self._initialize_serial_ocr_predictor()

    def _initialize_serial_ocr_predictor(self):
        """初始化用于串行处理的OCR预测器。"""
        self.logger.info("OCR Handler: Initializing OCR for serial processing...")
        if not os.path.exists(self.rec_model_dir):
            self.logger.error(f"OCR Handler: Serial OCR model directory not found: {self.rec_model_dir}. OCR will fail.")
            return
        try:
            self.serial_ocr_predictor = pdx.inference.create_predictor(
                model_dir=self.rec_model_dir, model_name='PP-OCRv5_server_rec', device='cpu'
            )
            self.logger.info("OCR Handler: Serial OCR predictor initialized.")
        except Exception as e:
            self.logger.error(f"OCR Handler: Serial OCR predictor initialization FAILED: {e}", exc_info=True)
            self.serial_ocr_predictor = None

    def prepare_ocr_tasks_from_detections(self,
                                          original_image_cv2: np.ndarray,
                                          yolo_detections: List[Dict[str, Any]],
                                          session_id_for_saving: str = None, # 用于保存调试图片
                                          save_ocr_slices: bool = False,
                                          process_photo_dir_base: str = "process_photo"
                                          ) -> Tuple[List[Tuple[int, np.ndarray]], List[Dict[str, Any]]]:
        """
        从YOLO检测结果中提取、预处理数字ROI，并准备OCR任务。
        Args:
            original_image_cv2 (np.ndarray): 原始OpenCV图像 (BGR)。
            yolo_detections (List[Dict[str, Any]]): YOLO检测结果列表，每个元素包含
                'box_yolo_xyxy', 'score', 'original_index', 'cx', 'cy', 'w', 'h'.
            session_id_for_saving (str, optional): 用于保存调试切片的会话ID。
            save_ocr_slices (bool, optional): 是否保存OCR切片图。
            process_photo_dir_base (str, optional): 保存过程图片的根目录。
        Returns:
            Tuple[List[Tuple[int, np.ndarray]], List[Dict[str, Any]]]:
                - ocr_tasks_for_pool (list): [(original_yolo_idx, roi_image_for_ocr), ...]
                - ocr_input_metadata_list (list): 每个YOLO检测框对应的元数据字典列表。
        """
        ocr_tasks_for_pool = []
        ocr_input_metadata_list = [None] * len(yolo_detections)
        img_h_orig, img_w_orig = original_image_cv2.shape[:2]

        for i, det in enumerate(yolo_detections):
            original_yolo_idx = det['original_index'] # 这个索引是相对于 yolo_detections 列表的
            yolo_box_coords = det['box_yolo_xyxy']

            # 计算数字ROI的精确裁剪坐标
            x1_y, y1_y, x2_y, y2_y = yolo_box_coords
            h_y, w_y = y2_y - y1_y, x2_y - x1_y

            y1_d_ideal = y1_y + int(h_y * self.digit_roi_y_offset_factor)
            h_d_ideal = int(h_y * self.digit_roi_height_factor)
            y2_d_ideal = y1_d_ideal + h_d_ideal

            w_d_exp = int(w_y * self.digit_roi_width_expand_factor)
            cx_y_img = x1_y + w_y / 2.0
            x1_d_ideal = int(cx_y_img - w_d_exp / 2.0)
            x2_d_ideal = int(cx_y_img + w_d_exp / 2.0)

            # 裁剪并确保在图像边界内
            roi_for_ocr_raw = crop_roi_from_image(original_image_cv2, [x1_d_ideal, y1_d_ideal, x2_d_ideal, y2_d_ideal])

            # 对裁剪出的ROI进行预处理
            processed_roi_for_ocr = preprocess_roi_for_ocr(roi_for_ocr_raw, self.target_ocr_input_height)

            ocr_input_metadata_list[i] = {
                "original_index": original_yolo_idx, # 对应YOLO检测结果的索引
                "bbox_yolo_abs": yolo_box_coords,
                "yolo_anchor_details": { # 存储YOLO锚点的详细信息，供后续使用
                    'cx': det['cx'], 'cy': det['cy'],
                    'w': det['w'], 'h': det['h'],
                    'score': det['score']
                }
            }
            ocr_tasks_for_pool.append((original_yolo_idx, processed_roi_for_ocr))

            if save_ocr_slices and processed_roi_for_ocr is not None and session_id_for_saving:
                try:
                    ocr_slice_dir = os.path.join(process_photo_dir_base, "ocr_slices", session_id_for_saving)
                    if not os.path.exists(ocr_slice_dir):
                        os.makedirs(ocr_slice_dir, exist_ok=True)

                    # 使用原始图像名的一部分来命名，避免文件名冲突
                    # (假设调用者会传递原始图像名，或者我们在这里简化)
                    slice_filename = f"s_idx{original_yolo_idx}_roi{i+1}.jpg"
                    slice_output_path = os.path.join(ocr_slice_dir, slice_filename)
                    cv2.imwrite(slice_output_path, processed_roi_for_ocr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception as e_save_slice:
                    self.logger.error(f"OCR Handler: 保存OCR切片图失败 {slice_output_path}: {e_save_slice}")

        return ocr_tasks_for_pool, ocr_input_metadata_list

    def recognize_prepared_tasks(self, ocr_tasks_for_pool: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, Dict[str, Any]]]:
        """
        执行已准备好的OCR任务列表（并行或串行）。
        Returns:
            List[Tuple[int, Dict[str, Any]]]: [(original_yolo_idx, raw_ocr_result_dict), ...]
                                            raw_ocr_result_dict 包含 'rec_text', 'rec_score', 'pid', 'duration'
        """
        if not ocr_tasks_for_pool:
            return []

        raw_ocr_pool_results = []
        if self.num_workers > 1 and self.ocr_processing_pool:
            self.logger.info(f"OCR Handler: 提交 {len(ocr_tasks_for_pool)} 个OCR任务到处理池...")
            try:
                raw_ocr_pool_results = self.ocr_processing_pool.map(ocr_task_for_worker_process, ocr_tasks_for_pool)
            except Exception as e_map:
                self.logger.error(f"OCR Handler: OCR Pool map error: {e_map}", exc_info=True)
                # Fallback or error handling: e.g., try serial, or return empty/error markers
                # For now, if pool fails, results will be empty or partial.
        else: # Serial processing
            self.logger.info(f"OCR Handler: 串行处理 {len(ocr_tasks_for_pool)} 个OCR任务...")
            if self.serial_ocr_predictor is None:
                self.logger.error("OCR Handler: 串行OCR预测器未初始化，无法处理任务。")
                # 返回带有错误标记的结果
                return [(task_idx, {'rec_text': 'OCR_SERIAL_PREDICTOR_FAIL', 'rec_score': 0.0})
                        for task_idx, _ in ocr_tasks_for_pool]

            for task_idx, roi_img in ocr_tasks_for_pool:
                _, result_dict = ocr_task_for_worker_process((task_idx, roi_img)) # 调用worker函数（它会使用全局的serial_predictor）
                raw_ocr_pool_results.append((task_idx, result_dict))

        return raw_ocr_pool_results

    def consolidate_ocr_results(self,
                                raw_ocr_pool_results: List[Tuple[int, Dict[str, Any]]],
                                ocr_input_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        整合OCR原始结果与输入元数据，进行文本后处理。
        Args:
            raw_ocr_pool_results (list): 来自recognize_prepared_tasks的原始结果。
                                         [(original_yolo_idx, raw_ocr_result_dict), ...]
            ocr_input_metadata_list (list): 与YOLO检测框一一对应的元数据列表。
                                            其顺序应与最初提交给prepare_ocr_tasks的yolo_detections一致。
        Returns:
            List[Dict[str, Any]]: 最终的OCR结果列表，每个元素是一个字典，包含
                                  ocr_final_text, ocr_confidence, 以及来自元数据的原始YOLO信息。
                                  列表顺序与 ocr_input_metadata_list 一致。
        """
        final_ocr_results_list = [None] * len(ocr_input_metadata_list)

        # 创建一个从 original_yolo_idx 到原始OCR结果的映射，以便快速查找
        map_idx_to_raw_ocr_result = {idx: res_dict for idx, res_dict in raw_ocr_pool_results}

        for i, metadata_item in enumerate(ocr_input_metadata_list):
            if metadata_item is None: # Should not happen if prepare_ocr_tasks works correctly
                final_ocr_results_list[i] = {"ocr_final_text": "N/A_METADATA_MISSING", "ocr_confidence": 0.0}
                continue

            original_yolo_idx = metadata_item["original_index"]
            raw_ocr_dict = map_idx_to_raw_ocr_result.get(original_yolo_idx)

            full_ocr_item = {**metadata_item} # Start with all metadata (bbox_yolo, yolo_anchor_details, etc.)

            if raw_ocr_dict and isinstance(raw_ocr_dict, dict):
                ocr_text_raw = raw_ocr_dict.get('rec_text', "")
                ocr_score_raw = raw_ocr_dict.get('rec_score', 0.0)

                if ocr_text_raw and not ocr_text_raw.startswith("OCR_"): # Not an error code
                    digits_only = "".join(re.findall(r'\d', ocr_text_raw))
                    full_ocr_item["ocr_final_text"] = digits_only if digits_only else "N/A_NO_DIGITS"
                else:
                    full_ocr_item["ocr_final_text"] = ocr_text_raw # Preserve error code or empty if no digits
                full_ocr_item["ocr_confidence"] = float(ocr_score_raw)
                # full_ocr_item["ocr_worker_pid"] = raw_ocr_dict.get('pid') # Optional: for debugging
                # full_ocr_item["ocr_duration_ms"] = raw_ocr_dict.get('duration', 0) * 1000 # Optional
            else:
                full_ocr_item["ocr_final_text"] = "N/A_OCR_RAW_RESULT_INVALID"
                full_ocr_item["ocr_confidence"] = 0.0

            final_ocr_results_list[i] = full_ocr_item

        return final_ocr_results_list

    def close_pool(self):
        """关闭OCR处理池（如果存在）。"""
        if self.ocr_processing_pool:
            self.logger.info("OCR Handler: Closing OCR processing pool...")
            try:
                self.ocr_processing_pool.close()
                self.ocr_processing_pool.join()
                self.logger.info("OCR Handler: OCR processing pool closed.")
            except Exception as e:
                self.logger.error(f"OCR Handler: Error closing OCR pool: {e}", exc_info=True)
            self.ocr_processing_pool = None

        # 清理串行预测器（如果已创建）
        if self.serial_ocr_predictor:
            del self.serial_ocr_predictor
            self.serial_ocr_predictor = None
            self.logger.info("OCR Handler: Serial OCR predictor released.")