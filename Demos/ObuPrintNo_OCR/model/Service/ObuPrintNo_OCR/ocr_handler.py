# ocr_handler.py (FINAL BATTLE VERSION)
import cv2
import numpy as np
import onnxruntime
import re
import multiprocessing
import os
import time
from typing import List, Dict, Tuple, Any

from image_utils import crop_roi_from_image, preprocess_roi_for_ocr
from config import SAVE_TRAINING_ROI_IMAGES, PROCESS_PHOTO_DIR

# --- OCR Worker Globals ---
_worker_onnx_session = None
_worker_char_dict = None

def init_ocr_worker_process(onnx_model_path: str, keys_path: str):
    """OCR工作进程的初始化函数 (ONNX版)。"""
    global _worker_onnx_session, _worker_char_dict
    worker_pid = os.getpid()
    print(f"[OCR Worker PID {worker_pid}] Initializing ONNX session from: {onnx_model_path}")
    try:
        _worker_onnx_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        # --- 核心修正点 1: 构建正确的字典 ---
        # PaddleOCR的CTC模型约定，索引0为'blank'，我们必须手动在最前面添加它。
        char_list = []
        char_list.append("[blank]") # 手动添加 blank token 到索引0
        with open(keys_path, 'r', encoding='utf-8') as f:
            for line in f:
                char_list.append(line.strip())

        _worker_char_dict = char_list
        print(f"[OCR Worker PID {worker_pid}] ONNX session and character dict (size={len(_worker_char_dict)}) initialized. Blank token added.")
    except Exception as e:
        print(f"[OCR Worker PID {worker_pid}] ONNX session initialization FAILED: {e}")
        _worker_onnx_session = None
        _worker_char_dict = None

def _ctc_greedy_decoder(preds: np.ndarray, char_dict: List[str]) -> str:
    """简单的CTC贪心解码器 (V8.1 健壮版)。"""
    preds_indices = np.argmax(preds, axis=2)
    result = []
    last_index = 0 # 初始化为blank的索引
    char_dict_len = len(char_dict)

    for i in range(preds_indices.shape[1]):
        idx = preds_indices[0][i]

        # 【核心修正】增加安全检查，防止模型在处理异常输入时返回越界索引
        if idx >= char_dict_len:
            # 如果索引超出字典范围，直接忽略这个预测，保证程序不崩溃
            continue

        # 如果当前索引不是 blank (索引0)，并且不与上一个字符重复，则添加
        if idx != 0 and idx != last_index:
            result.append(char_dict[idx])
        last_index = idx

    return "".join(result)

def ocr_task_for_worker_process(task_data_tuple: Tuple[int, np.ndarray, Tuple[int, int]]) -> Tuple[int, Dict[str, Any]]:
    """在工作进程中执行单个OCR任务 (ONNX版)。"""
    global _worker_onnx_session, _worker_char_dict
    original_yolo_idx, roi_image_for_ocr, _ = task_data_tuple
    pid = os.getpid()
    start_time = time.time()

    if roi_image_for_ocr is None:
        return original_yolo_idx, {'rec_text': 'OCR_PREPROC_FAIL_IMG_NONE', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_time}
    if _worker_onnx_session is None or _worker_char_dict is None:
        return original_yolo_idx, {'rec_text': 'OCR_WORKER_INIT_FAIL', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_time}

    try:
        normalized_image = (roi_image_for_ocr.astype(np.float32) / 255.0 - 0.5) / 0.5
        input_tensor = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, :]

        input_name = _worker_onnx_session.get_inputs()[0].name
        onnx_outputs = _worker_onnx_session.run(None, {input_name: input_tensor})
        preds = onnx_outputs[0]

        rec_text = _ctc_greedy_decoder(preds, _worker_char_dict)
        rec_score = 0.95

        if rec_text:
            print(f"[OCR Worker PID {pid}] Decoded raw text for yolo_idx={original_yolo_idx}: '{rec_text}'")

        return original_yolo_idx, {
            'rec_text': rec_text, 'rec_score': rec_score,
            'pid': pid, 'duration': time.time() - start_time
        }
    except Exception as e:
        print(f"[OCR Worker PID {pid}] ONNX OCR prediction FAILED for yolo_idx {original_yolo_idx}: {e}")
        return original_yolo_idx, {'rec_text': 'OCR_PREDICT_FAIL_IN_WORKER', 'rec_score': 0.0, 'pid': pid, 'duration': time.time() - start_time, 'error_msg': str(e)}

# --- OcrHandler 类的其他部分保持不变 ---
class OcrHandler:
    def __init__(self,
                 onnx_model_path: str,
                 keys_path: str,
                 num_workers: int,
                 target_ocr_input_height: int,
                 digit_roi_y_offset_factor: float,
                 digit_roi_height_factor: float,
                 digit_roi_width_expand_factor: float,
                 logger: Any):
        self.onnx_model_path = onnx_model_path
        self.keys_path = keys_path
        self.num_workers = num_workers
        self.target_ocr_input_height = target_ocr_input_height
        self.digit_roi_y_offset_factor = digit_roi_y_offset_factor
        self.digit_roi_height_factor = digit_roi_height_factor
        self.digit_roi_width_expand_factor = digit_roi_width_expand_factor
        self.logger = logger

        self.ocr_processing_pool = None
        self.serial_onnx_session = None
        self.serial_char_dict = None

        if self.num_workers > 1:
            self._initialize_ocr_pool()
        else:
            self._initialize_serial_onnx_session()

    def _initialize_ocr_pool(self):
        self.logger.info(f"OCR Handler: Initializing ONNX OCR processing pool with {self.num_workers} workers...")
        try:
            if os.name == 'nt':
                multiprocessing.set_start_method('spawn', force=True)

            self.ocr_processing_pool = multiprocessing.Pool(
                processes=self.num_workers,
                initializer=init_ocr_worker_process,
                initargs=(self.onnx_model_path, self.keys_path)
            )
            self.logger.info("OCR Handler: ONNX OCR processing pool initialized.")
        except Exception as e:
            self.logger.critical(f"OCR Handler: Failed to create ONNX OCR pool: {e}", exc_info=True)
            self.ocr_processing_pool = None; self.num_workers = 1
            self._initialize_serial_onnx_session()

    def _initialize_serial_onnx_session(self):
        self.logger.info("OCR Handler: Initializing ONNX OCR for serial processing...")
        try:
            self.serial_onnx_session = onnxruntime.InferenceSession(self.onnx_model_path, providers=['CPUExecutionProvider'])

            char_list = []
            char_list.append("[blank]")
            with open(self.keys_path, 'r', encoding='utf-8') as f:
                for line in f:
                    char_list.append(line.strip())
            self.serial_char_dict = char_list

            self.logger.info("OCR Handler: Serial ONNX session and char dict initialized.")
        except Exception as e:
            self.logger.error(f"OCR Handler: Serial ONNX session initialization FAILED: {e}", exc_info=True)
            self.serial_onnx_session = None

    def prepare_ocr_tasks_from_detections(self,
                                          original_image_cv2: np.ndarray,
                                          yolo_detections: List[Dict[str, Any]],
                                          session_id: str,
                                          current_frame_num: int,
                                          save_process_photos: bool
                                          ) -> Tuple[List[Tuple[int, np.ndarray, Tuple[int,int]]], List[Dict[str, Any]]]:
        ocr_tasks_for_pool = []
        ocr_input_metadata_list = [None] * len(yolo_detections)

        training_roi_base_dir = os.path.join(PROCESS_PHOTO_DIR, "training_rois")
        if SAVE_TRAINING_ROI_IMAGES and session_id:
            session_training_roi_dir = os.path.join(training_roi_base_dir, session_id)
            if not os.path.exists(session_training_roi_dir):
                os.makedirs(session_training_roi_dir, exist_ok=True)
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

            if processed_roi_for_ocr is not None and SAVE_TRAINING_ROI_IMAGES and session_training_roi_dir:
                train_roi_filename = f"f{current_frame_num}_yolo{original_yolo_idx}_h{processed_roi_for_ocr.shape[0]}_w{processed_roi_for_ocr.shape[1]}.png"
                cv2.imwrite(os.path.join(session_training_roi_dir, train_roi_filename), processed_roi_for_ocr)

            ocr_input_metadata_list[i] = {
                "original_index": original_yolo_idx,
                "bbox_yolo_abs": yolo_box_coords,
                "yolo_anchor_details": {'cx': det['cx'], 'cy': det['cy'], 'w': det['w'], 'h': det['h'], 'score': det['score']}
            }
            ocr_tasks_for_pool.append((original_yolo_idx, processed_roi_for_ocr, original_roi_shape_hw_before_our_preprocess))
        return ocr_tasks_for_pool, ocr_input_metadata_list

    def recognize_prepared_tasks(self, ocr_tasks_for_pool: List[Tuple[int, np.ndarray, Tuple[int,int]]]) -> List[Tuple[int, Dict[str, Any]]]:
        if not ocr_tasks_for_pool: return []

        if self.num_workers > 1 and self.ocr_processing_pool:
            self.logger.info(f"OCR Handler: Submitting {len(ocr_tasks_for_pool)} ONNX OCR tasks to pool...")
            return self.ocr_processing_pool.map(ocr_task_for_worker_process, ocr_tasks_for_pool)
        else:
            self.logger.info(f"OCR Handler: Serially processing {len(ocr_tasks_for_pool)} ONNX OCR tasks...")
            if self.serial_onnx_session is None:
                return [(task_idx, {'rec_text': 'OCR_SERIAL_SESSION_FAIL', 'rec_score': 0.0}) for task_idx, _, _ in ocr_tasks_for_pool]

            results = []
            for task_idx, roi_img, orig_roi_shape in ocr_tasks_for_pool:
                if roi_img is None:
                    results.append((task_idx, {'rec_text': 'OCR_PREPROC_FAIL_IMG_NONE', 'rec_score': 0.0}))
                    continue
                try:
                    normalized_image = (roi_img.astype(np.float32) / 255.0 - 0.5) / 0.5
                    input_tensor = np.transpose(normalized_image, (2, 0, 1))[np.newaxis, :]

                    input_name = self.serial_onnx_session.get_inputs()[0].name
                    onnx_outputs = self.serial_onnx_session.run(None, {input_name: input_tensor})
                    preds = onnx_outputs[0]
                    rec_text = _ctc_greedy_decoder(preds, self.serial_char_dict)
                    results.append((task_idx, {'rec_text': rec_text, 'rec_score': 0.95}))
                except Exception as e:
                    self.logger.error(f"OCR Handler: Serial ONNX prediction failed for yolo_idx {task_idx}: {e}")
                    results.append((task_idx, {'rec_text': 'OCR_PREDICT_FAIL_SERIAL', 'rec_score': 0.0}))
            return results

    def consolidate_ocr_results(self,
                                raw_ocr_pool_results: List[Tuple[int, Dict[str, Any]]],
                                ocr_input_metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_ocr_results_list = [None] * len(ocr_input_metadata_list)
        map_idx_to_raw_ocr_result = {idx: res_dict for idx, res_dict in raw_ocr_pool_results}

        for i, metadata_item in enumerate(ocr_input_metadata_list):
            if metadata_item is None: continue
            original_yolo_idx = metadata_item["original_index"]
            raw_ocr_dict = map_idx_to_raw_ocr_result.get(original_yolo_idx)
            full_ocr_item = {**metadata_item}
            if raw_ocr_dict and isinstance(raw_ocr_dict, dict):
                ocr_text_raw = raw_ocr_dict.get('rec_text', "")
                if ocr_text_raw and not ocr_text_raw.startswith("OCR_"):
                    digits_only = "".join(re.findall(r'\d', ocr_text_raw))
                    full_ocr_item["ocr_final_text"] = digits_only if digits_only else "N/A_NO_DIGITS"
                else:
                    full_ocr_item["ocr_final_text"] = ocr_text_raw if ocr_text_raw else "N/A_EMPTY_RAW"
                full_ocr_item["ocr_confidence"] = float(raw_ocr_dict.get('rec_score', 0.0))
            else:
                full_ocr_item["ocr_final_text"] = "N/A_OCR_RAW_RESULT_INVALID"
                full_ocr_item["ocr_confidence"] = 0.0
            final_ocr_results_list[i] = full_ocr_item
        return final_ocr_results_list

    def close_pool(self):
        if self.ocr_processing_pool:
            self.logger.info("OCR Handler: Closing ONNX OCR processing pool...")
            self.ocr_processing_pool.close()
            self.ocr_processing_pool.join()
            self.logger.info("OCR Handler: ONNX OCR processing pool closed.")
        self.serial_onnx_session = None
        self.serial_char_dict = None