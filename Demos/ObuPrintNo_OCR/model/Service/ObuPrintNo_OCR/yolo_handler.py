# yolo_handler.py
import cv2
import numpy as np
import onnxruntime
import time
from typing import List, Dict, Tuple, Any

# 从 image_utils 导入必要的函数
from image_utils import get_box_center_and_dims

class YoloHandler:
    def __init__(self, model_path: str,
                 conf_threshold: float,
                 iou_threshold: float,
                 min_area_px: int,
                 max_area_factor: float,
                 coco_classes: List[str],
                 logger: Any):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area_px = min_area_px
        self.max_area_factor = max_area_factor
        self.coco_classes = coco_classes # num_classes will be len(self.coco_classes)
        self.logger = logger

        self.onnx_session = None
        self.input_name = None
        self.input_shape = None
        self.model_input_height = 640
        self.model_input_width = 640

        self._load_model()

    def _load_model(self):
        self.logger.info(f"YOLO Handler: 正在从 {self.model_path} 加载ONNX模型...")
        try:
            # --- 恢复：移除所有SessionOptions，让ONNX在新环境下自由发挥 ---
            self.onnx_session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

            input_meta = self.onnx_session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = input_meta.shape

            # 动态获取模型输入尺寸，兼容 'height', 'width' 字符串或整数
            if len(self.input_shape) == 4:
                if isinstance(self.input_shape[2], str) or self.input_shape[2] is None: # 'height' or dynamic
                    self.logger.warning(f"YOLO Handler: 模型输入高度为动态值 ('{self.input_shape[2]}')，将使用默认值 {self.model_input_height}。")
                elif isinstance(self.input_shape[2], int):
                    self.model_input_height = self.input_shape[2]

                if isinstance(self.input_shape[3], str) or self.input_shape[3] is None: # 'width' or dynamic
                    self.logger.warning(f"YOLO Handler: 模型输入宽度为动态值 ('{self.input_shape[3]}')，将使用默认值 {self.model_input_width}。")
                elif isinstance(self.input_shape[3], int):
                    self.model_input_width = self.input_shape[3]

            self.logger.info(f"YOLO Handler: ONNX模型加载成功。输入名: {self.input_name}, "
                             f"原始输入形状: {self.input_shape}, "
                             f"实际使用模型输入尺寸 HxW: {self.model_input_height}x{self.model_input_width}")
        except Exception as e:
            self.logger.error(f"YOLO Handler: ONNX模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load YOLO ONNX model: {e}")

    def _preprocess_image(self, image_cv2: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        img_h_orig, img_w_orig = image_cv2.shape[:2]
        ratio = min(self.model_input_width / img_w_orig, self.model_input_height / img_h_orig)
        new_w = int(img_w_orig * ratio)
        new_h = int(img_h_orig * ratio)
        resized_img = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.model_input_height, self.model_input_width, 3), 128, dtype=np.uint8)
        pad_x = (self.model_input_width - new_w) // 2
        pad_y = (self.model_input_height - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_img
        tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(tensor, axis=0)
        return input_tensor, ratio, pad_x, pad_y

    def _non_max_suppression(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> List[int]:
        """执行非极大值抑制 (来自您提供的代码)。"""
        if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0:
            return []
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
        order = scores.argsort()[::-1]
        keep_indices = []
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            order = order[1:]
            if order.size == 0:
                break
            xx1 = np.maximum(x1[i], x1[order])
            yy1 = np.maximum(y1[i], y1[order])
            xx2 = np.minimum(x2[i], x2[order])
            yy2 = np.minimum(y2[i], y2[order])
            w_intersect = np.maximum(0.0, xx2 - xx1)
            h_intersect = np.maximum(0.0, yy2 - yy1)
            intersection_area = w_intersect * h_intersect
            iou = intersection_area / (areas[i] + areas[order] - intersection_area + 1e-6)
            order = order[np.where(iou <= self.iou_threshold)[0]] # 使用 self.iou_threshold
        return keep_indices

    def _postprocess_outputs(self, raw_outputs: List[np.ndarray],
                             original_shape_hw: Tuple[int, int],
                             resize_ratio: float,
                             pad_x: int, pad_y: int) -> List[Dict[str, Any]]:
        """
        对YOLOv8模型的原始输出进行后处理 (基于您提供的代码)。
        """
        raw_output_tensor = np.squeeze(raw_outputs[0])
        if raw_output_tensor.ndim != 2:
            self.logger.error(f"YOLO Handler: Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}")
            return []

        # 确定迭代方向 (num_predictions 行, attributes 列)
        predictions_to_iterate = raw_output_tensor.transpose() if raw_output_tensor.shape[0] < raw_output_tensor.shape[1] else raw_output_tensor

        candidate_boxes_xyxy_model_space = []
        candidate_scores = []
        candidate_class_ids = []

        num_classes = len(self.coco_classes)
        expected_attributes_with_obj = 4 + 1 + num_classes
        expected_attributes_simple_conf = 4 + num_classes # 修正：之前这里是4+1，对于单类别且无obj是5，对于多类别且无obj是4+N

        actual_attributes = predictions_to_iterate.shape[1]

        for i_pred, pred_data in enumerate(predictions_to_iterate):
            box_coords_raw = pred_data[:4] # cx, cy, w, h in model input space
            final_confidence = 0.0
            class_id = 0

            if actual_attributes == expected_attributes_with_obj:
                objectness = float(pred_data[4])
                class_scores_all = pred_data[5:]
                if num_classes == 1:
                    final_confidence = objectness * float(class_scores_all[0])
                else:
                    class_id = np.argmax(class_scores_all)
                    max_class_score = float(class_scores_all[class_id])
                    final_confidence = objectness * max_class_score
            elif actual_attributes == expected_attributes_simple_conf: # 直接是类别置信度
                class_scores_all = pred_data[4:]
                if num_classes == 1:
                    final_confidence = float(class_scores_all[0])
                else:
                    class_id = np.argmax(class_scores_all)
                    final_confidence = float(class_scores_all[class_id])
            else: # Fallback or unexpected format
                if i_pred == 0:
                    self.logger.warning(
                        f"YOLO Handler: Unexpected number of attributes per prediction ({actual_attributes}). "
                        f"Expected {expected_attributes_with_obj} (with obj_conf) or "
                        f"{expected_attributes_simple_conf} (direct class_conf). "
                        f"Model output shape: {raw_output_tensor.shape}. Iteration shape: {predictions_to_iterate.shape}"
                    )
                # 尝试按5属性单类别解析 (cx,cy,w,h,conf) - 这是您之前日志中遇到的情况
                if actual_attributes == 5 and num_classes == 1:
                    final_confidence = float(pred_data[4])
                elif actual_attributes >= 5 : # 如果属性多于5，尝试取第5个作为置信度（非常规）
                    final_confidence = float(pred_data[4])
                    self.logger.debug(f"  Fallback: Using attribute 4 as confidence for pred {i_pred}")
                else:
                    self.logger.error(f"YOLO Handler: Prediction data too short ({actual_attributes} attributes) to extract confidence.")
                    continue

            if final_confidence >= self.conf_threshold: # 使用 self.conf_threshold
                cx, cy, w, h = box_coords_raw
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                candidate_boxes_xyxy_model_space.append([x1, y1, x2, y2])
                candidate_scores.append(final_confidence)
                candidate_class_ids.append(class_id)

        if not candidate_boxes_xyxy_model_space:
            return []

        keep_indices = self._non_max_suppression(np.array(candidate_boxes_xyxy_model_space), np.array(candidate_scores))

        processed_detections = []
        orig_h, orig_w = original_shape_hw
        for k_idx in keep_indices:
            idx = int(k_idx)
            box_model_coords = candidate_boxes_xyxy_model_space[idx]
            score = candidate_scores[idx]
            class_id_val = candidate_class_ids[idx]

            box_no_pad_x1 = box_model_coords[0] - pad_x
            box_no_pad_y1 = box_model_coords[1] - pad_y
            box_no_pad_x2 = box_model_coords[2] - pad_x
            box_no_pad_y2 = box_model_coords[3] - pad_y

            if resize_ratio == 0: continue

            orig_x1 = box_no_pad_x1 / resize_ratio
            orig_y1 = box_no_pad_y1 / resize_ratio
            orig_x2 = box_no_pad_x2 / resize_ratio
            orig_y2 = box_no_pad_y2 / resize_ratio

            final_x1 = int(np.clip(orig_x1, 0, orig_w))
            final_y1 = int(np.clip(orig_y1, 0, orig_h))
            final_x2 = int(np.clip(orig_x2, 0, orig_w))
            final_y2 = int(np.clip(orig_y2, 0, orig_h))

            # Area filtering (已移到 detect 方法的末尾，对最终坐标进行)
            # 这里先不进行面积筛选，因为坐标还是模型空间的相对值
            # 面积筛选应该在坐标转换到原始图像空间后进行

            # 转换为期望的字典格式
            cx_orig, cy_orig, w_orig, h_orig = get_box_center_and_dims([final_x1, final_y1, final_x2, final_y2])
            if cx_orig is not None: # 确保转换成功
                processed_detections.append({
                    "box_yolo_xyxy": [final_x1, final_y1, final_x2, final_y2],
                    "score": float(score),
                    "class_id": int(class_id_val),
                    "cx": cx_orig, "cy": cy_orig, "w": w_orig, "h": h_orig
                })
        return processed_detections

    def detect(self, image_cv2: np.ndarray) -> List[Dict[str, Any]]:
        if self.onnx_session is None:
            self.logger.error("YOLO Handler: ONNX session not initialized. Cannot detect.")
            return []

        original_shape_hw = image_cv2.shape[:2]

        input_tensor, resize_ratio, pad_x, pad_y = self._preprocess_image(image_cv2)

        try:
            raw_outputs = self.onnx_session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            self.logger.error(f"YOLO Handler: ONNX inference failed: {e}", exc_info=True)
            return []

        # Postprocess to get detections in original image space
        detections_before_area_filter = self._postprocess_outputs(
            raw_outputs, original_shape_hw, resize_ratio, pad_x, pad_y
        )

        # Apply area filtering as a final step
        final_filtered_detections = []
        if detections_before_area_filter:
            orig_h_img, orig_w_img = original_shape_hw
            max_area_threshold_px = self.max_area_factor
            if self.max_area_factor <= 1.0: # Factor is a ratio
                max_area_threshold_px = (orig_h_img * orig_w_img) * self.max_area_factor

            for i_det, det in enumerate(detections_before_area_filter):
                # 'w' and 'h' are already in original image scale
                area = det['w'] * det['h']

                if area < self.min_area_px:
                    # self.logger.debug(f"YOLO Handler: Box area {area} < min_area {self.min_area_px}, filtered.")
                    continue
                if area > max_area_threshold_px:
                    # self.logger.debug(f"YOLO Handler: Box area {area} > max_area {max_area_threshold_px}, filtered.")
                    continue

                det['original_index'] = len(final_filtered_detections) # Assign index after filtering
                final_filtered_detections.append(det)

        self.logger.info(f"YOLO Handler: Detection complete. Found {len(final_filtered_detections)} objects after all filters.")
        return final_filtered_detections