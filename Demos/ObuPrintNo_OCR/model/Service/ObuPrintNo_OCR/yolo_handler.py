# yolo_handler.py
import cv2
import numpy as np
import onnxruntime
import time # 用于可能的性能分析
from typing import List, Dict, Tuple, Any

# 从 image_utils 导入必要的函数
from image_utils import get_box_center_and_dims

class YoloHandler:
    def __init__(self, model_path: str,
                 conf_threshold: float,
                 iou_threshold: float,
                 min_area_px: int,
                 max_area_factor: float, # 0.0 to 1.0, or >1 for absolute px
                 coco_classes: List[str],
                 logger: Any):
        """
        初始化YOLOv8模型处理器。
        Args:
            model_path (str): ONNX模型文件路径。
            conf_threshold (float): 置信度阈值。
            iou_threshold (float): NMS的IOU阈值。
            min_area_px (int): 最小检测框面积（像素）。
            max_area_factor (float): 最大检测框面积因子。如果 <=1.0, 则为图像面积比例；如果 >1, 则为绝对像素值。
            coco_classes (List[str]): 模型能识别的类别名称列表。
            logger: 日志记录器实例。
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area_px = min_area_px
        self.max_area_factor = max_area_factor
        self.coco_classes = coco_classes
        self.logger = logger

        self.onnx_session = None
        self.input_name = None
        self.input_shape = None # (batch, channels, height, width)
        self.model_input_height = 640 # Default, will be updated from model
        self.model_input_width = 640  # Default, will be updated from model

        self._load_model()

    def _load_model(self):
        """加载ONNX模型到推理会话。"""
        self.logger.info(f"YOLO Handler: 正在从 {self.model_path} 加载ONNX模型...")
        try:
            self.onnx_session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            # 获取模型输入信息
            input_meta = self.onnx_session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = input_meta.shape # e.g., [1, 3, 640, 640]

            if len(self.input_shape) == 4 and \
               isinstance(self.input_shape[2], int) and \
               isinstance(self.input_shape[3], int):
                self.model_input_height = self.input_shape[2]
                self.model_input_width = self.input_shape[3]

            self.logger.info(f"YOLO Handler: ONNX模型加载成功。输入名: {self.input_name}, "
                             f"输入形状: {self.input_shape}, "
                             f"模型输入尺寸 HxW: {self.model_input_height}x{self.model_input_width}")
        except Exception as e:
            self.logger.error(f"YOLO Handler: ONNX模型加载失败: {e}", exc_info=True)
            # 可以在这里抛出异常，或者让后续的detect方法处理session为None的情况
            raise RuntimeError(f"Failed to load YOLO ONNX model: {e}")

    def _preprocess_image(self, image_cv2: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        对输入图像进行YOLOv8所需的预处理。
        Returns:
            Tuple: (input_tensor, resize_ratio, pad_x, pad_y)
        """
        img_h_orig, img_w_orig = image_cv2.shape[:2]

        # 计算缩放比例和新的尺寸
        ratio = min(self.model_input_width / img_w_orig, self.model_input_height / img_h_orig)
        new_w = int(img_w_orig * ratio)
        new_h = int(img_h_orig * ratio)

        resized_img = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建一个画布并填充
        canvas = np.full((self.model_input_height, self.model_input_width, 3), 128, dtype=np.uint8)
        pad_x = (self.model_input_width - new_w) // 2
        pad_y = (self.model_input_height - new_h) // 2

        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_img

        # 转换为张量 (HWC to CHW, BGR to RGB if needed - YOLOv8 typically expects BGR, scaling)
        # YOLOv8 ONNX通常期望 BGR, 0-1范围, CHW 格式
        tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(tensor, axis=0) # Add batch dimension

        return input_tensor, ratio, pad_x, pad_y

    def _non_max_suppression(self, boxes_xyxy: np.ndarray, scores: np.ndarray) -> List[int]:
        """执行非极大值抑制。"""
        if not isinstance(boxes_xyxy, np.ndarray) or boxes_xyxy.size == 0:
            return []

        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6) # Add epsilon to avoid division by zero
        order = scores.argsort()[::-1] # Sort by score in descending order

        keep_indices = []
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            order = order[1:] # Remove the current best box

            if order.size == 0:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order])
            yy1 = np.maximum(y1[i], y1[order])
            xx2 = np.minimum(x2[i], x2[order])
            yy2 = np.minimum(y2[i], y2[order])

            w_intersect = np.maximum(0.0, xx2 - xx1)
            h_intersect = np.maximum(0.0, yy2 - yy1)
            intersection_area = w_intersect * h_intersect

            iou = intersection_area / (areas[i] + areas[order] - intersection_area + 1e-6)

            # Keep boxes with IoU less than or equal to the threshold
            order = order[np.where(iou <= self.iou_threshold)[0]]

        return keep_indices

    def _postprocess_outputs(self, raw_outputs: List[np.ndarray],
                             original_shape_hw: Tuple[int, int],
                             resize_ratio: float,
                             pad_x: int, pad_y: int) -> List[Dict[str, Any]]:
        """
        对YOLOv8模型的原始输出进行后处理。
        YOLOv8 ONNX通常输出是 [batch, num_predictions, 4_coords + 1_obj_conf + num_classes_conf]
        或者 [batch, 4_coords + 1_obj_conf + num_classes_conf, num_predictions]
        我们处理的是 squeeze(raw_outputs[0]) 后的结果。
        """
        # raw_outputs[0] is typically the main detection output tensor
        raw_output_tensor = np.squeeze(raw_outputs[0]) # Remove batch dimension if it's 1

        if raw_output_tensor.ndim != 2:
            self.logger.error(f"YOLO Handler: Squeezed ONNX output is not 2D. Shape: {raw_output_tensor.shape}")
            return []

        # YOLOv8 output format: (cx, cy, w, h, obj_conf, class1_conf, class2_conf, ...)
        # Or if transposed: (4+1+num_classes, num_predictions)
        # We expect num_predictions rows, (4+1+num_classes) columns after potential transpose

        # If the number of columns is (4 + 1 + num_classes), it's likely [num_preds, attributes]
        # If the number of rows is (4 + 1 + num_classes), it's likely [attributes, num_preds], so transpose.
        num_expected_attributes = 4 + 1 + len(self.coco_classes)

        if raw_output_tensor.shape[0] == num_expected_attributes and raw_output_tensor.shape[1] > num_expected_attributes:
            predictions_to_iterate = raw_output_tensor.transpose() # [num_preds, attributes]
        elif raw_output_tensor.shape[1] == num_expected_attributes:
            predictions_to_iterate = raw_output_tensor
        else:
            self.logger.error(f"YOLO Handler: Unexpected ONNX output shape {raw_output_tensor.shape}. "
                              f"Expected {num_expected_attributes} attributes.")
            return []

        candidate_boxes_xyxy = []
        candidate_scores = []
        candidate_class_ids = []

        for i_pred, pred_data in enumerate(predictions_to_iterate):
            box_coords_raw = pred_data[:4] # cx, cy, w, h
            objectness_confidence = float(pred_data[4])
            class_confidences = pred_data[5:]

            if len(class_confidences) != len(self.coco_classes):
                self.logger.warning(f"YOLO Handler: Prediction {i_pred} has {len(class_confidences)} class scores, "
                                    f"but model configured for {len(self.coco_classes)} classes. Skipping.")
                continue

            final_confidence = 0.0
            class_id = 0 # Default for single class or if determination fails

            if len(self.coco_classes) == 1:
                final_confidence = objectness_confidence * float(class_confidences[0])
                class_id = 0
            else: # Multi-class scenario
                class_id = np.argmax(class_confidences)
                max_class_score = float(class_confidences[class_id])
                final_confidence = objectness_confidence * max_class_score

            if final_confidence >= self.conf_threshold:
                cx, cy, w_box, h_box = box_coords_raw
                x1 = cx - w_box / 2
                y1 = cy - h_box / 2
                x2 = cx + w_box / 2
                y2 = cy + h_box / 2
                candidate_boxes_xyxy.append([x1, y1, x2, y2])
                candidate_scores.append(final_confidence)
                candidate_class_ids.append(class_id)

        if not candidate_boxes_xyxy:
            return []

        # Perform NMS
        keep_indices = self._non_max_suppression(np.array(candidate_boxes_xyxy), np.array(candidate_scores))

        final_detections = []
        orig_h, orig_w = original_shape_hw

        for k_idx in keep_indices:
            idx = int(k_idx) # NMS returns indices into the candidate lists
            box_model_coords = candidate_boxes_xyxy[idx] # xyxy in model input space
            score = candidate_scores[idx]
            class_id_val = candidate_class_ids[idx]

            # Convert coordinates back to original image space
            # 1. Remove padding
            box_no_pad_x1 = box_model_coords[0] - pad_x
            box_no_pad_y1 = box_model_coords[1] - pad_y
            box_no_pad_x2 = box_model_coords[2] - pad_x
            box_no_pad_y2 = box_model_coords[3] - pad_y

            # 2. Rescale
            if resize_ratio == 0: # Avoid division by zero
                self.logger.warning("YOLO Handler: Resize ratio is zero, cannot rescale box.")
                continue

            orig_x1 = box_no_pad_x1 / resize_ratio
            orig_y1 = box_no_pad_y1 / resize_ratio
            orig_x2 = box_no_pad_x2 / resize_ratio
            orig_y2 = box_no_pad_y2 / resize_ratio

            # 3. Clip to original image dimensions
            final_x1 = int(np.clip(orig_x1, 0, orig_w))
            final_y1 = int(np.clip(orig_y1, 0, orig_h))
            final_x2 = int(np.clip(orig_x2, 0, orig_w))
            final_y2 = int(np.clip(orig_y2, 0, orig_h))

            # Area filtering
            det_w = final_x2 - final_x1
            det_h = final_y2 - final_y1
            area = det_w * det_h

            max_area_threshold_px = self.max_area_factor
            if self.max_area_factor <= 1.0: # Factor is a ratio
                max_area_threshold_px = (orig_h * orig_w) * self.max_area_factor

            if area < self.min_area_px:
                # self.logger.debug(f"YOLO Handler: Box area {area} < min_area {self.min_area_px}, filtered.")
                continue
            if area > max_area_threshold_px:
                # self.logger.debug(f"YOLO Handler: Box area {area} > max_area {max_area_threshold_px}, filtered.")
                continue

            cx_orig, cy_orig, w_orig, h_orig = get_box_center_and_dims([final_x1, final_y1, final_x2, final_y2])

            final_detections.append({
                "box_yolo_xyxy": [final_x1, final_y1, final_x2, final_y2],
                "score": float(score),
                "class_id": int(class_id_val),
                "cx": cx_orig,
                "cy": cy_orig,
                "w": w_orig,
                "h": h_orig
                # 'original_index' will be added by the caller based on the order in this list
            })

        return final_detections

    def detect(self, image_cv2: np.ndarray) -> List[Dict[str, Any]]:
        """
        对输入的OpenCV图像执行YOLOv8检测。
        Args:
            image_cv2 (np.ndarray): BGR格式的OpenCV图像。
        Returns:
            List[Dict[str, Any]]: 检测结果列表。每个字典包含:
                'box_yolo_xyxy': [x1, y1, x2, y2] in original image coords
                'score': float, detection confidence
                'class_id': int, class index
                'cx', 'cy', 'w', 'h': center coords and dimensions in original image
                'original_index': int, index of this detection in the returned list
        """
        if self.onnx_session is None:
            self.logger.error("YOLO Handler: ONNX session not initialized. Cannot detect.")
            return []

        original_shape_hw = image_cv2.shape[:2]

        # 1. Preprocess
        input_tensor, resize_ratio, pad_x, pad_y = self._preprocess_image(image_cv2)

        # 2. Inference
        try:
            raw_outputs = self.onnx_session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            self.logger.error(f"YOLO Handler: ONNX inference failed: {e}", exc_info=True)
            return []

        # 3. Postprocess
        detections = self._postprocess_outputs(raw_outputs, original_shape_hw, resize_ratio, pad_x, pad_y)

        # Add original_index to each detection dict
        for i, det in enumerate(detections):
            det['original_index'] = i

        self.logger.info(f"YOLO Handler: Detection complete. Found {len(detections)} objects after all filters.")
        return detections