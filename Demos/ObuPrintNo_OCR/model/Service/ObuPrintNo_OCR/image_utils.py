# image_utils.py (FINAL VERSION)
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

def read_image_cv2(image_path: str) -> np.ndarray:
    """读取图片并返回OpenCV图像对象。"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from path: {image_path}")
    return image

def get_box_center_and_dims(box_xyxy: list) -> tuple:
    """从[x1, y1, x2, y2]格式的bbox计算中心点(cx, cy)和宽高(w, h)。"""
    if box_xyxy is None or len(box_xyxy) != 4:
        return None, None, None, None
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return int(cx), int(cy), int(w), int(h)

def crop_roi_from_image(image: np.ndarray, bbox_xyxy: list) -> Optional[np.ndarray]:
    """从图像中裁剪ROI区域。"""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    h_img, w_img = image.shape[:2]
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(w_img, x2)
    y2_clip = min(h_img, y2)

    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        return None
    return image[y1_clip:y2_clip, x1_clip:x2_clip]

def preprocess_roi_for_ocr(roi_image: Optional[np.ndarray], target_height: int) -> Optional[np.ndarray]:
    """
    对裁剪出的ROI进行OCR预处理。
    根据模型配置文件(inference.yml)，模型需要一个固定尺寸的输入。
    """
    if roi_image is None:
        return None

    fixed_width = 320

    resized_roi = cv2.resize(roi_image, (fixed_width, target_height),
                             interpolation=cv2.INTER_LANCZOS4)

    return resized_roi

def draw_yolo_detections_on_image(image_to_draw_on: np.ndarray, yolo_detections: List[Dict[str, Any]], ocr_texts_map: Optional[Dict[int, str]] = None, coco_classes: Optional[List[str]] = None) -> np.ndarray:
    """
    在图像上绘制YOLO检测框和可选的OCR文本。(旧版，用于调试)
    """
    img_out = image_to_draw_on.copy()
    if coco_classes is None:
        coco_classes = ['Object']

    for det in yolo_detections:
        box = det.get('box_yolo_xyxy')
        score = det.get('score', 0.0)
        original_idx = det.get('original_index', -1)

        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_name = coco_classes[0]
        yolo_label_text = f"{label_name}: {score:.2f}"

        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_out, yolo_label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if ocr_texts_map and original_idx in ocr_texts_map:
            ocr_text_to_draw = ocr_texts_map[original_idx]
            if ocr_text_to_draw and ocr_text_to_draw not in ["N/A", "ERR", "N/A_NO_DIGITS"]:
                 cv2.putText(img_out, ocr_text_to_draw, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img_out

def draw_ocr_results_on_image(
    original_image: np.ndarray,
    yolo_detections: List[Dict[str, Any]],
    final_ocr_results: List[Dict[str, Any]],
    valid_obu_codes: set # 虽然不再直接用它校验，但保留参数以备未来使用
) -> np.ndarray:
    """
    在图像上绘制带有半透明高亮色块的最终识别结果。
    - 绿色: 成功识别并通过校验的OBU (包括被纠错的)。
    - 红色: YOLO检测到，但最终未被采纳的OBU。
    """
    img_overlay = original_image.copy()
    alpha = 0.3  # 透明度

    color_success = (0, 255, 0)  # 绿色
    color_fail = (0, 0, 255)      # 红色

    # 创建一个映射，方便通过YOLO的原始索引查找OCR结果
    ocr_results_map = {item['original_index']: item for item in final_ocr_results}

    for det in yolo_detections:
        box = det.get('box_yolo_xyxy')
        if not box:
            continue

        x1, y1, x2, y2 = map(int, box)

        ocr_result_item = ocr_results_map.get(det['original_index'])

        # ==================================================================
        # ===               【 buddy，最终的修正就在这里！ 】               ===
        # ==================================================================
        # 我们直接检查 'final_corrected_text' 这个键是否存在。
        # 这个键只在 app.py 中被成功采纳（无论是精确匹配还是纠错）时才会被添加。
        is_successful = ocr_result_item is not None and 'final_corrected_text' in ocr_result_item
        # ==================================================================

        # 选择颜色
        fill_color = color_success if is_successful else color_fail

        # 绘制半透明填充色块
        sub_img = img_overlay[y1:y2, x1:x2]
        color_rect = np.full(sub_img.shape, fill_color, dtype=np.uint8)
        res_colored = cv2.addWeighted(sub_img, 1 - alpha, color_rect, alpha, 0.0)
        img_overlay[y1:y2, x1:x2] = res_colored

        # (可选) 绘制边框
        cv2.rectangle(img_overlay, (x1, y1), (x2, y2), fill_color, 2)

    return img_overlay