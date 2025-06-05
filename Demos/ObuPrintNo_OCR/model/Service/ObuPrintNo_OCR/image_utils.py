# image_utils.py
import cv2
import numpy as np

def read_image_cv2(image_path: str):
    """读取图片并返回OpenCV图像对象。"""
    image = cv2.imread(image_path)
    if image is None:
        # 可以考虑在这里记录日志或抛出更具体的异常
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

def crop_roi_from_image(image, bbox_xyxy: list):
    """从图像中裁剪ROI区域。"""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    # 确保裁剪坐标在图像范围内
    h_img, w_img = image.shape[:2]
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(w_img, x2)
    y2_clip = min(h_img, y2)

    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        return None # 无效的裁剪区域
    return image[y1_clip:y2_clip, x1_clip:x2_clip]

def preprocess_roi_for_ocr(roi_image, target_height: int):
    """对裁剪出的ROI进行OCR预处理（缩放、灰度、二值化）。"""
    if roi_image is None:
        return None

    h_roi, w_roi = roi_image.shape[:2]
    if h_roi == 0 or w_roi == 0:
        return None

    scale = target_height / h_roi
    target_w = int(w_roi * scale)
    if target_w <= 0:
        target_w = 1 # 避免宽度为0

    resized_roi = cv2.resize(roi_image, (target_w, target_height),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

    gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # PaddleOCR通常需要BGR格式输入
    return cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)


def draw_yolo_detections_on_image(image_to_draw_on, yolo_detections, ocr_texts_map=None, coco_classes=None):
    """
    在图像上绘制YOLO检测框和可选的OCR文本。
    Args:
        image_to_draw_on: OpenCV图像。
        yolo_detections (list): YOLO检测结果列表，每个元素是包含 'box_yolo_xyxy', 'score', 'original_index' 的字典。
        ocr_texts_map (dict, optional): {original_yolo_index: "ocr_text"} 的字典。
        coco_classes (list, optional): 类别名称列表。
    Returns:
        OpenCV图像 (绘制后的副本)。
    """
    img_out = image_to_draw_on.copy()
    if coco_classes is None:
        coco_classes = ['Object'] # 默认类别名

    for det in yolo_detections:
        box = det.get('box_yolo_xyxy')
        score = det.get('score', 0.0)
        # class_id = det.get('class_id', 0) # 如果有类别ID
        original_idx = det.get('original_index', -1)

        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)

        # label_name = coco_classes[class_id] if 0 <= class_id < len(coco_classes) else f"ID:{class_id}"
        label_name = coco_classes[0] # 假设单类别
        yolo_label_text = f"{label_name}: {score:.2f}"

        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_out, yolo_label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if ocr_texts_map and original_idx in ocr_texts_map:
            ocr_text_to_draw = ocr_texts_map[original_idx]
            if ocr_text_to_draw and ocr_text_to_draw not in ["N/A", "ERR", "N/A_NO_DIGITS"]: # 只绘制有效文本
                 cv2.putText(img_out, ocr_text_to_draw, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 可以选择性绘制 original_idx
        # cv2.putText(img_out, f"idx:{original_idx}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return img_out