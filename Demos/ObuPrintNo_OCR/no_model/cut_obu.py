import os
import cv2
import numpy as np
from datetime import datetime
import argparse
import time
import easyocr
from pyzbar import pyzbar

# 版本号
VERSION = "v2.9.21"

# 参数解析
parser = argparse.ArgumentParser(description=f"OBU检测工具（版本 {VERSION}）")
parser.add_argument("--img_path", default="3.jpg", help="输入图像路径")
parser.add_argument("--enable_ocr", action="store_true", help="是否启用OCR模块")
parser.add_argument("--first_ocr", action="store_true", default=True, help="是否启用首次OCR")
parser.add_argument("--second_ocr", action="store_true", default=True, help="是否启用二次OCR")
parser.add_argument("--upper_black_v", type=int, default=55, help="手动模式下的UPPER_BLACK_V")
parser.add_argument("--morph_size", type=int, default=9, help="手动模式下的MORPH_KERNEL_SIZE")
parser.add_argument("--min_area", type=int, default=500, help="手动模式下的MIN_AREA")
parser.add_argument("--min_w", type=int, default=70, help="手动模式下的MIN_W")
parser.add_argument("--min_h", type=int, default=30, help="手动模式下的MIN_H")
parser.add_argument("--min_valid_area", type=int, default=2000, help="最小有效面积")
parser.add_argument("--min_gap", type=int, default=20, help="最小间距")
parser.add_argument("--overlap_threshold", type=float, default=0.1, help="重叠检查阈值")
args = parser.parse_args()

# 固定参数
IMG_PATH = args.img_path
OUT_DIR = "cut_obu"
LOG_DIR = os.path.join(OUT_DIR, "log")
PADDING = 5
MAX_AREA = 100000
MIN_VALID_AREA = args.min_valid_area
MIN_GAP = args.min_gap
MIN_ASPECT_RATIO = 0.5

# 初始化EasyOCR
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# 预处理函数
def preprocess_image(image, upper_black_v, morph_size):
    start_time = time.time()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, upper_black_v]))
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    preprocess_time = time.time() - start_time
    cv2.imwrite(os.path.join(OUT_DIR, "mask.jpg"), mask)
    return mask, preprocess_time

# 轮廓检测与框定
def detect_and_bound_contours(image, mask, min_area, min_w, min_h):
    start_time = time.time()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_boxes = []
    cnt = 0
    if not contours:
        print("[DEBUG] 无轮廓检测到，检查掩码或参数")
        return cnt, image, drawn_boxes, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        if (w >= min_w and h >= min_h and min_area <= area <= MAX_AREA and
            area >= MIN_VALID_AREA and aspect_ratio >= MIN_ASPECT_RATIO):
            padding = 10
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
            current_box = (x1, y1, x2, y2)
            overlap = False
            for box in drawn_boxes:
                if overlap_check(current_box, box, args.overlap_threshold) or min_distance_check(current_box, box, min_w):
                    overlap = True
                    break
            if not overlap:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                drawn_boxes.append(current_box)
                cnt += 1
    detect_time = time.time() - start_time
    cv2.imwrite(os.path.join(OUT_DIR, "detected_boxes.jpg"), image)
    print(f"[DEBUG] 检测到 {len(drawn_boxes)} 个框，cnt={cnt}")
    return cnt, image, drawn_boxes, detect_time

# 最小间距检查
def min_distance_check(box1, box2, min_w):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    if abs(x1 - bx2) < MIN_GAP or abs(x2 - bx1) < MIN_GAP:
        if abs(y1 - by1) < min_w * 0.5:
            return True
    return False

# 重叠检查
def overlap_check(box1, box2, threshold):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    intersection = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
    union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union > threshold if union > 0 else False

# 透视矫正函数
def correct_perspective(roi_gray, aspect_ratio_range=(2, 10), area_range=(300, 5000), epsilon_factor=0.01):
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[DEBUG] 未找到轮廓，返回原图")
        return roi_gray
    # 过滤轮廓
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        if (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and
            area_range[0] < area < area_range[1]):
            valid_contours.append((contour, x, y, w, h))
    if not valid_contours:
        print("[DEBUG] 未找到符合条件的轮廓，返回原图")
        return roi_gray
    # 选择靠近下部的轮廓
    selected_contour, x, y, w, h = min(valid_contours, key=lambda x: x[1] + x[2] + x[3])
    epsilon = epsilon_factor * cv2.arcLength(selected_contour, True)
    approx = cv2.approxPolyDP(selected_contour, epsilon, True)
    if len(approx) != 4:
        print(f"[DEBUG] 轮廓近似不是四边形，点数={len(approx)}，返回原图")
        return roi_gray
    points = approx.reshape(4, 2)
    sorted_points = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    sorted_points[0] = points[np.argmin(s)]  # 左上
    sorted_points[2] = points[np.argmax(s)]  # 右下
    diff = np.diff(points, axis=1)
    sorted_points[1] = points[np.argmin(diff)]  # 右上
    sorted_points[3] = points[np.argmax(diff)]  # 左下
    # 动态目标矩形
    w_target = int(np.sqrt((sorted_points[1][0] - sorted_points[0][0])**2 + (sorted_points[1][1] - sorted_points[0][1])**2))
    h_target = int(w_target / 3)  # 假设宽高比3:1
    dst_points = np.float32([[0, 0], [w_target, 0], [w_target, h_target], [0, h_target]])
    M = cv2.getPerspectiveTransform(sorted_points, dst_points)
    corrected = cv2.warpPerspective(roi_gray, M, (w_target, h_target))
    print(f"[DEBUG] 矫正成功，宽高比={w/h:.2f}, 目标宽高={w_target}x{h_target}")
    return corrected

# 直接定位数字区域
def locate_digit_region(roi_enhanced):
    roi_height = roi_enhanced.shape[0]
    # 假设条形码位于下半部分
    barcode_y1 = int(roi_height * 0.6)
    barcode_y2 = roi_height
    _, thresh = cv2.threshold(roi_enhanced[barcode_y1:barcode_y2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        barcode_y1 += y
        barcode_y2 = barcode_y1 + h
        digit_y1 = max(0, barcode_y1 - 2 * h)
        digit_y2 = barcode_y1
        digit_roi = roi_enhanced[digit_y1:digit_y2, 0:roi_enhanced.shape[1]]
        return digit_roi
    print("[DEBUG] 直接定位数字区域失败")
    return None

# 定位条形码+数字区域
def locate_combined_region(roi_enhanced):
    roi_height = roi_enhanced.shape[0]
    # 假设条形码+数字区域位于下半部分
    barcode_y1 = int(roi_height * 0.5)
    barcode_y2 = roi_height
    _, thresh = cv2.threshold(roi_enhanced[barcode_y1:barcode_y2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        barcode_y1 += y
        barcode_y2 = barcode_y1 + h
        combined_y1 = max(0, barcode_y1 - 3 * h)
        combined_y2 = barcode_y2
        combined_roi = roi_enhanced[combined_y1:combined_y2, 0:roi_enhanced.shape[1]]
        return combined_roi
    print("[DEBUG] 定位条形码+数字区域失败")
    return None

# OCR和条形码定位
def process_ocr_and_barcode(image, boxes, log_file=None):
    start_time = time.time()
    results = []
    height, width = image.shape[:2]
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"[WARNING] 框 {idx+1} 坐标超出图像边界: ({x1},{y1},{x2},{y2})")
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 透视矫正
        roi_corrected = correct_perspective(roi_gray)
        cv2.imwrite(os.path.join(OUT_DIR, f"roi_corrected_{idx+1}.jpg"), roi_corrected)

        # 增强处理
        roi_denoised = cv2.fastNlMeansDenoising(roi_corrected, h=25)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(roi_denoised)
        cv2.imwrite(os.path.join(OUT_DIR, f"roi_cut_{idx+1}.jpg"), roi_enhanced)

        # 首次OCR
        ocr_text_1 = "未启用"
        ocr_time_1 = 0
        if args.first_ocr and roi_enhanced.size > 0:
            ocr_start_1 = time.time()
            ocr_result_1 = reader.readtext(roi_enhanced, detail=0, paragraph=False)
            ocr_text_1 = " ".join(ocr_result_1) if ocr_result_1 else "未识别"
            ocr_time_1 = time.time() - ocr_start_1

        # 直接定位数字区域
        digit_roi_direct = locate_digit_region(roi_enhanced)
        if digit_roi_direct is not None:
            cv2.imwrite(os.path.join(OUT_DIR, f"digit_roi_direct_{idx+1}.jpg"), digit_roi_direct)

        # 定位条形码+数字区域
        combined_roi = locate_combined_region(roi_enhanced)
        if combined_roi is not None:
            cv2.imwrite(os.path.join(OUT_DIR, f"combined_roi_{idx+1}.jpg"), combined_roi)

        # 条形码检测（仅记录，不用于二次切割）
        barcode_start = time.time()
        barcodes = pyzbar.decode(roi_enhanced)
        barcode_time = time.time() - barcode_start
        barcode_text = barcodes[0].data.decode("utf-8") if barcodes else "未识别"
        results.append({
            "box": box,
            "ocr_1": ocr_text_1,
            "barcode": barcode_text,
            "barcode_time": barcode_time,
            "ocr_time_1": ocr_time_1
        })
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"框 {idx+1}: 坐标=({x1},{y1},{x2},{y2}), OCR_1={ocr_text_1}, "
                        f"Barcode={barcode_text}, Barcode时间={barcode_time:.3f}s, OCR_1时间={ocr_time_1:.3f}s\n")
    ocr_total_time = time.time() - start_time
    return results, ocr_total_time

# 主流程
def process_image():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取 {IMG_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"tuning_log_{VERSION}_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"OBU检测工具日志（版本 {VERSION}） - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("========================================\n")

    total_start = time.time()
    mask, preprocess_time = preprocess_image(img, args.upper_black_v, args.morph_size)
    cnt, img, valid_boxes, detect_time = detect_and_bound_contours(img, mask, args.min_area, args.min_w, args.min_h)
    complete_time = 0
    if args.enable_ocr:
        results, ocr_time = process_ocr_and_barcode(img, valid_boxes, log_file=log_file)
        print("[INFO] OCR和条形码识别结果：")
        for res in results:
            print(f"框: 坐标={res['box']}, OCR_1={res['ocr_1']}, "
                  f"Barcode={res['barcode']}, Barcode时间={res['barcode_time']:.3f}s, "
                  f"OCR_1时间={res['ocr_time_1']:.3f}s")
    else:
        ocr_time = 0
    total_time = time.time() - total_start
    output_path = os.path.join(OUT_DIR, f"marked_upper{args.upper_black_v}_morph{args.morph_size}_"
                                       f"area{args.min_area}_w{args.min_w}_h{args.min_h}_cnt{cnt}.jpg")
    cv2.imwrite(output_path, img)
    with open(log_file, "a", encoding="utf-8") as f:
        cmd = f"--upper_black_v {args.upper_black_v} --morph_size {args.morph_size} --min_area {args.min_area} " \
              f"--min_w {args.min_w} --min_h {args.min_h} --min_valid_area {args.min_valid_area} " \
              f"--min_gap {args.min_gap} --overlap_threshold {args.overlap_threshold}    本组合cnt={cnt}"
        f.write(f"手动模式: {cmd}, 总时间={total_time:.2f}s, 预处理={preprocess_time:.2f}s, "
                f"检测={detect_time:.2f}s, 补全={complete_time:.2f}s, OCR={ocr_time:.2f}s\n")

    print(f"[INFO] 运行完成，日志: {log_file}, 图片: {output_path}")

if __name__ == "__main__":
    process_image()