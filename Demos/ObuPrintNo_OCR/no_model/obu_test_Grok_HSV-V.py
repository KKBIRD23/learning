import os
import cv2
import numpy as np
from datetime import datetime
import argparse
import itertools
import sys
import easyocr
from pyzbar import pyzbar

# 版本号
VERSION = "v2.0"

# 参数解析
parser = argparse.ArgumentParser(description=f"OBU检测工具（版本 {VERSION}）")
parser.add_argument("--img_path", default="3.jpg", help="输入图像路径")
parser.add_argument("--auto_tuning", action="store_true", help="是否启用自动调优模式")
parser.add_argument("--enable_ocr", action="store_true", help="是否启用OCR模块")
parser.add_argument("--min_area", type=int, default=200, help="手动模式下的MIN_AREA")
parser.add_argument("--upper_black_v", type=int, default=50, help="手动模式下的UPPER_BLACK_V")
parser.add_argument("--morph_size", type=int, default=7, help="手动模式下的MORPH_KERNEL_SIZE")
args = parser.parse_args()

# 固定参数
IMG_PATH = args.img_path
OUT_DIR = "cut_obu"
LOG_DIR = os.path.join(OUT_DIR, "log")
SELECTED_DIR = os.path.join(OUT_DIR, "selected")
TARGET_ROWS, TARGET_COLS = 4, 6  # 4×6网格
ROW_THRESHOLD = 30
PADDING = 5

# 调优参数范围
UPPER_BLACK_V_RANGE = list(range(20, 71, 5))  # 20-70, 步长5
MORPH_KERNEL_SIZE_RANGE = list(range(3, 10, 2))  # (3, 3)到(9, 9), 步长2
MIN_AREA_RANGE = list(range(200, 1001, 100))  # 200-1000, 步长100
MIN_CONTOUR_SIZE_RANGE = list(itertools.product(range(50, 151, 20), range(30, 91, 10)))  # (50, 30)到(150, 90)

# OCR参数范围（可选）
OCR_DENOISE_H_RANGE = list(range(10, 31, 5))  # 10-30, 步长5
OCR_CLIP_LIMIT_RANGE = [round(x, 1) for x in np.arange(1.0, 3.1, 0.5)]  # 1.0-3.0, 步长0.5
OCR_PSM_RANGE = [6, 7, 8, 10, 11]  # Tesseract PSM选项

# 初始化EasyOCR
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# 预处理函数
def preprocess_image(image, upper_black_v, morph_size):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, upper_black_v]))
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# 轮廓检测与框定
def detect_and_bound_contours(image, mask, min_area, min_w, min_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_boxes = []
    cnt = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_w and h >= min_h and cv2.contourArea(contour) >= min_area:
            current_box = (x, y, x + w, y + h)
            if not any(overlap_check(current_box, box) for box in drawn_boxes):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                drawn_boxes.append(current_box)
                cnt += 1
    return cnt, image

# 网格分组与补全
def group_and_complete(image, boxes, row_threshold):
    if not boxes:
        return 0, image
    boxes.sort(key=lambda b: b[1])
    groups = []
    current_group = [boxes[0]]
    for i in range(1, len(boxes)):
        if abs(boxes[i][1] - current_group[0][1]) < row_threshold:
            current_group.append(boxes[i])
        else:
            groups.append(current_group)
            current_group = [boxes[i]]
    groups.append(current_group)

    for group in groups[:TARGET_ROWS]:
        group.sort(key=lambda b: b[0])
        for i in range(len(group) - 1):
            x1, y1, x2, y2 = group[i]
            next_x1, _, next_x2, _ = group[i + 1]
            gap = next_x1 - x2
            avg_w = (x2 - x1 + next_x2 - next_x1) / 2
            if gap > avg_w * 0.5 and gap < avg_w * 2.0 and len(group) < TARGET_COLS:
                new_x = x2
                new_y = (y1 + y2) // 2
                new_box = (new_x, new_y - 20, new_x + (x2 - x1), new_y + 20)
                if not any(overlap_check(new_box, b) for b in boxes):
                    cv2.rectangle(image, (new_x, new_y - 20), (new_x + (x2 - x1), new_y + 20), (0, 255, 255), 2)
                    boxes.append(new_box)
                    cnt += 1
    return len(boxes), image

# OCR处理（待完善）
def process_ocr(roi_gray, denoise_h, clip_limit, psm):
    # 实现基于1.8.7的预处理和识别逻辑
    pass

# 主流程
def process_image():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取 {IMG_PATH}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"tuning_log_{VERSION}_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"OBU检测工具日志（版本 {VERSION}） - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("========================================\n")

    if args.auto_tuning:
        # 自动调优逻辑
        param_combinations = list(itertools.product(
            UPPER_BLACK_V_RANGE, MORPH_KERNEL_SIZE_RANGE, MIN_AREA_RANGE, MIN_CONTOUR_SIZE_RANGE
        ))
        total_combinations = len(param_combinations)
        best_count = 0
        best_params = None
        for idx, (upper_v, morph_size, min_area, (min_w, min_h)) in enumerate(param_combinations, 1):
            img_copy = img.copy()
            mask = preprocess_image(img_copy, upper_v, morph_size)
            cnt, img_copy = detect_and_bound_contours(img_copy, mask, min_area, min_w, min_h)
            cnt, img_copy = group_and_complete(img_copy, [(x, y, x2, y2) for x, y, x2, y2 in drawn_boxes], ROW_THRESHOLD)
            if 17 <= cnt <= 25:
                output_path = os.path.join(SELECTED_DIR, f"marked_{timestamp}_cnt{cnt}.jpg")
                cv2.imwrite(output_path, img_copy)
            if cnt > best_count:
                best_count = cnt
                best_params = (upper_v, morph_size, min_area, min_w, min_h)
            with open(log_file, "a") as f:
                f.write(f"组合 {idx}/{total_combinations}: upper_v={upper_v}, morph_size={morph_size}, min_area={min_area}, min_w={min_w}, min_h={min_h}, cnt={cnt}\n")
        with open(log_file, "a") as f:
            f.write(f"\n=== 调优总结 ===\n总共测试 {total_combinations} 组，最佳成绩: {best_count} 个OBU，参数: {best_params}\n")
    else:
        # 手动模式
        mask = preprocess_image(img, args.upper_black_v, args.morph_size)
        cnt, img = detect_and_bound_contours(img, mask, args.min_area, args.min_w, args.min_h)
        cnt, img = group_and_complete(img, [(x, y, x2, y2) for x, y, x2, y2 in drawn_boxes], ROW_THRESHOLD)
        output_path = os.path.join(OUT_DIR, f"marked_{timestamp}_cnt{cnt}.jpg")
        cv2.imwrite(output_path, img)
        with open(log_file, "a") as f:
            f.write(f"手动模式: upper_v={args.upper_black_v}, morph_size={args.morph_size}, min_area={args.min_area}, min_w={args.min_w}, min_h={args.min_h}, cnt={cnt}\n")

    if args.enable_ocr:
        # OCR调优（待实现）
        pass

    print(f"[INFO] 运行完成，日志: {log_file}, 图片: {SELECTED_DIR}")

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(SELECTED_DIR, exist_ok=True)
    process_image()