import os
import cv2
import numpy as np
from datetime import datetime
import argparse
import itertools
import sys
import easyocr
import multiprocessing as mp
from pyzbar import pyzbar

# 版本号
VERSION = "v2.9"

# 参数解析
parser = argparse.ArgumentParser(description=f"OBU检测工具（版本 {VERSION}）")
parser.add_argument("--img_path", default="3.jpg", help="输入图像路径")
parser.add_argument("--auto_tuning", action="store_true", help="是否启用自动调优模式")
parser.add_argument("--enable_ocr", action="store_true", help="是否启用OCR模块")
parser.add_argument("--upper_black_v", type=int, default=60, help="手动模式下的UPPER_BLACK_V")
parser.add_argument("--morph_size", type=int, default=7, help="手动模式下的MORPH_KERNEL_SIZE")
parser.add_argument("--min_area", type=int, default=600, help="手动模式下的MIN_AREA")
parser.add_argument("--min_w", type=int, default=80, help="手动模式下的MIN_W")
parser.add_argument("--min_h", type=int, default=40, help="手动模式下的MIN_H")
args = parser.parse_args()

# 固定参数
IMG_PATH = args.img_path
OUT_DIR = "cut_obu"
LOG_DIR = os.path.join(OUT_DIR, "log")
SELECTED_DIR = os.path.join(OUT_DIR, "selected")
TARGET_ROWS, TARGET_COLS = 4, 6  # 4×6网格
ROW_THRESHOLD = 30
PADDING = 5
MAX_AREA = 100000  # 图像面积10%上限
MIN_GAP = 10  # 新增：最小间距

# 调优参数范围（优化）
UPPER_BLACK_V_RANGE = [55, 60, 65]  # 55-65
MORPH_KERNEL_SIZE_RANGE = [7]  # 固定7
MIN_AREA_RANGE = [500, 600, 700]  # 500-700
MIN_CONTOUR_SIZE_RANGE = list(itertools.product(range(70, 101, 10), range(30, 51, 5)))  # (70, 30)到(100, 50)

# OCR参数范围（预留）
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
    if cv2.countNonZero(mask) == 0:
        print(f"[DEBUG] 掩码为空，upper_v={upper_black_v}, morph_size={morph_size}")
    return mask

# 轮廓检测与框定
def detect_and_bound_contours(image, mask, min_area, min_w, min_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_boxes = []
    cnt = 0
    if not contours:
        print("[DEBUG] 无轮廓检测到，检查掩码或参数")
        return cnt, image, drawn_boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w >= min_w and h >= min_h and min_area <= area <= MAX_AREA:
            current_box = (x, y, x + w, y + h)
            overlap = False
            for box in drawn_boxes:
                if overlap_check(current_box, box) or min_distance_check(current_box, box, min_w):
                    overlap = True
                    break
            if not overlap:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                drawn_boxes.append(current_box)
                cnt += 1
    print(f"[DEBUG] 检测到 {len(drawn_boxes)} 个框，cnt={cnt}")
    return cnt, image, drawn_boxes

# 最小间距检查
def min_distance_check(box1, box2, min_w):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    if abs(x1 - bx2) < MIN_GAP or abs(x2 - bx1) < MIN_GAP:
        if abs(y1 - by1) < min_w * 0.5:  # 垂直方向也需检查
            return True
    return False

# 网格分组与补全
def group_and_complete(image, boxes, row_threshold, target_rows, target_cols):
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
    groups = groups[:target_rows]  # 限制4行

    total_cnt = 0
    all_boxes = boxes.copy()  # 保留所有框
    for group in groups:
        group.sort(key=lambda b: b[0])
        # 放宽补全条件，多次补全
        iteration = 0
        while len(group) < target_cols and iteration < 2:  # 限制补全次数
            new_boxes = []
            for i in range(len(group) - 1):
                x1, y1, x2, y2 = group[i]
                next_x1, _, next_x2, _ = group[i + 1]
                gap = next_x1 - x2
                avg_w = (x2 - x1 + next_x2 - next_x1) / 2
                if gap > avg_w * 0.2 and gap < avg_w * 3.0:
                    new_x = x2
                    new_y = (y1 + y2) // 2
                    new_w = x2 - x1
                    new_box = (new_x, new_y - new_w//2, new_x + new_w, new_y + new_w//2)
                    if not any(overlap_check(new_box, b) or min_distance_check(new_box, b, avg_w) for b in all_boxes):
                        cv2.rectangle(image, (new_x, new_y - new_w//2), (new_x + new_w, new_y + new_w//2), (0, 255, 255), 2)
                        all_boxes.append(new_box)
                        new_boxes.append(new_box)
            group.extend(new_boxes)
            group.sort(key=lambda b: b[0])
            iteration += 1
            if not new_boxes:
                break
        # 强制补全至target_cols（针对稀疏行）
        if len(group) < target_cols:
            x1, y1, x2, y2 = group[-1]
            avg_w = x2 - x1
            new_w = avg_w  # 明确定义new_w
            for _ in range(target_cols - len(group)):
                new_x = x2 + avg_w // 2
                new_y = (y1 + y2) // 2
                new_box = (new_x, new_y - new_w//2, new_x + new_w, new_y + new_w//2)
                if not any(overlap_check(new_box, b) or min_distance_check(new_box, b, avg_w) for b in all_boxes):
                    cv2.rectangle(image, (new_x, new_y - new_w//2), (new_x + new_w, new_y + new_w//2), (0, 255, 255), 2)
                    all_boxes.append(new_box)
                    group.append(new_box)
                    x2 = new_x + avg_w
        total_cnt += min(len(group), target_cols)
    # 清理多余补全框
    valid_boxes = []
    for group in groups:
        group.sort(key=lambda b: b[0])
        valid_boxes.extend(group[:target_cols])
    print(f"[DEBUG] 补全后总框数: {len(valid_boxes)}, total_cnt={total_cnt}")
    return total_cnt, image

# 重叠检查
def overlap_check(box1, box2):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    intersection = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
    union = (x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union > 0.1 if union > 0 else False

# OCR处理（预留）
def process_ocr(roi_gray, denoise_h, clip_limit, psm, log_file=None):
    if not args.enable_ocr:
        return None
    # 待实现：1.8.7的预处理和识别逻辑
    return None

# 主流程
def process_image():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取 {IMG_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SELECTED_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"tuning_log_{VERSION}_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"OBU检测工具日志（版本 {VERSION}） - 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("========================================\n")

    if args.auto_tuning:
        param_combinations = list(itertools.product(
            UPPER_BLACK_V_RANGE, MORPH_KERNEL_SIZE_RANGE, MIN_AREA_RANGE, MIN_CONTOUR_SIZE_RANGE
        ))
        total_combinations = len(param_combinations)
        best_count = 0
        best_params = None
        print(f"[INFO] 总共 {total_combinations} 组参数组合，开始自动调优...")
        for idx, (upper_v, morph_size, min_area, (min_w, min_h)) in enumerate(param_combinations, 1):
            img_copy = img.copy()
            mask = preprocess_image(img_copy, upper_v, morph_size)
            cnt, img_copy, boxes = detect_and_bound_contours(img_copy, mask, min_area, min_w, min_h)
            cnt, img_copy = group_and_complete(img_copy, boxes, ROW_THRESHOLD, TARGET_ROWS, TARGET_COLS)
            print(f"[INFO] 组合 {idx}/{total_combinations}: upper_v={upper_v}, morph_size={morph_size}, "
                  f"min_area={min_area}, min_w={min_w}, min_h={min_h}, cnt={cnt}")
            if 20 <= cnt <= 22:  # 调整目标范围
                output_path = os.path.join(SELECTED_DIR, f"marked_{timestamp}_cnt{cnt}.jpg")
                cv2.imwrite(output_path, img_copy)
                print(f"[INFO] ROI数量 {cnt} 在目标范围内，保存结果至 {output_path}")
            if abs(cnt - 21) < abs(best_count - 21):  # 更接近21
                best_count = cnt
                best_params = (upper_v, morph_size, min_area, min_w, min_h)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"组合 {idx}/{total_combinations}: upper_v={upper_v}, morph_size={morph_size}, "
                        f"min_area={min_area}, min_w={min_w}, min_h={min_h}, cnt={cnt}\n")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== 调优总结 ===\n总共测试 {total_combinations} 组，最佳成绩: {best_count} 个OBU，"
                    f"参数: {best_params}\n")
    else:
        mask = preprocess_image(img, args.upper_black_v, args.morph_size)
        cnt, img, boxes = detect_and_bound_contours(img, mask, args.min_area, args.min_w, args.min_h)
        cnt, img = group_and_complete(img, boxes, ROW_THRESHOLD, TARGET_ROWS, TARGET_COLS)
        output_path = os.path.join(OUT_DIR, f"marked_{timestamp}_cnt{cnt}.jpg")
        cv2.imwrite(output_path, img)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"手动模式: upper_v={args.upper_black_v}, morph_size={args.morph_size}, "
                    f"min_area={args.min_area}, min_w={args.min_w}, min_h={args.min_h}, cnt={cnt}\n")

    print(f"[INFO] 运行完成，日志: {log_file}, 图片: {SELECTED_DIR}")

if __name__ == "__main__":
    process_image()