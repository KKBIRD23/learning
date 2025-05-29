VERSION = "1.7.0"  # 迭代版本号

import cv2
import numpy as np
import pytesseract
import argparse
import re
import os
import json
import time
import glob
from datetime import datetime
from pyzbar import pyzbar
import easyocr
import multiprocessing as mp
import warnings
import itertools

# 屏蔽特定警告（全局生效）
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using CPU. Note: This module is much faster with a GPU.")
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")

# 配置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 全局 EasyOCR Reader 单例
reader = None
def init_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def setup_logging(log_dir=r"C:\Users\KKBIRD\Desktop\photo\log", enable_log=True):
    if enable_log:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(log_filename, 'w', encoding='utf-8', buffering=1)
        log_file.write(f"### 日志文件创建于: {log_filename}\n")
        log_file.write(f"### 版本号: {VERSION}\n")
        log_file.flush()
        return log_file
    return None

def save_best_params(params, digit_count, digits, output_dir=r"C:\Users\KKBIRD\Desktop\photo\log"):
    best_params_file = os.path.join(output_dir, "best_params.json")
    data = {
        "best_params": params,
        "digit_count": digit_count,
        "digits": sorted(list(digits)),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def clear_process_images(output_dir=r"C:\Users\KKBIRD\Desktop\photo\process_photo"):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    for file in image_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"警告：无法删除历史过程图像 {file}: {e}")

def save_debug_image(image, title, output_dir=r"C:\Users\KKBIRD\Desktop\photo\process_photo", counter=0):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_{counter}.jpg")
    cv2.imwrite(filename, image)

def preprocess_for_digits(image, denoise_h=20, clipLimit=2.0, use_clahe=False, sharpen=True):
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(4, 4))
        enhanced = clahe.apply(image)
    else:
        enhanced = image
    denoised = cv2.medianBlur(enhanced, 3)
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        denoised = cv2.filter2D(denoised, -1, kernel)
    thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 5)
    binary = thresh_otsu if np.std(thresh_otsu) > np.std(thresh_adaptive) else thresh_adaptive
    return binary

def read_digits_from_image(image, psm=6, verbose=False, log_file=None):
    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config)
    cleaned_text = re.sub(r'[^\d]', '', text)
    if verbose and log_file:
        log_file.write(f"原始 OCR 结果 (Tesseract, psm={psm}): {cleaned_text}\n")
        log_file.flush()
    if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith(('5001', '5000')):
        cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
        if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in set(range(5201, 5247)):
            return cleaned_text
    return None

def read_digits_with_easyocr(image, verbose=False, log_file=None):
    if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) != 2:
        if verbose and log_file:
            log_file.write("EasyOCR 输入无效，跳过识别\n")
            log_file.flush()
        return None
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    try:
        result = reader.readtext(image_bgr, detail=0, allowlist='0123456789')
        text = ''.join(result)
        cleaned_text = re.sub(r'[^\d]', '', text)
        if verbose and log_file:
            log_file.write(f"原始 OCR 结果 (EasyOCR): {cleaned_text}\n")
            log_file.flush()
        if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith(('5001', '5000')):
            cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
            if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in set(range(5201, 5247)):
                return cleaned_text
    except Exception as e:
        if verbose and log_file:
            log_file.write(f"EasyOCR 识别失败: {str(e)}\n")
            log_file.flush()
    return None

def process_roi(args):
    roi_gray, roi_color, roi_x, roi_y, gray_image, denoise_h, clipLimit, psm, verbose, log_queue = args
    log_messages = []
    digit_roi = roi_gray
    for sharpen in [True, False]:
        digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h, clipLimit, sharpen=sharpen)
        digit_roi_raw = digit_roi
        digits_easyocr = read_digits_with_easyocr(digit_roi_processed, verbose, None)
        if digits_easyocr:
            log_messages.append(f"EasyOCR 识别结果: {digits_easyocr}\n")
            return digits_easyocr, log_messages
        digits_tesseract = read_digits_from_image(digit_roi_processed, psm, verbose, None)
        if digits_tesseract:
            log_messages.append(f"Tesseract 识别结果: {digits_tesseract}\n")
            return digits_tesseract, log_messages
        digits_easyocr_raw = read_digits_with_easyocr(digit_roi_raw, verbose, None)
        if digits_easyocr_raw:
            log_messages.append(f"EasyOCR 原始结果: {digits_easyocr_raw}\n")
            return digits_easyocr_raw, log_messages
        digits_tesseract_raw = read_digits_from_image(digit_roi_raw, psm, verbose, None)
        if digits_tesseract_raw:
            log_messages.append(f"Tesseract 原始结果: {digits_tesseract_raw}\n")
            return digits_tesseract_raw, log_messages
    return None, log_messages

def tune_digit_extraction_params(roi_gray, roi_color, roi_x, roi_y, gray_image, verbose=False, log_file=None):
    denoise_h = 20
    clipLimit = 2.0
    psm = 6
    best_params = {'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm}
    best_digit = None
    digit_roi = roi_gray

    for sharpen in [True, False]:
        digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h, clipLimit, sharpen=sharpen)
        digit = read_digits_with_easyocr(digit_roi_processed, verbose, log_file)
        if digit:
            return digit_roi, best_params, digit
        digit = read_digits_from_image(digit_roi_processed, psm, verbose, log_file)
        if digit:
            return digit_roi, best_params, digit

    h, w = roi_gray.shape
    for part in [0, h//2]:
        roi_part = roi_gray[part:part+h//2, 0:w]
        blurred = cv2.GaussianBlur(roi_part, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours_roi, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_roi:
            contours_roi = sorted(contours_roi, key=cv2.contourArea, reverse=True)
            for contour in contours_roi[:5]:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / float(h_contour)
                if 2.0 < aspect_ratio < 5.0 and h_contour > 10 and w_contour > 30:
                    digit_roi_y = max(0, roi_y + part + y)
                    digit_roi_h = h_contour * 2
                    digit_roi_x = roi_x + x
                    digit_roi_w = w_contour
                    if (digit_roi_y + digit_roi_h <= gray_image.shape[0] and
                        digit_roi_x + digit_roi_w <= gray_image.shape[1] and
                        digit_roi_h > 0 and digit_roi_w > 0):
                        digit_roi = gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
                        for sharpen in [True, False]:
                            digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h, clipLimit, sharpen=sharpen)
                            digit = read_digits_with_easyocr(digit_roi_processed, verbose, log_file)
                            if digit:
                                return digit_roi, best_params, digit
                            digit = read_digits_from_image(digit_roi_processed, psm, verbose, log_file)
                            if digit:
                                return digit_roi, best_params, digit
                    break
    return digit_roi, best_params, best_digit

def evaluate_roi_distribution(contours, image_shape, log_file=None):
    if not contours:
        return 0, 0
    center_points = [(x + (x2 - x) // 2, y + (y2 - y) // 2) for x, y, x2, y2 in contours]
    if len(center_points) <= 1:
        return 1, 1
    center_points.sort(key=lambda x: (x[1], x[0]))
    rows = []
    current_row = [center_points[0]]
    for i in range(1, len(center_points)):
        if abs(center_points[i][1] - current_row[0][1]) < 50:
            current_row.append(center_points[i])
        else:
            rows.append(current_row)
            current_row = [center_points[i]]
    rows.append(current_row)
    row_count = len(rows)
    col_count = max(len(row) for row in rows) if rows else 0
    if log_file:
        log_file.write(f"推断网格模式: {row_count} 行 x {col_count} 列\n")
        log_file.flush()
    return row_count, col_count

def detect_contours(image, gray_image, hsv_image, upper_black_v, canny_low, canny_high, merge_dist, target_roi_count=21, verbose=False, log_file=None):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, upper_black_v])
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    edges = cv2.Canny(mask, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_obu_area = int(0.003 * gray_image.shape[0] * gray_image.shape[1])
    max_obu_area = int(0.05 * gray_image.shape[0] * gray_image.shape[1])
    contours_to_process = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_obu_area < area < max_obu_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.33 < aspect_ratio < 3.0 and w > 50 and h > 50:
                padding = 5
                roi_x = max(0, x - padding)
                roi_y = max(0, y - padding)
                roi_w = min(gray_image.shape[1] - roi_x, w + 2 * padding)
                roi_h = min(gray_image.shape[0] - roi_y, h + 2 * padding)
                contours_to_process.append([roi_x, roi_y, roi_x + roi_w, roi_y + roi_h])

    merged_contours = []
    for contour in contours_to_process:
        roi_x, roi_y, roi_x2, roi_y2 = contour
        merged = False
        for i, merged_contour in enumerate(merged_contours):
            m_x, m_y, m_x2, m_y2 = merged_contour
            if (abs(roi_x - m_x) < merge_dist and abs(roi_y - m_y) < merge_dist):
                merged_contours[i] = [min(roi_x, m_x), min(roi_y, m_y), max(roi_x2, m_x2), max(roi_y2, m_y2)]
                merged = True
                break
        if not merged:
            merged_contours.append(contour)

    return merged_contours

def tune_roi_detection_params(image, gray_image, hsv_image, target_roi_count=21, verbose=False, log_file=None):
    # 参数范围和步长
    upper_black_v_values = list(range(20, 81, 5))
    canny_low_values = list(range(5, 61, 5))
    canny_high_values = list(range(60, 161, 10))
    merge_dist_values = list(range(3, 34, 3))

    # 初始参数
    best_params = {'upper_black_v': 40, 'canny_low': 30, 'canny_high': 100, 'merge_dist': 5}
    best_contours = []
    best_score = 0
    best_digit_count = 0
    best_digits = set()

    # 8小时时间限制（秒）
    max_duration = 8 * 60 * 60  # 28,800秒
    start_time = time.time()

    # 参数组合
    param_combinations = list(itertools.product(upper_black_v_values, canny_low_values, canny_high_values, merge_dist_values))
    total_combinations = len(param_combinations)
    attempted_combinations = set()

    attempt = 0
    try:
        while time.time() - start_time < max_duration:
            attempt += 1
            # 选择未尝试的参数组合
            remaining_combinations = [comb for comb in param_combinations if comb not in attempted_combinations]
            if not remaining_combinations:
                if verbose and log_file:
                    log_file.write("所有参数组合已尝试，停止调优\n")
                    log_file.flush()
                break
            upper_black_v, canny_low, canny_high, merge_dist = remaining_combinations[0]
            attempted_combinations.add((upper_black_v, canny_low, canny_high, merge_dist))

            if verbose and log_file:
                log_file.write(f"自动 ROI 调优尝试 {attempt}: upper_black_v={upper_black_v}, canny_low={canny_low}, canny_high={canny_high}, merge_dist={merge_dist}\n")
                log_file.flush()

            # 检测ROI
            contours = detect_contours(image, gray_image, hsv_image, upper_black_v, canny_low, canny_high, merge_dist, target_roi_count, verbose, log_file)
            roi_count = len(contours)

            # 调整有效范围为5-40
            if roi_count < 5 or roi_count > 40:
                if verbose and log_file:
                    log_file.write(f"ROI 数量 {roi_count} 超出有效范围（5-40），跳过\n")
                    log_file.flush()
                continue

            # 评估分布
            row_count, col_count = evaluate_roi_distribution(contours, gray_image.shape, log_file)
            distribution_score = row_count * col_count if row_count * col_count <= 21 else 21

            # 处理ROI并识别数字
            all_valid_digits = set()
            tasks = []
            log_queue = mp.Manager().Queue()
            roi_counter = 0
            for contour in contours:
                roi_x, roi_y, roi_x2, roi_y2 = contour
                roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
                if roi_w <= 20 or roi_h <= 20:
                    continue
                roi_counter += 1
                obu_roi_gray = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                obu_roi_color = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                if obu_roi_gray.size == 0 or len(obu_roi_gray.shape) != 2:
                    continue
                tasks.append((obu_roi_gray, obu_roi_color, roi_x, roi_y, gray_image, 20, 2.0, 6, verbose, log_queue))

            if tasks:
                pool = mp.Pool(mp.cpu_count(), initializer=init_reader)
                results = pool.map(process_roi, tasks)
                pool.close()
                pool.join()
                for i, (digit, logs) in enumerate(results):
                    roi_idx = i + 1
                    for log_msg in logs:
                        if verbose and log_file:
                            log_file.write(f"ROI {roi_idx} {log_msg}")
                            log_file.flush()
                    if digit:
                        all_valid_digits.add(digit)

            digit_count = len(all_valid_digits)
            score = digit_count * 100 + abs(roi_count - target_roi_count) * -1 + distribution_score

            if verbose and log_file:
                log_file.write(f"尝试 {attempt} 结果: ROI数量={roi_count}, 分布={row_count}x{col_count}, 识别数字={digit_count}, 得分={score}\n")
                log_file.flush()

            if digit_count > best_digit_count or (digit_count == best_digit_count and score > best_score):
                best_digit_count = digit_count
                best_score = score
                best_params = {'upper_black_v': upper_black_v, 'canny_low': canny_low, 'canny_high': canny_high, 'merge_dist': merge_dist}
                best_contours = contours
                best_digits = all_valid_digits
                save_best_params(best_params, best_digit_count, best_digits)

            if digit_count >= 20:  # 接近最大值21，提前终止
                if verbose and log_file:
                    log_file.write("识别数字数量接近目标（≥20），提前终止调优\n")
                    log_file.flush()
                break

            elapsed = time.time() - start_time
            remaining = max_duration - elapsed
            if attempt % 100 == 0:  # 每100次保存中间结果
                save_best_params(best_params, best_digit_count, best_digits)
            if verbose and log_file:
                log_file.write(f"已用时: {elapsed:.2f}秒，剩余时间: {remaining:.2f}秒，尝试进度: {len(attempted_combinations)}/{total_combinations}\n")
                log_file.flush()

    except KeyboardInterrupt:
        if verbose and log_file:
            log_file.write("用户中断调优，保存当前最优结果\n")
            log_file.flush()
        save_best_params(best_params, best_digit_count, best_digits)
        raise

    if verbose and log_file:
        log_file.write(f"自动 ROI 调优完成，最优参数: {best_params}, 识别数字数量: {best_digit_count}\n")
        log_file.flush()

    return best_contours, best_params

def process_image_for_dpm_barcodes(image_path, debug_visualization=False, manual_rois=None,
                                 denoise_h=20, psm=6, verbose=True, auto_close=False, log_file=None):
    start_time = time.time()
    all_valid_digits = set()
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法加载图像 {image_path}")
        if log_file:
            log_file.write(f"错误：无法加载图像 {image_path}\n")
            log_file.flush()
        return []

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    if debug_visualization:
        cv2.imshow("0. Original", cv2.resize(original_image, (0,0), fx=0.15, fy=0.15))
        save_debug_image(cv2.resize(original_image, (0,0), fx=0.15, fy=0.15), "Original")

    if not manual_rois:
        contours_to_process, roi_params = tune_roi_detection_params(original_image, gray_image, hsv_image, target_roi_count=21, verbose=verbose, log_file=log_file)
    else:
        valid_contours = []
        for roi in manual_rois:
            roi_x, roi_y, roi_x2, roi_y2 = roi
            roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
            if roi_w <= 20 or roi_h <= 20:
                continue
            valid_contours.append(roi)
        contours_to_process = valid_contours

    roi_counter = 0
    annotated_image = original_image.copy()
    center_points = []
    digit_params = {'denoise_h': denoise_h, 'clipLimit': 2.0, 'psm': psm}

    tasks = []
    log_queue = mp.Manager().Queue()

    for contour in contours_to_process:
        roi_x, roi_y, roi_x2, roi_y2 = contour
        roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
        if roi_w <= 20 or roi_h <= 20:
            continue
        roi_counter += 1
        center_x, center_y = roi_x + roi_w // 2, roi_y + roi_h // 2
        center_points.append((center_x, center_y))
        obu_roi_gray = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        obu_roi_color = original_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if obu_roi_gray.size == 0 or len(obu_roi_gray.shape) != 2:
            continue

        if verbose and log_file:
            log_file.write(f"ROI {roi_counter} 坐标: (x1={roi_x}, y1={roi_y}, x2={roi_x2}, y2={roi_y2})\n")
            log_file.flush()

        if debug_visualization:
            cv2.rectangle(annotated_image, (roi_x, roi_y), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(roi_counter), (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(f"3. ROI {roi_counter} Original", obu_roi_gray)
            save_debug_image(obu_roi_gray, f"ROI_{roi_counter}_Original", counter=roi_counter)

        if roi_counter == 1:
            digit_roi, digit_params, digit = tune_digit_extraction_params(obu_roi_gray, obu_roi_color, roi_x, roi_y, gray_image, verbose, log_file)
            if digit:
                all_valid_digits.add(digit)
            if verbose and log_file:
                log_file.write(f"ROI 1 数字提取调优完成，最优参数: {digit_params}\n")
                log_file.flush()
        tasks.append((obu_roi_gray, obu_roi_color, roi_x, roi_y, gray_image, digit_params['denoise_h'], digit_params['clipLimit'], digit_params['psm'], verbose, log_queue))

    if tasks:
        pool = mp.Pool(mp.cpu_count(), initializer=init_reader)
        results = pool.map(process_roi, tasks[1:])
        pool.close()
        pool.join()
        for i, (digit, logs) in enumerate(results):
            roi_idx = i + 2
            for log_msg in logs:
                if verbose and log_file:
                    log_file.write(f"ROI {roi_idx} {log_msg}")
                    log_file.flush()
            if digit:
                all_valid_digits.add(digit)

    if center_points and debug_visualization:
        if len(center_points) > 1:
            center_points.sort(key=lambda x: (x[1], x[0]))
            rows = []
            current_row = [center_points[0]]
            for i in range(1, len(center_points)):
                if abs(center_points[i][1] - current_row[0][1]) < 50:
                    current_row.append(center_points[i])
                else:
                    rows.append(current_row)
                    current_row = [center_points[i]]
            rows.append(current_row)
            row_count = len(rows)
            col_count = max(len(row) for row in rows) if rows else 0
            for i, row in enumerate(rows):
                for j, (x, y) in enumerate(row):
                    cv2.putText(annotated_image, f"({i},{j})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            log_file.write(f"推断网格模式: {row_count} 行 x {col_count} 列\n")
            log_file.flush()
        save_debug_image(annotated_image, "Grid_Annotation")

    elapsed_time = time.time() - start_time
    if debug_visualization:
        print(f"\n总共处理了 {roi_counter} 个候选OBU区域。")
        print(f"当前识别到的有效数字 (集合): {all_valid_digits}")
        if log_file:
            log_file.write(f"\n总共处理了 {roi_counter} 个候选OBU区域。\n")
            log_file.write(f"当前识别到的有效数字 (集合): {all_valid_digits}\n")
            log_file.flush()
        if not auto_close:
            print("按任意键关闭所有调试窗口...")
            if log_file:
                log_file.write("按任意键关闭所有调试窗口...\n")
                log_file.flush()
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sorted(list(all_valid_digits)), elapsed_time

if __name__ == "__main__":
    init_reader()
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        os.chdir(script_dir)
    except Exception as e:
        print(f"### 设置工作目录失败: {e}")
        raise

    clear_process_images(r"C:\Users\KKBIRD\Desktop\photo\process_photo")

    parser = argparse.ArgumentParser(description="OBU Image Processing for Digit Recognition")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--manual-rois", type=str, help="Path to manual ROIs JSON file")
    parser.add_argument("--denoise_h", type=int, default=20, help="Denoising strength")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode")
    parser.add_argument("--log", action="store_true", help="Enable logging to file")
    parser.add_argument("--auto-close", action="store_true", help="Auto close debug windows")
    args = parser.parse_args()

    manual_rois = None
    if args.manual_rois:
        try:
            with open(args.manual_rois, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'annotations' in data:
                    manual_rois = [[int(ann['bbox'][0]), int(ann['bbox'][1]),
                                   int(ann['bbox'][0] + ann['bbox'][2]),
                                   int(ann['bbox'][1] + ann['bbox'][3])]
                                  for ann in data['annotations']]
                else:
                    manual_rois = json.load(f)
        except Exception as e:
            print(f"### 加载 ROI 文件失败: {e}")
            manual_rois = None

    try:
        log_file = setup_logging(r"C:\Users\KKBIRD\Desktop\photo\log", args.log)
    except Exception as e:
        print(f"### 初始化日志失败: {e}")
        raise

    try:
        digits, elapsed_time = process_image_for_dpm_barcodes(
            args.image_path,
            debug_visualization=args.debug,
            manual_rois=manual_rois,
            denoise_h=args.denoise_h,
            psm=args.psm,
            verbose=True,
            auto_close=args.auto_close,
            log_file=log_file
        )
        print(f"\n最终数字识别结果: {digits}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        if log_file:
            log_file.write(f"\n最终数字识别结果: {digits}\n")
            log_file.write(f"耗时: {elapsed_time:.2f} 秒\n")
            log_file.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.flush()
            print(f"日志已保存至: {log_file.name}")
    except Exception as e:
        print(f"### 处理图像失败: {e}")
        raise
    finally:
        if log_file:
            log_file.close()