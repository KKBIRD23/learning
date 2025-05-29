VERSION = "0.6.0"  # 迭代版本号

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
from pyzbar.pyzbar import ZBarSymbol
import easyocr

# 配置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 全局 EasyOCR Reader 单例
reader = easyocr.Reader(['en'], gpu=False)

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

def preprocess_for_digits(image, denoise_h=20, clipLimit=2.0, use_clahe=True, sharpen=True):
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(4, 4))
        enhanced = clahe.apply(image)
    else:
        enhanced = image
    # 中值滤波去噪
    denoised = cv2.medianBlur(enhanced, 3)
    # 可选锐化
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        denoised = cv2.filter2D(denoised, -1, kernel)
    # 结合 Otsu 和自适应阈值
    thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 5)
    binary = thresh_otsu if np.std(thresh_otsu) > np.std(thresh_adaptive) else thresh_adaptive
    return binary

def enhance_for_barcode(image, alpha=1.5, sharpen_strength=12):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=0)
    kernel = np.array([[-1,-1,-1], [-1,sharpen_strength,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def detect_white_button(roi_color):
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        if y < roi_color.shape[0] // 3:
            return x, y, w, h
    return None

def read_digits_from_image(image, psm=6, verbose=False, log_file=None):
    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config)
    cleaned_text = re.sub(r'[^\d]', '', text)
    if verbose and log_file:
        log_file.write(f"原始 OCR 结果 (Tesseract, psm={psm}): {cleaned_text}\n")
        log_file.flush()
    if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith("5001"):
        cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
        if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in set(range(5201, 5247)):
            return cleaned_text
    return None

def read_digits_with_easyocr(image, verbose=False, log_file=None):
    if not isinstance(image, np.ndarray):
        if verbose and log_file:
            log_file.write("EasyOCR 输入不是 numpy 数组，跳过识别\n")
            log_file.flush()
        return None
    if image.size == 0 or len(image.shape) != 2 or image.shape[0] < 10 or image.shape[1] < 10:
        if verbose and log_file:
            log_file.write(f"EasyOCR 输入图像无效 (shape: {image.shape})，跳过识别\n")
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
        if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith("5001"):
            cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
            if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in set(range(5201, 5247)):
                return cleaned_text
    except Exception as e:
        if verbose and log_file:
            log_file.write(f"EasyOCR 识别失败: {str(e)}\n")
            log_file.flush()
    return None

def tune_digit_extraction_params(roi_gray, roi_color, roi_x, roi_y, gray_image, verbose=False, log_file=None):
    denoise_h_values = [10, 15, 20, 25]
    clipLimit_values = [2.0, 3.0, 4.0, 5.0]
    psm_values = [6, 7, 8]  # 优化 PSM 模式，适合单行数字
    best_params = {'denoise_h': 20, 'clipLimit': 2.0, 'psm': 6}
    best_digit = None
    digit_roi = None

    # 优先直接使用整个 ROI 进行 OCR
    digit_roi = roi_gray
    for denoise_h in denoise_h_values:
        for clipLimit in clipLimit_values:
            for sharpen in [True, False]:
                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                for psm in psm_values:
                    # 优先 EasyOCR
                    digit = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                    if digit:
                        best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                        return digit_roi, best_params, digit
                    # 备选 Tesseract
                    digit = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                    if digit:
                        best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                        return digit_roi, best_params, digit

    # 如果直接 OCR 失败，尝试上下分割
    h, w = roi_gray.shape
    upper_roi = roi_gray[0:h//2, 0:w]
    lower_roi = roi_gray[h//2:h, 0:w]

    # 上半部分
    blurred = cv2.GaussianBlur(upper_roi, (5, 5), 0)
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
        best_contour = None
        for contour in contours_roi[:5]:
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            aspect_ratio = w_contour / float(h_contour)
            if 2.0 < aspect_ratio < 5.0 and h_contour > 10 and w_contour > 30:
                best_contour = contour
                break
        if best_contour is not None:
            x, y, w_contour, h_contour = cv2.boundingRect(best_contour)
            digit_roi_y = max(0, roi_y + y)
            digit_roi_h = h_contour * 2
            digit_roi_x = roi_x + x
            digit_roi_w = w_contour
            if (digit_roi_y + digit_roi_h <= gray_image.shape[0] and
                digit_roi_x + digit_roi_w <= gray_image.shape[1] and
                digit_roi_h > 0 and digit_roi_w > 0 and
                digit_roi_h >= 10 and digit_roi_w >= 10):
                digit_roi = gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
                for denoise_h in denoise_h_values:
                    for clipLimit in clipLimit_values:
                        for sharpen in [True, False]:
                            digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                            for psm in psm_values:
                                digit = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                                if digit:
                                    best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                                    return digit_roi, best_params, digit
                                digit = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                                if digit:
                                    best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                                    return digit_roi, best_params, digit

    # 下半部分
    if digit_roi is None or best_digit is None:
        blurred = cv2.GaussianBlur(lower_roi, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x**2 + grad_y**2)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours_roi, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_roi:
            contours_roi = sorted(contours_roi, key=cv2.contourArea, reverse=True)
            best_contour = None
            for contour in contours_roi[:5]:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / float(h_contour)
                if 2.0 < aspect_ratio < 5.0 and h_contour > 10 and w_contour > 30:
                    best_contour = contour
                    break
            if best_contour is not None:
                x, y, w_contour, h_contour = cv2.boundingRect(best_contour)
                digit_roi_y = max(0, roi_y + h//2 + y)
                digit_roi_h = h_contour * 2
                digit_roi_x = roi_x + x
                digit_roi_w = w_contour
                if (digit_roi_y + digit_roi_h <= gray_image.shape[0] and
                    digit_roi_x + digit_roi_w <= gray_image.shape[1] and
                    digit_roi_h > 0 and digit_roi_w > 0 and
                    digit_roi_h >= 10 and digit_roi_w >= 10):
                    digit_roi = gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
                    for denoise_h in denoise_h_values:
                        for clipLimit in clipLimit_values:
                            for sharpen in [True, False]:
                                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                                for psm in psm_values:
                                    digit = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                                    if digit:
                                        best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                                        return digit_roi, best_params, digit
                                    digit = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                                    if digit:
                                        best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                                        return digit_roi, best_params, digit

    return digit_roi, best_params, best_digit

def tune_roi_detection_params(image, target_roi_count=21, verbose=False, log_file=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1.0)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    brightness_median = np.median(v_channel)
    upper_black_v = int(brightness_median * 1.5)
    upper_black_v = max(30, min(50, upper_black_v))

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_std = np.std(grad)
    canny_low = int(grad_std * 0.5)
    canny_high = int(grad_std * 1.5)
    canny_low = max(20, min(40, canny_low))
    canny_high = max(80, min(120, canny_high))

    merge_dist = 5  # 进一步收紧初始合并距离
    best_params = {'upper_black_v': upper_black_v, 'canny_low': canny_low, 'canny_high': canny_high, 'merge_dist': merge_dist}
    best_contours = []

    for attempt in range(3):
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, upper_black_v])
        mask = cv2.inRange(hsv_image, lower_black, upper_black)
        edges = cv2.Canny(mask, canny_low, canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_obu_area = int(0.003 * gray_image.shape[0] * gray_image.shape[1])  # 收紧面积范围
        max_obu_area = int(0.05 * gray_image.shape[0] * gray_image.shape[1])
        contours_to_process = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_obu_area < area < max_obu_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                # 增加边界检查
                if (0.33 < aspect_ratio < 3.0 and w > 50 and h > 50 and
                    x > 10 and y > 10 and
                    x + w < gray_image.shape[1] - 10 and
                    y + h < gray_image.shape[0] - 10):
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
                if (abs(roi_x - m_x) < merge_dist and abs(roi_y - m_y) < merge_dist) or (abs(roi_x2 - m_x2) < merge_dist and abs(roi_y2 - m_y2) < merge_dist):
                    merged_contours[i] = [min(roi_x, m_x), min(roi_y, m_y), max(roi_x2, m_x2), max(roi_y2, m_y2)]
                    merged = True
                    break
            if not merged:
                merged_contours.append(contour)

        # 动态调整 merge_dist
        if len(merged_contours) > target_roi_count:
            contours_in_roi = []
            for roi in merged_contours:
                x, y, x2, y2 = roi
                roi_img = gray_image[y:y2, x:x2]
                roi_edges = cv2.Canny(roi_img, canny_low, canny_high)
                roi_contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_in_roi.append(len(roi_contours))
            if max(contours_in_roi) > 1:
                merge_dist = max(3, merge_dist - 2)

        roi_count = len(merged_contours)
        if verbose and log_file:
            log_file.write(f"自动 ROI 调优尝试 {attempt+1}: 检测到 {roi_count} 个 ROI\n")
            log_file.write(f"当前参数: upper_black_v={upper_black_v}, canny_low={canny_low}, canny_high={canny_high}, merge_dist={merge_dist}\n")
            log_file.flush()

        if roi_count == target_roi_count:
            best_contours = merged_contours
            break
        elif roi_count < target_roi_count:
            upper_black_v += 5
            canny_low -= 5
            canny_high -= 10
            merge_dist += 3
        else:
            upper_black_v -= 5
            canny_low += 5
            canny_high += 10
            merge_dist -= 2

        upper_black_v = max(30, min(50, upper_black_v))
        canny_low = max(20, min(40, canny_low))
        canny_high = max(80, min(120, canny_high))
        merge_dist = max(3, min(15, merge_dist))

        best_params.update({'upper_black_v': upper_black_v, 'canny_low': canny_low, 'canny_high': canny_high, 'merge_dist': merge_dist})
        best_contours = merged_contours

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

    if debug_visualization:
        cv2.imshow("0. Original", cv2.resize(original_image, (0,0), fx=0.15, fy=0.15))
        save_debug_image(cv2.resize(original_image, (0,0), fx=0.15, fy=0.15), "Original")

    if not manual_rois:
        contours_to_process, roi_params = tune_roi_detection_params(original_image, target_roi_count=21, verbose=verbose, log_file=log_file)
        if verbose and log_file:
            log_file.write(f"自动 ROI 调优完成，最优参数: {roi_params}\n")
            log_file.flush()
    else:
        valid_contours = []
        for roi in manual_rois:
            roi_x, roi_y, roi_x2, roi_y2 = roi
            roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
            if (roi_x < 0 or roi_y < 0 or roi_x2 > gray_image.shape[1] or roi_y2 > gray_image.shape[0] or
                roi_w <= 20 or roi_h <= 20):
                if verbose and log_file:
                    log_file.write(f"手动 ROI 无效 (x1={roi_x}, y1={roi_y}, x2={roi_x2}, y2={roi_y2})，跳过\n")
                    log_file.flush()
                continue
            valid_contours.append(roi)
        contours_to_process = valid_contours

    roi_counter = 0
    annotated_image = original_image.copy()
    center_points = []
    digit_params = {'denoise_h': denoise_h, 'clipLimit': 2.0, 'psm': psm}

    for contour in contours_to_process:
        roi_x, roi_y, roi_x2, roi_y2 = contour
        roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
        if roi_w <= 20 or roi_h <= 20:
            if verbose and log_file:
                log_file.write(f"ROI 尺寸过小 ({roi_w}x{roi_h})，跳过\n")
                log_file.flush()
            continue
        roi_counter += 1
        center_x, center_y = roi_x + roi_w // 2, roi_y + roi_h // 2
        center_points.append((center_x, center_y))
        obu_roi_gray = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        obu_roi_color = original_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if obu_roi_gray.size == 0 or len(obu_roi_gray.shape) != 2:
            if verbose and log_file:
                log_file.write(f"ROI {roi_counter} 提取失败，区域为空或形状无效 (shape: {obu_roi_gray.shape})，跳过\n")
                log_file.flush()
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
            digit_roi, digit_params, digit = tune_digit_extraction_params(
                obu_roi_gray, obu_roi_color, roi_x, roi_y, gray_image, verbose=verbose, log_file=log_file
            )
            if digit:
                all_valid_digits.add(digit)
            if verbose and log_file:
                log_file.write(f"ROI 1 数字提取调优完成，最优参数: {digit_params}\n")
                log_file.flush()
            if digit_roi is not None and digit_roi.size > 0 and len(digit_roi.shape) == 2:
                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=digit_params['denoise_h'], clipLimit=digit_params['clipLimit'])
            else:
                if verbose and log_file:
                    log_file.write(f"ROI {roi_counter} 数字区域提取失败，回退到整个 ROI\n")
                    log_file.flush()
                digit_roi = obu_roi_gray
                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=digit_params['denoise_h'], clipLimit=digit_params['clipLimit'])
        else:
            denoise_h = digit_params['denoise_h']
            clipLimit = digit_params['clipLimit']
            psm = digit_params['psm']

            # 直接使用整个 ROI 进行 OCR
            digit_roi = obu_roi_gray
            for sharpen in [True, False]:
                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                digit_roi_raw = digit_roi

                digits_easyocr = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                if digits_easyocr:
                    all_valid_digits.add(digits_easyocr)
                digits_tesseract = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                if digits_tesseract:
                    all_valid_digits.add(digits_tesseract)
                digits_easyocr_raw = read_digits_with_easyocr(digit_roi_raw, verbose=verbose, log_file=log_file)
                if digits_easyocr_raw:
                    all_valid_digits.add(digits_easyocr_raw)
                digits_tesseract_raw = read_digits_from_image(digit_roi_raw, psm=psm, verbose=verbose, log_file=log_file)
                if digits_tesseract_raw:
                    all_valid_digits.add(digits_tesseract_raw)

                if digits_easyocr or digits_tesseract or digits_easyocr_raw or digits_tesseract_raw:
                    break

            # 如果直接 OCR 失败，尝试上下分割
            if not (digits_easyocr or digits_tesseract or digits_easyocr_raw or digits_tesseract_raw):
                h, w = obu_roi_gray.shape
                upper_roi = obu_roi_gray[0:h//2, 0:w]
                lower_roi = obu_roi_gray[h//2:h, 0:w]

                blurred = cv2.GaussianBlur(upper_roi, (5, 5), 0)
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
                    best_contour = None
                    for contour in contours_roi[:5]:
                        x, y, w_contour, h_contour = cv2.boundingRect(contour)
                        aspect_ratio = w_contour / float(h_contour)
                        if 2.0 < aspect_ratio < 5.0 and h_contour > 10 and w_contour > 30:
                            best_contour = contour
                            break
                    if best_contour is not None:
                        x, y, w_contour, h_contour = cv2.boundingRect(best_contour)
                        digit_roi_y = max(0, roi_y + y)
                        digit_roi_h = h_contour * 2
                        digit_roi_x = roi_x + x
                        digit_roi_w = w_contour
                        if (digit_roi_y + digit_roi_h <= gray_image.shape[0] and
                            digit_roi_x + digit_roi_w <= gray_image.shape[1] and
                            digit_roi_h > 0 and digit_roi_w > 0 and
                            digit_roi_h >= 10 and digit_roi_w >= 10):
                            digit_roi = gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
                            for sharpen in [True, False]:
                                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                                digit_roi_raw = digit_roi

                                digits_easyocr = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                                if digits_easyocr:
                                    all_valid_digits.add(digits_easyocr)
                                digits_tesseract = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                                if digits_tesseract:
                                    all_valid_digits.add(digits_tesseract)
                                digits_easyocr_raw = read_digits_with_easyocr(digit_roi_raw, verbose=verbose, log_file=log_file)
                                if digits_easyocr_raw:
                                    all_valid_digits.add(digits_easyocr_raw)
                                digits_tesseract_raw = read_digits_from_image(digit_roi_raw, psm=psm, verbose=verbose, log_file=log_file)
                                if digits_tesseract_raw:
                                    all_valid_digits.add(digits_tesseract_raw)

                                if digits_easyocr or digits_tesseract or digits_easyocr_raw or digits_tesseract_raw:
                                    break

                if not (digits_easyocr or digits_tesseract or digits_easyocr_raw or digits_tesseract_raw):
                    blurred = cv2.GaussianBlur(lower_roi, (5, 5), 0)
                    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                    grad = np.sqrt(grad_x**2 + grad_y**2)
                    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    contours_roi, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_roi:
                        contours_roi = sorted(contours_roi, key=cv2.contourArea, reverse=True)
                        best_contour = None
                        for contour in contours_roi[:5]:
                            x, y, w_contour, h_contour = cv2.boundingRect(contour)
                            aspect_ratio = w_contour / float(h_contour)
                            if 2.0 < aspect_ratio < 5.0 and h_contour > 10 and w_contour > 30:
                                best_contour = contour
                                break
                        if best_contour is not None:
                            x, y, w_contour, h_contour = cv2.boundingRect(best_contour)
                            digit_roi_y = max(0, roi_y + h//2 + y)
                            digit_roi_h = h_contour * 2
                            digit_roi_x = roi_x + x
                            digit_roi_w = w_contour
                            if (digit_roi_y + digit_roi_h <= gray_image.shape[0] and
                                digit_roi_x + digit_roi_w <= gray_image.shape[1] and
                                digit_roi_h > 0 and digit_roi_w > 0 and
                                digit_roi_h >= 10 and digit_roi_w >= 10):
                                digit_roi = gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
                                for sharpen in [True, False]:
                                    digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit, sharpen=sharpen)
                                    digit_roi_raw = digit_roi

                                    digits_easyocr = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file)
                                    if digits_easyocr:
                                        all_valid_digits.add(digits_easyocr)
                                    digits_tesseract = read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                                    if digits_tesseract:
                                        all_valid_digits.add(digits_tesseract)
                                    digits_easyocr_raw = read_digits_with_easyocr(digit_roi_raw, verbose=verbose, log_file=log_file)
                                    if digits_easyocr_raw:
                                        all_valid_digits.add(digits_easyocr_raw)
                                    digits_tesseract_raw = read_digits_from_image(digit_roi_raw, psm=psm, verbose=verbose, log_file=log_file)
                                    if digits_tesseract_raw:
                                        all_valid_digits.add(digits_tesseract_raw)

                                    if digits_easyocr or digits_tesseract or digits_easyocr_raw or digits_tesseract_raw:
                                        break

        if debug_visualization and 'digit_roi_processed' in locals():
            cv2.imshow(f"4. ROI {roi_counter} Digit Area", digit_roi_processed)
            save_debug_image(digit_roi_processed, f"ROI_{roi_counter}_Digit_Area", counter=roi_counter)

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
            for i in range(row_count):
                if i < len(rows):
                    for j in range(len(rows[i])):
                        if j + 1 < len(rows[i]):
                            cv2.line(annotated_image, rows[i][j], rows[i][j + 1], (0, 0, 255), 1)
                    if i + 1 < len(rows) and rows[i] and rows[i + 1]:
                        cv2.line(annotated_image, rows[i][0], rows[i + 1][0], (0, 0, 255), 1)
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
                if not isinstance(manual_rois, list) or not all(len(roi) == 4 for roi in manual_rois):
                    raise ValueError("ROI 文件格式错误，需为 [[x1,y1,x2,y2], ...] 或 COCO 格式")
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