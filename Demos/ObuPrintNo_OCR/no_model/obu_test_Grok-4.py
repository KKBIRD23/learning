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

# 配置Tesseract的路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化EasyOCR Reader（全局单例，避免重复加载）
reader = easyocr.Reader(['en'], gpu=False)

def setup_logging(log_dir=r"C:\Users\KKBIRD\Desktop\photo\log", enable_log=True):
    """设置日志文件，如果启用则创建日志目录和文件"""
    if enable_log:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(log_filename, 'w', encoding='utf-8', buffering=1)
        log_file.write(f"### Log file created at: {log_filename}\n")
        log_file.write(f"### Version: 0.6.2\n")
        log_file.flush()
        return log_file
    return None

def clear_process_images(output_dir=r"C:\Users\KKBIRD\Desktop\photo\process_photo"):
    """清除历史调试图像"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    for file in image_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Unable to delete historical process image {file}: {e}")

def save_debug_image(image, title, output_dir=r"C:\Users\KKBIRD\Desktop\photo\process_photo", counter=0):
    """保存调试图像到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_{counter}.jpg")
    cv2.imwrite(filename, image)

def preprocess_for_digits(image, denoise_h=20, clipLimit=2.0, use_clahe=True, sharpen=True):
    """
    预处理图像以便数字识别。

    参数:
    - image: 输入灰度图像
    - denoise_h: 去噪强度，值越大去噪效果越强，但可能丢失细节
    - clipLimit: CLAHE对比度限制，值越大对比度增强越明显
    - use_clahe: 是否使用CLAHE增强对比度
    - sharpen: 是否应用锐化处理

    返回:
    - 处理后的二值化图像
    """
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(4, 4))
        enhanced = clahe.apply(image)
    else:
        enhanced = image
    denoised = cv2.medianBlur(enhanced, 3)  # 中值滤波去噪
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # 锐化核
        denoised = cv2.filter2D(denoised, -1, kernel)
    thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    binary = thresh_otsu if np.std(thresh_otsu) > np.std(thresh_adaptive) else thresh_adaptive
    return binary

def detect_barcode(image):
    """
    检测图像中的条形码，使用pyzbar并辅以备用方案。

    参数:
    - image: 输入彩色图像

    返回:
    - 检测到的条形码对象，若失败则返回None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)  # 增强对比度以提高检测率
    barcodes = pyzbar.decode(enhanced, symbols=[ZBarSymbol.CODE128])
    if barcodes:
        return barcodes[0]

    # 备用方案：使用边缘检测定位条形码区域
    edges = cv2.Canny(enhanced, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2.0 < aspect_ratio < 10.0 and w > 50:  # 假设条形码宽高比和宽度
            return type('Barcode', (), {'rect': type('Rect', (), {'left': x, 'top': y, 'width': w, 'height': h})})()
    return None

def detect_white_panel(roi_color):
    """检测ROI是否包含白色面板"""
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    white_area = cv2.countNonZero(mask)
    total_area = roi_color.shape[0] * roi_color.shape[1]
    return white_area / total_area > 0.1

def read_digits_from_image(image, psm=6, verbose=False, log_file=None):
    """
    使用Tesseract OCR读取图像中的数字。

    参数:
    - image: 预处理后的二值化图像
    - psm: Tesseract页面分割模式，6表示单行文本
    - verbose: 是否记录详细日志
    - log_file: 日志文件对象

    返回:
    - 合法的数字字符串，否则返回None
    """
    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config)
    cleaned_text = re.sub(r'[^\d]', '', text)
    if verbose and log_file:
        log_file.write(f"Raw OCR result (Tesseract, psm={psm}): {cleaned_text}\n")
        log_file.flush()
    if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith("5001"):
        cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
        if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in range(5201, 5247):
            return cleaned_text
    return None

def read_digits_with_easyocr(image, verbose=False, log_file=None):
    """
    使用EasyOCR读取图像中的数字。

    参数:
    - image: 预处理后的二值化图像
    - verbose: 是否记录详细日志
    - log_file: 日志文件对象

    返回:
    - 合法的数字字符串，否则返回None
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    try:
        result = reader.readtext(image_bgr, detail=0, allowlist='0123456789')
        text = ''.join(result)
        cleaned_text = re.sub(r'[^\d]', '', text)
        if verbose and log_file:
            log_file.write(f"Raw OCR result (EasyOCR): {cleaned_text}\n")
            log_file.flush()
        if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith("5001"):
            cleaned_text = cleaned_text[:16] if len(cleaned_text) > 16 else cleaned_text.zfill(16)
            if cleaned_text[8:12] in ["0000", "0001"] and int(cleaned_text[-4:]) in range(5201, 5247):
                return cleaned_text
    except Exception as e:
        if verbose and log_file:
            log_file.write(f"EasyOCR recognition failed: {str(e)}\n")
            log_file.flush()
    return None

def tune_digit_extraction_params(roi_gray, roi_color, roi_x, roi_y, gray_image, cached_params=None, verbose=False, log_file=None):
    """
    调优数字提取参数或使用缓存参数。

    参数:
    - roi_gray: ROI灰度图像
    - roi_color: ROI彩色图像
    - roi_x, roi_y: ROI在原图中的左上角坐标
    - gray_image: 原灰度图像
    - cached_params: 缓存的最佳参数
    - verbose: 是否记录详细日志
    - log_file: 日志文件对象

    返回:
    - digit_roi: 提取的数字区域
    - best_params: 最佳参数
    - digit: 识别出的数字，若失败则为None
    """
    if cached_params:
        digit_roi = extract_digit_roi(roi_gray, roi_color, roi_x, roi_y, gray_image)
        if digit_roi is not None:
            digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=cached_params['denoise_h'], clipLimit=cached_params['clipLimit'])
            digit = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file) or \
                    read_digits_from_image(digit_roi_processed, psm=cached_params['psm'], verbose=verbose, log_file=log_file)
            return digit_roi, cached_params, digit
        return None, cached_params, None

    # 仅对首个ROI进行参数调优
    denoise_h_values = [10, 15, 20, 25]
    clipLimit_values = [2.0, 3.0, 4.0]
    psm_values = [6, 7, 8]
    best_params = {'denoise_h': 20, 'clipLimit': 2.0, 'psm': 6}
    digit_roi = extract_digit_roi(roi_gray, roi_color, roi_x, roi_y, gray_image)
    if digit_roi is not None:
        for denoise_h in denoise_h_values:
            for clipLimit in clipLimit_values:
                digit_roi_processed = preprocess_for_digits(digit_roi, denoise_h=denoise_h, clipLimit=clipLimit)
                for psm in psm_values:
                    digit = read_digits_with_easyocr(digit_roi_processed, verbose=verbose, log_file=log_file) or \
                            read_digits_from_image(digit_roi_processed, psm=psm, verbose=verbose, log_file=log_file)
                    if digit:
                        best_params.update({'denoise_h': denoise_h, 'clipLimit': clipLimit, 'psm': psm})
                        return digit_roi, best_params, digit
    return digit_roi, best_params, None

def extract_digit_roi(roi_gray, roi_color, roi_x, roi_y, gray_image):
    """
    从ROI中提取数字区域，利用条形码定位。

    参数:
    - roi_gray: ROI灰度图像
    - roi_color: ROI彩色图像
    - roi_x, roi_y: ROI在原图中的左上角坐标
    - gray_image: 原灰度图像

    返回:
    - 提取的数字区域，若失败则为None
    """
    barcode = detect_barcode(roi_color)
    if barcode:
        rect = barcode.rect
        digit_roi_y = roi_y + rect.top - int(rect.height * 1.2)  # 数字在条形码上方，预留1.2倍高度
        digit_roi_h = rect.height
        digit_roi_x = roi_x + rect.left
        digit_roi_w = rect.width
        if digit_roi_y >= 0 and digit_roi_y + digit_roi_h <= gray_image.shape[0] and digit_roi_x + digit_roi_w <= gray_image.shape[1]:
            return gray_image[digit_roi_y:digit_roi_y + digit_roi_h, digit_roi_x:digit_roi_x + digit_roi_w]
    return None  # 若无条形码，默认不提取数字区域

def process_image_for_dpm_barcodes(image_path, debug_visualization=False, manual_rois=None, denoise_h=20, psm=6, verbose=True, auto_close=False, log_file=None):
    """
    处理图像以检测OBU区域并识别数字。

    参数:
    - image_path: 输入图像路径
    - debug_visualization: 是否启用调试可视化
    - manual_rois: 手动ROI列表
    - denoise_h: 初始去噪强度
    - psm: 初始Tesseract PSM模式
    - verbose: 是否记录详细日志
    - auto_close: 是否自动关闭调试窗口
    - log_file: 日志文件对象

    返回:
    - 识别出的数字列表
    - 处理耗时
    """
    start_time = time.time()
    all_valid_digits = set()
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Unable to load image {image_path}")
        return [], 0

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if debug_visualization:
        save_debug_image(original_image, "Original")

    # ROI检测（手动或自动）
    if not manual_rois:
        contours_to_process = [cv2.boundingRect(c) for c in cv2.findContours(cv2.Canny(gray_image, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
    else:
        contours_to_process = [(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]) for roi in manual_rois]

    roi_counter = 0
    cached_params = None
    for contour in contours_to_process:
        roi_x, roi_y, roi_w, roi_h = contour
        if roi_w <= 20 or roi_h <= 20:
            continue
        roi_counter += 1
        obu_roi_gray = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        obu_roi_color = original_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if not detect_white_panel(obu_roi_color):
            continue

        digit_roi, cached_params, digit = tune_digit_extraction_params(
            obu_roi_gray, obu_roi_color, roi_x, roi_y, gray_image, cached_params=cached_params, verbose=verbose, log_file=log_file
        )
        if digit:
            all_valid_digits.add(digit)
        if debug_visualization and digit_roi is not None:
            save_debug_image(digit_roi, f"ROI_{roi_counter}_Digit_Area")

    elapsed_time = time.time() - start_time
    if debug_visualization:
        print(f"Processed {roi_counter} candidate OBU regions.")
        print(f"Current recognized valid digits (set): {all_valid_digits}")
    return sorted(list(all_valid_digits)), elapsed_time

if __name__ == "__main__":
    clear_process_images()
    parser = argparse.ArgumentParser(description="OBU Image Processing for Digit Recognition")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--manual-rois", type=str, help="Path to manual ROIs JSON file")
    parser.add_argument("--denoise_h", type=int, default=20, help="Initial denoising strength")
    parser.add_argument("--psm", type=int, default=6, help="Initial Tesseract PSM mode")
    parser.add_argument("--log", action="store_true", help="Enable logging to file")
    parser.add_argument("--auto-close", action="store_true", help="Auto close debug windows")
    args = parser.parse_args()

    manual_rois = None
    if args.manual_rois:
        with open(args.manual_rois, 'r', encoding='utf-8') as f:
            manual_rois = json.load(f)

    log_file = setup_logging(enable_log=args.log)
    digits, elapsed_time = process_image_for_dpm_barcodes(
        args.image_path, debug_visualization=args.debug, manual_rois=manual_rois,
        denoise_h=args.denoise_h, psm=args.psm, verbose=True, auto_close=args.auto_close, log_file=log_file
    )
    print(f"\nFinal digit recognition results: {digits}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    if log_file:
        log_file.close()