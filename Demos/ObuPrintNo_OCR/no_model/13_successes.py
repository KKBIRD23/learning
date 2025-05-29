import cv2
import numpy as np
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
import pytesseract
import argparse
import re
import os
import random
import json

# 自动优化模式：python 13_successes.py 3.jpg --debug --skip-perspective --optimize --manual-rois rois.json
# 手动单次测试：python 13_successes.py.py 3.jpg --debug --skip-perspective --manual-rois rois.json


# 配置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def read_digits_from_image(image, psm=6):
    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config)
    cleaned_text = re.sub(r'[^\d]', '', text)
    if 15 <= len(cleaned_text) <= 17 and cleaned_text.startswith("5001"):
        cleaned_text = cleaned_text[:16]
    return cleaned_text

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

def preprocess_for_code128(image, denoise_h=10):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=denoise_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    refined = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    return refined

def process_image_for_dpm_barcodes(image_path, debug_visualization=False, skip_perspective=True, manual_rois=None,
                                 upper_black_v=50, canny_low=50, canny_high=150, clahe_clip=2.0, kernel_size=5,
                                 denoise_h=10, psm=6, verbose=True, auto_close=False):
    all_valid_barcodes = set()
    all_valid_digits = set()
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"错误：无法加载图像 {image_path}")
        return [], []

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image = sharpen_image(gray_image)

    if debug_visualization:
        cv2.imshow("0. Original", cv2.resize(original_image, (0,0), fx=0.15, fy=0.15))
        cv2.imshow("1. Grayscale (Sharpened)", cv2.resize(gray_image, (0,0), fx=0.15, fy=0.15))

    if verbose:
        print("尝试在原始图像上识别：")
    barcodes = pyzbar.decode(gray_image, symbols=[ZBarSymbol.CODE128])
    digits = read_digits_from_image(gray_image, psm)
    if verbose:
        print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
        print(f"数字识别结果: {digits}")
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        if len(barcode_data) == 16 and barcode_data.startswith("5001"):
            all_valid_barcodes.add(barcode_data)
    if digits and len(digits) == 16 and digits.startswith("5001"):
        all_valid_digits.add(digits)

    code128_image = preprocess_for_code128(gray_image, denoise_h)
    if debug_visualization:
        cv2.imshow("1a. Code128 Preprocessed", cv2.resize(code128_image, (0,0), fx=0.15, fy=0.15))
        cv2.imwrite("code128_global.jpg", code128_image)
    if verbose:
        print("尝试在 Code128 预处理图像上识别：")
    barcodes = pyzbar.decode(code128_image, symbols=[ZBarSymbol.CODE128])
    digits = read_digits_from_image(code128_image, psm)
    if verbose:
        print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
        print(f"数字识别结果: {digits}")
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        if len(barcode_data) == 16 and barcode_data.startswith("5001"):
            all_valid_barcodes.add(barcode_data)
    if digits and len(digits) == 16 and digits.startswith("5001"):
        all_valid_digits.add(digits)

    # OBU 分割
    if not manual_rois:
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        upper_black = np.array([180, 255, upper_black_v])
        mask = cv2.inRange(hsv_image, np.array([0, 0, 0]), upper_black)
        edges = cv2.Canny(gray_image, canny_low, canny_high)
        combined = cv2.bitwise_and(edges, edges, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        obu_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        obu_contours = []

    if debug_visualization and not manual_rois:
        debug_obu_contours_img = original_image.copy()
        cv2.drawContours(debug_obu_contours_img, obu_contours, -1, (0,0,255), 2)
        cv2.imshow("2. OBU Contours (HSV + Canny)", cv2.resize(debug_obu_contours_img, (0,0), fx=0.15, fy=0.15))
        cv2.imwrite("obu_contours.jpg", debug_obu_contours_img)

    roi_counter = 0
    contours_to_process = manual_rois if manual_rois else obu_contours

    min_obu_area = int(0.001 * gray_image.shape[0] * gray_image.shape[1])
    max_obu_area = int(0.1 * gray_image.shape[0] * gray_image.shape[1])

    for contour in contours_to_process:
        if manual_rois:
            x, y, x2, y2 = contour
            w, h = int(x2 - x), int(y2 - y)
        else:
            area = cv2.contourArea(contour)
            if not (min_obu_area < area < max_obu_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
        padding = 20
        roi_x = int(max(0, x - padding))
        roi_y = int(max(0, y - padding))
        roi_w = int(min(gray_image.shape[1] - roi_x, w + 2 * padding))
        roi_h = int(min(gray_image.shape[0] - roi_y, h + 2 * padding))

        if roi_w <= 20 or roi_h <= 20:
            continue

        obu_roi_gray = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        roi_counter += 1

        if debug_visualization:
            cv2.imshow(f"3. ROI {roi_counter} Original", obu_roi_gray)
            cv2.imwrite(f"roi_{roi_counter}_original.jpg", obu_roi_gray)

        if verbose:
            print(f"\nROI {roi_counter} 原始图像：")
        barcodes = pyzbar.decode(obu_roi_gray, symbols=[ZBarSymbol.CODE128])
        digits = read_digits_from_image(obu_roi_gray, psm)
        if verbose:
            print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
            print(f"数字识别结果: {digits}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            if len(barcode_data) == 16 and barcode_data.startswith("5001"):
                all_valid_barcodes.add(barcode_data)
        if digits and len(digits) == 16 and digits.startswith("5001"):
            all_valid_digits.add(digits)

        code128_roi = preprocess_for_code128(obu_roi_gray, denoise_h)
        if debug_visualization:
            cv2.imshow(f"3a. ROI {roi_counter} Code128 Preprocessed", code128_roi)
            cv2.imwrite(f"roi_{roi_counter}_code128_enhanced.jpg", code128_roi)
        if verbose:
            print(f"ROI {roi_counter} Code128 预处理图像（请检查 roi_{roi_counter}_code128_enhanced.jpg）：")
        barcodes = pyzbar.decode(code128_roi, symbols=[ZBarSymbol.CODE128])
        digits = read_digits_from_image(code128_roi, psm)
        if verbose:
            print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
            print(f"数字识别结果: {digits}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            if len(barcode_data) == 16 and barcode_data.startswith("5001"):
                all_valid_barcodes.add(barcode_data)
        if digits and len(digits) == 16 and digits.startswith("5001"):
            all_valid_digits.add(digits)

        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        enhanced_roi = clahe.apply(obu_roi_gray)
        enhanced_roi = cv2.GaussianBlur(enhanced_roi, (3, 3), 0)

        if debug_visualization:
            cv2.imshow(f"4. ROI {roi_counter} CLAHE", enhanced_roi)
            cv2.imwrite(f"roi_{roi_counter}_clahe.jpg", enhanced_roi)
        if verbose:
            print(f"ROI {roi_counter} CLAHE图像：")
        barcodes = pyzbar.decode(enhanced_roi, symbols=[ZBarSymbol.CODE128])
        digits = read_digits_from_image(enhanced_roi, psm)
        if verbose:
            print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
            print(f"数字识别结果: {digits}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            if len(barcode_data) == 16 and barcode_data.startswith("5001"):
                all_valid_barcodes.add(barcode_data)
        if digits and len(digits) == 16 and digits.startswith("5001"):
            all_valid_digits.add(digits)

        _, roi_otsu_thresh = cv2.threshold(enhanced_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug_visualization:
            cv2.imshow(f"5. ROI {roi_counter} Otsu", roi_otsu_thresh)
            cv2.imwrite(f"roi_{roi_counter}_otsu.jpg", roi_otsu_thresh)
        if verbose:
            print(f"ROI {roi_counter} Otsu图像：")
        barcodes = pyzbar.decode(roi_otsu_thresh, symbols=[ZBarSymbol.CODE128])
        digits = read_digits_from_image(roi_otsu_thresh, psm)
        if verbose:
            print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
            print(f"数字识别结果: {digits}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            if len(barcode_data) == 16 and barcode_data.startswith("5001"):
                all_valid_barcodes.add(barcode_data)
        if digits and len(digits) == 16 and digits.startswith("5001"):
            all_valid_digits.add(digits)

        roi_adaptive_thresh = cv2.adaptiveThreshold(enhanced_roi, 255,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, blockSize=15, C=4)
        if debug_visualization:
            cv2.imshow(f"6. ROI {roi_counter} Adaptive", roi_adaptive_thresh)
            cv2.imwrite(f"roi_{roi_counter}_adaptive.jpg", roi_adaptive_thresh)
        if verbose:
            print(f"ROI {roi_counter} Adaptive图像：")
        barcodes = pyzbar.decode(roi_adaptive_thresh, symbols=[ZBarSymbol.CODE128])
        digits = read_digits_from_image(roi_adaptive_thresh, psm)
        if verbose:
            print(f"条码识别结果: {[b.data.decode('utf-8') for b in barcodes]}")
            print(f"数字识别结果: {digits}")
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            if len(barcode_data) == 16 and barcode_data.startswith("5001"):
                all_valid_barcodes.add(barcode_data)
        if digits and len(digits) == 16 and digits.startswith("5001"):
            all_valid_digits.add(digits)

    if debug_visualization:
        print(f"\n总共处理了 {roi_counter} 个候选OBU区域。")
        print(f"当前识别到的有效条码 (集合): {all_valid_barcodes}")
        print(f"当前识别到的有效数字 (集合): {all_valid_digits}")
        if not auto_close:
            print("按任意键关闭所有调试窗口...")
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return sorted(list(all_valid_barcodes)), sorted(list(all_valid_digits))

def optimize_parameters(image_path, debug_visualization=False, manual_rois=None):
    # 初始参数（基于成功版本）
    best_params = {'upper_black_v': 50, 'canny_low': 50, 'canny_high': 150, 'clahe_clip': 2.0,
                   'kernel_size': 5, 'denoise_h': 10, 'psm': 6}
    history = []

    print("Testing initial params:", best_params)
    barcodes, digits = process_image_for_dpm_barcodes(
        image_path, debug_visualization, True, manual_rois,
        upper_black_v=best_params['upper_black_v'], canny_low=best_params['canny_low'],
        canny_high=best_params['canny_high'], clahe_clip=best_params['clahe_clip'],
        kernel_size=best_params['kernel_size'], denoise_h=best_params['denoise_h'],
        psm=best_params['psm'], verbose=False, auto_close=True
    )
    best_score = len(barcodes) + len(digits)
    history.append({'params': best_params.copy(), 'score': best_score, 'barcodes': barcodes, 'digits': digits})
    print(f"Initial score: {best_score}, Digits: {digits}")

    # 扩展参数范围
    param_ranges = {
        'upper_black_v': [30, 40, 50, 60, 70, 80],
        'canny_low': [30, 40, 50, 60, 70],
        'canny_high': [120, 135, 150, 165, 180],
        'clahe_clip': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        'kernel_size': [3, 5, 7, 9],
        'denoise_h': [5, 8, 10, 12, 15],
        'psm': [3, 4, 6, 7]
    }

    no_improve_count = {k: 0 for k in param_ranges.keys()}
    fixed_params = set()

    params_to_tune = ['upper_black_v', 'canny_low', 'canny_high', 'clahe_clip', 'kernel_size', 'denoise_h', 'psm']
    for param in params_to_tune:
        if param in fixed_params:
            continue

        print(f"\nTuning parameter: {param}")
        initial_value = best_params[param]
        values_to_test = param_ranges[param]
        idx = values_to_test.index(initial_value) if initial_value in values_to_test else -1
        if idx == -1:
            idx = values_to_test.index(min(values_to_test, key=lambda x: abs(x - initial_value)))
        lower_values = values_to_test[:idx][::-1]
        upper_values = values_to_test[idx + 1:]
        test_values = lower_values + upper_values

        for value in test_values:
            if param in fixed_params:
                break

            current_params = best_params.copy()
            current_params[param] = value
            print(f"Testing params: {current_params}")
            barcodes, digits = process_image_for_dpm_barcodes(
                image_path, debug_visualization, True, manual_rois,
                upper_black_v=current_params['upper_black_v'], canny_low=current_params['canny_low'],
                canny_high=current_params['canny_high'], clahe_clip=current_params['clahe_clip'],
                kernel_size=current_params['kernel_size'], denoise_h=current_params['denoise_h'],
                psm=current_params['psm'], verbose=False, auto_close=True
            )
            score = len(barcodes) + len(digits)
            history.append({'params': current_params.copy(), 'score': score, 'barcodes': barcodes, 'digits': digits})
            print(f"Score for current params: {score}, Digits: {digits}")

            if score > best_score:
                best_score = score
                best_params = current_params
                no_improve_count[param] = 0
                print(f"New best score: {best_score}, New best params: {best_params}")
            else:
                no_improve_count[param] += 1
                print(f"No improvement for {param}, count: {no_improve_count[param]}")

            if no_improve_count[param] >= 5:
                print(f"Parameter {param} has no improvement after 5 attempts, fixing at {best_params[param]}")
                fixed_params.add(param)
                break

    print("\nSummary of all tests:")
    for i, entry in enumerate(history):
        print(f"Test {i + 1}: Params: {entry['params']}, Score: {entry['score']}, Digits: {entry['digits']}")

    print(f"\nOptimal parameters: {best_params}, Best score: {best_score}")
    return best_params, best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="针对DPM条码和斜拍图像的高级条码识别脚本。")
    parser.add_argument("image_path", type=str, help="图像文件路径。")
    parser.add_argument("--debug", action="store_true", help="启用调试可视化窗口。")
    parser.add_argument("--skip-perspective", action="store_true", help="跳过透视校正。")
    parser.add_argument("--optimize", action="store_true", help="运行参数优化。")
    parser.add_argument("--manual-rois", type=str, help="手动 ROI 坐标文件路径 (JSON 格式)")
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
            print(f"加载 ROI 文件失败: {e}")
            manual_rois = None

    if args.optimize:
        best_params, best_score = optimize_parameters(args.image_path, args.debug, manual_rois)
        print(f"Optimal parameters: {best_params}, Best score: {best_score}")
    else:
        final_barcodes, final_digits = process_image_for_dpm_barcodes(args.image_path, args.debug, args.skip_perspective, manual_rois)
        print("\n最终识别结果：")
        print("Barcodes:", final_barcodes)
        print("Digits:", final_digits)