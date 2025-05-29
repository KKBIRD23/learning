import easyocr
import cv2
import numpy as np

# 加载图像
image_path = r'C:\Users\KKBIRD\Desktop\photo\111.png'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 可选：预处理图像（灰度、增强对比度等）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 可选增强（可根据需要打开）
# gray = cv2.equalizeHist(gray)

# 保存临时处理图像（EasyOCR 读取图像路径或numpy数组）
temp_path = 'temp_processed.png'
cv2.imwrite(temp_path, gray)

# 初始化 easyocr Reader（只使用英文和数字）
reader = easyocr.Reader(['en'], gpu=False)

# 执行OCR
results = reader.readtext(temp_path)

# 只提取数字
digit_results = []
for bbox, text, conf in results:
    digits_only = ''.join(filter(str.isdigit, text))
    if digits_only:
        digit_results.append((bbox, digits_only, conf))

# 输出结果
for bbox, digits, conf in digit_results:
    print(f"识别数字: {digits}，置信度: {conf:.2f}，位置: {bbox}")
