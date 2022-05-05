from sys import exit
# from Image import _ImageCrop
from PIL import Image
import numpy as np
import pyzbar as zbar
import cv2

# 加载图片并把它转换为灰度图片
image = cv2.imread('F:/work/barcode/bar_code/20.jpg')
img = Image.open(r'F:/work/barcode/bar_code/20.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("sobel_Image", gray)
# cv2.waitKey(0)
# 使用Canny做边缘检测
gradient = cv2.Canny(gray, 20, 520)
# cv2.imshow("Canny_Image", gradient)
# cv2.waitKey(0)

(_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)  # 二值化
cv2.imshow("threshold_Image", thresh)
# cv2.waitKey(0)
# 构建kernel然后应用到 thresholded 图像上
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))  # 形态学处理，定义矩形结构
closed = cv2.dilate(thresh, kernel, iterations=1)  # 膨胀图像，连接断点
# cv2.imshow("dilate_Image", closed)
# cv2.waitKey(0)

im, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print contours
x = len(contours)
a = []
s = []

# 打印面积
for i in range(0, x):
    s.append(cv2.contourArea(contours[i]))

# 保留面积大于8000的轮廓
for m in range(0, x):
    if 8000 <= s[m] <= 25000:
        a.append(s[m])
    else:
        continue

z = max(a)

# for j in a:
#    print "a was : %f",j

for k in range(0, x):
    # 增加一些筛选条件
    if 8000 <= s[k] <= 25000 and ((z - s[k]) <= 8500):
        rect = cv2.minAreaRect(contours[k])  # 返回矩形的中心点坐标，长宽，旋转角度
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (255, 0, 0), 2)  # 画一个方框把条形码区域圈起来

        u, v, w, t = cv2.boundingRect(contours[k])  # 获取轮廓坐标
        # print u,v,w,t
        # 根据坐标把条码裁剪下来并保存
        o = (u, v, u + w, v + t)
        barcode = img.crop(o)
        barcode.save(r'F:/work/barcode/bar_code/crop4.jpg')
        # print "s : %f",s[k]
        # 构建解码器
        scanner = zbar.ImageScanner()
        scanner.parse_config('enable')
        pil = Image.open('F:/work/barcode/bar_code/crop4.jpg').convert('L')
        width, height = pil.size
        # 解码
        raw = pil.tostring()
        image0 = zbar.Image(width, height, 'Y800', raw)
        scanner.scan(image0)
        for symbol in image0:
            print
            'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
    else:
        continue

cv2.imshow("Image", image)
cv2.waitKey(0)
exit(0)