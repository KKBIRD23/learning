from sys import exit
import numpy as np
import cv2

# 加载图片并把它转换为灰度图片
image = cv2.imread('F:/work/barcode/bar_code/8.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_Image", gray)
cv2.waitKey(0)
'''
# 索贝尔算子（Sobel operator）主要用作边缘检测，在技术上，它是一离散性差分算子，用来运算图像亮度函数的灰度之近似值。在图像的任何一点使用此算子，将会产生对应的灰度矢量或是其法矢量
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
# x灰度减去y灰度
gradient = cv2.subtract(gradX, gradY)
# 转回uint8
gradient = cv2.convertScaleAbs(gradient)  
absX = cv2.convertScaleAbs(gradX)   # 转回uint8  
absY = cv2.convertScaleAbs(gradY)    
gradient = cv2.addWeighted(absX,0.5,absY,0.5,0) # 组合
'''
gradient = cv2.Canny(gray, 50, 400)
cv2.imshow("S/C_Image", gradient)
cv2.waitKey(0)

(_, thresh) = cv2.threshold(gradient, 250, 255, cv2.THRESH_BINARY)  # 二值化
cv2.imshow("threshold_Image", thresh)
cv2.waitKey(0)
# 构建kernel然后应用到 thresholded 图像上
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))  # 形态学处理，定义矩形结构
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)#闭运算，先膨胀后腐蚀
# closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)#开运算，先腐蚀后膨胀
# closed = cv2.erode(thresh, kernel, iterations = 1)#腐蚀图像，去除噪声点
closed = cv2.dilate(thresh, kernel, iterations=1)  # 膨胀图像，连接断点
cv2.imshow("dilate_Image", closed)
cv2.waitKey(0)
# 找到条码轮廓
# 保留大的区域，有时会有没过滤掉的干扰
im, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# 获取轮廓数量
x = len(contours)
s = []
for i in range(0, x):
    s.append(cv2.contourArea(contours[i]))
# 打印面积
for j in s:
    print
    "s was : %f", j

# 筛选出面积大于等于8000的轮廓
for k in range(0, x):
    if s[k] >= 8000.0:
        rect = cv2.minAreaRect(contours[k])  # 返回矩形的中心点坐标，长宽，旋转角度
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (255, 0, 0), 2)  # 画一个方框把条形码区域圈起来
    else:
        continue

cv2.imshow("Image", image)
cv2.waitKey(0)
exit(0)
