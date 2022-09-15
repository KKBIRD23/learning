import cv2
from cv2 import dnn
import numpy as np

"""
1.导入模型,创建神经网络
2.读取图片,转化为张量
3.将张量输入到神经网络,并进行预测
4.得到结果,并显示
"""

# 导入模型,创建神经网络
config = "./model/bvlc_googlenet.prototxt"
model = "./model/bvlc_googlenet.caffemodel"

cv2.dnn.readNetFromCaffer(config, model)

