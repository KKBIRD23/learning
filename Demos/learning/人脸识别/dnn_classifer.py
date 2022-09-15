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
net = dnn.readNetFromCaffe(config, model)

# 读取图片,转化为张量
img = cv2.imread("D:\WorkSpaces\Python\WorkSpaces\Demos\识别多个条码\obu_img\ATS01.jpg")
blob = dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123))

# 将张量输入到神经网络,并进行预测
net.setInput(blob)
r = net.forward()

# 读入类目文件
classes = []
path = "./model/synset_words.txt"
with open(path, "rt") as f:
    classes = [x[x.find(" ") + 1:] for x in f]

order = sorted(r[0], reverse=True)
z = list(range(3))
for i in range(0, 3):
    z[i] = np.where(r[0] == order[i])[0][0]
    print('第', i + 1, '项,匹配', classes[z[i]], end='')
    print('类所在行:', z[i] + 1, ' ', '可能性', order[i])
# 得到结果,并显示
