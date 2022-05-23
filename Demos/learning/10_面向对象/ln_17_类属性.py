"""
需求：
定义一个 工具 类
每个工具都有自己的 name
需求—— 知道使用这个累，创建了多少个工具对象？
"""


class Tools(object):
    # 使用赋值语句，定义类属性，记录创造工具对象的总数
    count = 0

    def __init__(self, name):
        self.name = name

        Tools.count += 1


tool1 = Tools("斧头")

tool2 = Tools("榔头")

# 输出工具对象的总数
print(Tools.count)
