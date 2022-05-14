"""
定义一个工具类
每件工具都有自己的 name
在 类 中封装一个show_tool_count 的类方法，输出使用当前这个类创建的对象个数
"""


class Tools(object):
    count = 0

    @classmethod
    def show_tool_count(cls):
        print(f'当前使用这个类的对象个数为 {Tools.count}')

    def __init__(self, name):
        self.name = name

        Tools.count += 1


tool1 = Tools("斧头")
tool2 = Tools("榔头")

# 调用类方法
Tools.show_tool_count()
