"""
在属性名或者方法名前加两个下划线__，可以设置属性和方法为私有
小芳的年龄是18岁
"""


class Women:
    def __init__(self, name):
        self.name = name
        self.__age = 18

    def secret(self):
        print(f'{self.name}的年龄是{self.__age}')


xiaofang = Women("小芳")

# 私有属性在方法外部是无法调用的，所以会报错
print(xiaofang.__age)

# 但是通过对象内部的方法可以访问到私有属性
xiaofang.secret()
