"""
在属性名或者方法名前加两个下划线__，可以设置属性和方法为私有
事实上Python中是没有严格意义上的私有属性和方法的，当定义一个私有属性或方法时
Python只是在其名称前加上了“_类名” 即 _类名_名称
其实可以通过这个方式对其进行访问 但 强烈提示不要使用这种方式访问私有对象！
"""


class Women:
    def __init__(self, name):
        self.name = name
        self.__age = 18

    def secret(self):
        print(f'{self.name}的年龄是{self.__age}')


xiaofang = Women("小芳")

# 私有属性在方法外部是无法调用的，所以会报错
print(xiaofang.__age)   # 报错
print(xiaofang._Women__age) # 强行访问
# 但是通过对象内部的方法可以访问到私有属性
xiaofang.secret()
