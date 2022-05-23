"""
需求：
- 小猫爱吃鱼，小猫要喝水

需求分析：
1. 定义个猫类 Cat(类名为大驼峰)
2. 定义两个方法 eat 和 drink
3. 按照需求，不需要定义属性
"""


class Cat:

    def eat(self):
        print("小猫爱吃鱼")

    def drink(self):
        print("小猫要喝水")


# 创建猫对象
tom = Cat()

tom.eat()
tom.drink()
print(tom)
