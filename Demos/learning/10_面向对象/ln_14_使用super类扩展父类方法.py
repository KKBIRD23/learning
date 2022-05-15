class Animal:
    def eat(self):
        print("吃")

    def drink(self):
        print("喝")

    def run(self):
        print("跑")

    def sleep(self):
        print("睡")


# 使用继承后，子类天生具有父类的所有属性和方法
# 继承的语法
# class 类名(父类名)
class Dog(Animal):
    def bark(self):
        print("汪汪汪！")


class XiaoTianQuan(Dog):
    def fly(self):
        print("我会飞")

    def bark(self):
        # 使用super扩展
        # 1. 针对子类的特有需求，编写代码
        print("嘿嘿嘿")
        # 2. 使用super() 调用原本在父类中封装的方法
        super().bark()
        # 3. 增加其他子类代码
        print("哈哈哈")


# 创建哮天犬对象
xtq = XiaoTianQuan()

xtq.fly()
xtq.bark()
xtq.eat()
