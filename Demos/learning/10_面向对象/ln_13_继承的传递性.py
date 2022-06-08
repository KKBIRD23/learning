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

    # 如果父类的方法不能满足需要，可以在子类中重写方法
    # 在掉用时将调用子类而不是父类方法
    def bark(self):
        print("嘿嘿嘿")


# 创建哮天犬对象
xtq = XiaoTianQuan()

xtq.fly()
xtq.bark()
xtq.eat()
