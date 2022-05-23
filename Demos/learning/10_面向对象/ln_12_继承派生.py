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


wangcai = Dog()

wangcai.eat()
wangcai.drink()
wangcai.run()
wangcai.sleep()
