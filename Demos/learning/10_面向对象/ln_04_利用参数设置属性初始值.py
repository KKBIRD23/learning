class Cat:
    def __init__(self, new_name):
        print("这是一个初始化方法，会被自动调用")
        self.name = new_name

    def eat(self):
        print(f'{self.name} 爱吃鱼')


tom = Cat("Tom")

print(tom.name)

lazy_cat = Cat("大懒猫")
lazy_cat.eat()
