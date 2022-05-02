class Cat:

    def eat(self):
        # 哪个对象调用的方法，self参数就是哪一个对象的引用
        print("小猫爱吃鱼")
        print(self.name)    # 旧版本Pycharm才能这么干

    def drink(self):
        print("小猫要喝水")
        print(self.name)    # 旧版本Pycharm才能这么干


# 创建猫对象
tom = Cat()

# 使用 .属性名 利用赋值语句增加属性(不推荐的办法!!!)
tom.name = "汤姆"

tom.eat()
tom.drink()

print(tom)
# 在创建一个猫对象
lazy_cat = Cat()

lazy_cat.name = "大懒猫"

lazy_cat.eat()
lazy_cat.drink()
print(lazy_cat)
