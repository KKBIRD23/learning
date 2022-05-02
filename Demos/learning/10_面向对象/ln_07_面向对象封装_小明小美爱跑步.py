"""
小明 和 小美 都爱跑步
需求：
1. 小明 体重 75.0 公斤
   小美 体重 45.0 公斤
2. 小明每次 跑步 会减肥 0.5 公斤
3. 小明每次 吃东西 体重会增加 1 公斤
-------------------------------
需求分析：
类：Person
名字：name
体重：weight
方法：
    __init__(self, name, weight):
    __str__(self):
    run(self):
    eat(self):
"""


class Person:
    def __init__(self, name, weight):
        # self.属性 = 形参
        self.name = name
        self.weight = weight

    def __str__(self):
        return f"我的名字叫{self.name},我的体重是{self.weight:.2f}公斤！"

    def run(self):
        print(f"{self.name} 爱跑步，跑步锻炼身体")
        self.weight -= 0.5

    def eat(self):
        print(f"{self.name} 是吃货，吃完这顿再减肥！")
        self.weight += 1


xiaoming = Person("小明", 75.0)

xiaoming.run()
xiaoming.eat()
print(xiaoming)

# 小美爱跑步
xiaomei = Person("小美", 45.0)
xiaomei.eat()
xiaomei.run()
print(xiaomei)
