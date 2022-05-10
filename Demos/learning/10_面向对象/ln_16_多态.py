"""
需求：
1. 在 Dog 类中封装方法 game ，普通的够只能简单的玩耍
2. 在 XiaoTianQuan 类中继承 Dog 类，并且重写 game 方法，哮天犬可以上天
3. 定义 Person 类， 并且封装一个和 狗对象 玩耍的方法
"""


class Dog(object):
    def __init__(self, name):
        self.name = name

    def game(self):
        print(f"{self.name}开心的汪汪汪~~")


class XiaoTianQuan(Dog):

    def game(self):
        print(f'{self.name}飞到天上去开心的汪汪汪！！！')


class Person(object):
    def __init__(self, name):
        self.name = name

    def game_with_dog(self, dog):
        print(f'{self.name}和{dog.name}快乐的玩耍')

        dog.game()


# 1. 创建一个狗对象
# wangcai = Dog("旺财")
wangcai = XiaoTianQuan("旺财")
# 2. 创建一个小明对象
xiaoming = Person("小名")
# 3. 让小明和狗玩
xiaoming.game_with_dog(wangcai)
