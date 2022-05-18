"""
定义一个类属性`init_flag`标记是否 执行过初始化动作，初始值设置为`False`
在`__init__`方法中，判断init_flag，如果为False就执行初始化动作，然后将值设置为True
"""


class MusicPlayer(object):
    # 定义一个类属性，记录第一个被创建对象的引用
    instance = None
    # 记录是否执行过初始化动作
    init_flag = False

    def __new__(cls, *args, **kwargs):
        # 1. 判断类属性是否为空对象
        if cls.instance is None:
            # 2. 调用父类的方法，为第一个对象分配空间
            cls.instance = super().__new__(cls)

        # 3. 返回类属性保存的对象引用
        return cls.instance

    def __init__(self):
        # 1. 判断是否执行过初始化动作
        if MusicPlayer.init_flag:
            return
        # 2. 如果没有执行过，则执行初始化
        MusicPlayer.init_flag = True


player1 = MusicPlayer()
print(player1)

player2 = MusicPlayer()
print(player2)
