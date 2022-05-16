"""
设计模式:前人工作的总结和提炼，通常被人们广泛流传的设计模式都是针对某一特定问题的成熟解决方案
单例设计模式：即单实例设计模式。应用场景如：音乐播放器
    - 目的 —— 让 类 创建的对象，在系统中 只有 唯一的一个实例
    - 每一次执行 类名() 返回的对象，内存地址是相同的
----------------------------------------------------------------------------
__new__方法的重写格式是固定的：
def __new__(cls, *args, **kwargs):
    新代码
    return super().__new__(cls)
"""


class MusicPlayer(object):

    def __new__(cls, *args, **kwargs):
        # 1. 创建对象时，new方法会被自动调用
        print("创建对象，分配空间")
        # 2. 为对象分配空间,返回对象的引用
        return super().__new__(cls)

    def __init__(self):
        print("播放器初始化")


player = MusicPlayer()

print(player)
