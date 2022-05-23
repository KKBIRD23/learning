"""
需求：
1. 设计一个 Game 类
2. 属性：
    - 定义一个 类属性 `top_score` 记录游戏的 历史最高分
    - 定义一个 实例属性 `player_name` 记录 当前游戏玩家的姓名
3. 方法：
    - 静态方法 `show_help` 显示游戏帮助
    - 类方法 `show_top_score` 显示历史最高分
    - 实例方法 `start_game` 开始当前玩家的游戏
4. 主程序步骤
    - 查看帮助信息
    - 查看历史最高分
    - 创建游戏对象，开始游戏
-------------------------------------------------------------------------
总述：`Python`中 一切皆对象！类 也是一个特殊的对象
类属性：在类中直接使用复制语句定义的和对象无关的属性
类方法：在类中使用 `@classmethod`修饰符，在下一行中使用`def 类名(cls)`
实力属性：在类中使用`__init__`初始化方法定义 `def __init__(self,属性1,属性2...)`
实例方法：在类中使用`def 方法名(self)`进行定义
静态方法：在类中使用`@staticmethod`修饰符，在下一行中使用`def 方法名()`
        静态方法不调用类和实例的任何属性或方法，如：帮助信息,所以括号为空
-------------------------------------------------------------------------
"""


class Game(object):
    # 历史最高分
    top_score = 0

    def __init__(self, player_name):
        self.player_name = player_name

    @staticmethod
    def show_help():
        print("这是游戏帮助： 让僵尸进入大门")

    @classmethod
    def show_top_score(cls):
        print(f'历史记录： {cls.top_score}')

    def start_game(self):
        print(f'{self.player_name}')


# 1. 查看游戏帮助
Game.show_help()
# 2. 查看历史最高分
Game.show_top_score()
# 3. 创建游戏对象
player1 = Game("laowang")
player1.start_game()
