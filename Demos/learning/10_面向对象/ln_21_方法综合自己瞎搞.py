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
"""
import random


class Game(object):
    top_score = 0
    player_dic = {}

    @classmethod
    def show_top_score(cls):
        print(cls.player_dic)

        # 生成一个空字典，存放颠倒后的字典
        tmp_list = {}
        for key in cls.player_dic:
            tmp_list[str(cls.player_dic[key])] = key

        print(f'这个游戏的TOP玩家是：”{tmp_list[max(tmp_list)]}” 他的得分是”{max(tmp_list)}”')

    @staticmethod
    def show_help():
        print("这是一个游戏的帮助信息")

    def __init__(self, player_name):
        self.player_name = player_name
        self.score = 0

    def start_game(self):
        print(f'{self.player_name}已经开始游戏了哟~~')
        self.score = random.randint(60, 100)
        print(f'{self.player_name}玩家的得分是：{self.score}')
        Game.player_dic[f'{self.player_name}'] = self.score


player1 = Game("laowang")
player2 = Game("xiaomei")
player3 = Game("xiaohong")
player4 = Game("qinqin")

player1.start_game()
player2.start_game()
player3.start_game()
player4.start_game()

Game.show_top_score()
