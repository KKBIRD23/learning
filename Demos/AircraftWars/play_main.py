"""
目标: 明确主程序职责；实现主程序类；主板游戏精灵组

1. 明确主程序职责
一个游戏主程序的职责可以分为两个部分: 游戏初始化 和 游戏循环
设计 `PlaneGame` 类如下:
PlaneGame
————————————————————————
screen
clock
sprites和sprite_Group
————————————————————————
__init__(self):
__create_sprites(self):

start_game(self):
__event_handler(self):
__check_collide(self):
__update_sprites(self):
__game_over(self):
————————————————————————
"""

import pygame
from plane_sprites import *


class PlaneGame(object):
    """飞机大战主游戏"""

    def __init__(self):
        print("游戏初始化")

        # 1. 创建游戏窗口
        self.screen = pygame.display.set_mode(SCREEN_RECT.size)
        # 2. 创建游戏时钟
        self.clock = pygame.time.Clock()
        # 3. 调用私有方法,完成sprite和sprite_group的创建
        self.__create_sprites()
        # 4. 设置定时器事件——创建敌机 1s
        pygame.time.set_timer(CREATE_ENEMY_EVENT, 1000)

    def __create_sprites(self):
        # 创建背景精灵和精灵组
        bg1 = Background()
        bg2 = Background(True)

        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 创建敌机的精灵组
        self.enemy_group = pygame.sprite.Group()

    def start_game(self):
        print("游戏开始...")

        while True:
            # 1. 设置刷新率
            self.clock.tick(FRAME_PER_SEC)
            # 2. 进行事件监听
            self.__event_handler()
            # 3. 更新/绘制精灵组
            self.__update_sprites()
            # 4. 更新显示
            pygame.display.update()

    # 事件监听
    def __event_handler(self):
        for event in pygame.event.get():
            # 判断用户是否点击退出按钮
            if event.type == pygame.QUIT:
                PlaneGame.__game_over()
            elif event.type == CREATE_ENEMY_EVENT:
                print("敌机出场...")
                # 创建敌机精灵
                enemy = Enemy()
                # 将敌机精灵添加到敌机精灵组
                self.enemy_group.add(enemy)

    # 碰撞检测
    def __check_collide(self):
        pass

    # 更新/绘制精灵组
    def __update_sprites(self):
        self.back_group.update()
        self.back_group.draw(self.screen)

        self.enemy_group.update()
        self.enemy_group.draw(self.screen)

    # 退出游戏,这是个静态方法
    @staticmethod
    def __game_over():
        print("游戏结束")
        pygame.quit()
        exit()


# 固定套路:如果希望自己这个程序可以被当做模块调用,那就需要判断__name__的值是否是__main__
# 在pycharm中,直接敲`main`,就会有提示
if __name__ == '__main__':
    # 创建游戏对象
    game = PlaneGame()

    # 启动游戏
    game.start_game()
