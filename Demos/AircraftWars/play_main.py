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

    # 初始化方法
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
        # 5. 设置定时器事件——英雄发射子弹 0.5s
        pygame.time.set_timer(HERO_FIRE_EVENT, 500)

    # 创建精灵和精灵组
    def __create_sprites(self):
        # 创建背景精灵和精灵组
        bg1 = Background()
        bg2 = Background(True)

        self.back_group = pygame.sprite.Group(bg1, bg2)

        # 创建敌机的精灵组
        self.enemy_group = pygame.sprite.Group()

        # 创建英雄的精灵和精灵组
        self.hero = Hero()
        self.hero_group = pygame.sprite.Group(self.hero)

    # 开始游戏
    def start_game(self):
        print("游戏开始...")
        # 游戏循环
        while True:
            # 1. 设置刷新率
            self.clock.tick(FRAME_PER_SEC)
            # 2. 进行事件监听
            self.__event_handler()
            # 3. 碰撞检测
            self.__check_collide()
            # 4. 更新/绘制精灵组
            self.__update_sprites()
            # 5. 更新显示
            pygame.display.update()

    # 事件监听
    def __event_handler(self):
        for event in pygame.event.get():
            # 判断用户是否点击退出按钮
            if event.type == pygame.QUIT:
                PlaneGame.__game_over()
            elif event.type == CREATE_ENEMY_EVENT:
                # print("敌机出场...")
                # 创建敌机精灵
                enemy = Enemy()
                # 将敌机精灵添加到敌机精灵组
                self.enemy_group.add(enemy)
            # 事件监听的方式是无法搞定按住不放的情况的
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            #     print("向右移动...")
            elif event.type == HERO_FIRE_EVENT:
                self.hero.fire()

        # 使用键盘提供的方法获取键盘按键——按键元组
        keys_pressed = pygame.key.get_pressed()
        # 判断元组中对应的按键索引值 1
        if keys_pressed[pygame.K_RIGHT]:
            self.hero.speed_x = 2
        elif keys_pressed[pygame.K_LEFT]:
            self.hero.speed_x = -2
        elif keys_pressed[pygame.K_UP]:
            self.hero.speed_y = -2
        elif keys_pressed[pygame.K_DOWN]:
            self.hero.speed_y = 2
        else:
            self.hero.speed_x = 0
            self.hero.speed_y = 0

    # 碰撞检测
    """
    碰撞的实现方法
    pygame.sprite.groupcollide() —— 两个精灵组中所有精灵的碰撞检测
    groupcollide(group1,group2,dokill1,dokill2,collided=None) -> Sprite_dict
    dokill1和dokill2是布尔值,dokill1是针对group1的操作,dokill2是针对group2的操作
    collided参数用户计算碰撞的回调函数,如果没有指定,则每个精灵必须有一个rect属性
    
    pygame.sprite.spritecollide() —— 某个精灵和指定精灵组中的精灵的碰撞
    spritecollide(sprite,group,dokill,collided=None) -> Sprite_list
    这里的dokill是针对group的,Sprite_list记录碰撞发生时敌机精灵组敌机的列表
    """
    def __check_collide(self):
        # 1. 子弹摧毁敌机
        pygame.sprite.groupcollide(self.hero.bullets, self.enemy_group, True, True)

        # 2. 敌机撞毁英雄
        # 用一个列表变量接收spritecollide方法的返回值
        enemise = pygame.sprite.spritecollide(self.hero, self.enemy_group, True)
        # 判断列表有内容时
        if len(enemise) > 0:
            # 干掉英雄
            self.hero.kill()
            # 结束游戏
            PlaneGame.__game_over()

    # 更新/绘制精灵组
    def __update_sprites(self):
        self.back_group.update()
        self.back_group.draw(self.screen)

        self.enemy_group.update()
        self.enemy_group.draw(self.screen)

        self.hero_group.update()
        self.hero_group.draw(self.screen)

        self.hero.bullets.update()
        self.hero.bullets.draw(self.screen)

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
