"""
在游戏初始化时 定义一个变量调用 pygame.Rect 记录英雄的初始位置
在游戏循环中 每次让英雄的 y-1
y<=0 将英雄移动到屏幕底部

每一次调用 update() 方法之前,需要把所有的游戏图像都重新绘制一遍
而且应该最先重新绘制游戏的 背景图像
"""

import pygame

pygame.init()

# 创建游戏窗口 480 * 700
# 必须使用一个变量记录set_mode的返回结果！后续的所有图像都会基于这个返回结果
screen = pygame.display.set_mode((480, 700))

# 绘制背景图片
# 1. 加载图像
bg = pygame.image.load("./images/background.png")
# 2. blit绘制图像
screen.blit(bg, (0, 0))
# # 3.update刷新显示
# pygame.display.update()

# 绘制英雄的飞机,png是支持透明的
# 1. 加载图片
hero = pygame.image.load("./images/me1.png")
# 2. blit绘制图片
screen.blit(hero, (200, 500))

# 绘制工作完成后,统一update刷新显示
pygame.display.update()

# 创建时钟对象
clock = pygame.time.Clock()

# 1. 定义rect记录飞机的初始位置
hero_rect = pygame.Rect

# 这个循环在游戏中叫做"游戏循环",当代码执行到这里才意味着游戏的开始
while True:
    # 调用时钟对象的tick方法,设置刷新频率——帧率,为60次/秒
    clock.tick(60)

    pass

pygame.quit()
