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

# 这个循环在游戏中叫做"游戏循环",当代码执行到这里才意味着游戏的开始
while True:
    pass

pygame.quit()
