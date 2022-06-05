import pygame

pygame.init()

# 创建游戏窗口 480 * 700
# 必须使用一个变量记录set_mode的返回结果！后续的所有图像都会基于这个返回结果
screen = pygame.display.set_mode((480, 700))

# 1. 加载图像
bg = pygame.image.load("./images/background.png")
# 2. blit绘制图像
screen.blit(bg, (0, 0))
# 3.update刷新显示
pygame.display.update()

while True:
    pass

pygame.quit()
