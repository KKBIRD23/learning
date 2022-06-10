"""
1. 初始化背景、英雄的飞机
在游戏初始化时 定义一个变量调用 pygame.Rect 记录英雄的初始位置
在游戏循环中 每次让英雄的 y-1
y<=0 将英雄移动到屏幕底部

每一次调用 update() 方法之前,需要把所有的游戏图像都重新绘制一遍
而且应该最先重新绘制游戏的 背景图像
-----------------------------------------------------
2. 事件 event
- 游戏启动后,用户针对游戏所做的操作
- 例如:点击关闭按钮、点击鼠标、按下键盘...等

监听
- 在游戏循环中,判断用户具体操作
- 只有捕获到用户具体的操作,才能有针对性的做出响应

代码实现
- pygame中,通过pygame.event.get()方法 可以获得用户当前所做的动作的 `事件列表`\
- 注意这是一个列表,因为用户同一时间可以做很多事
-----------------------------------------------------
3. 使用 游戏精灵 和 精灵组 创建敌机,并且实现敌机动画
步骤:
- 导入 `plane_sprites` 模块
- 在 游戏初始化时创建 精灵对象 和 精灵组对象
    * 封装 图像image、位置rect 和 速度 speed
    * 提供 update() 方法,根据游戏需求,更新位置rect
- 在游戏循环中 让 精灵组调用 update() 和 draw() 方法
    * 包含多个 精灵对象
    * update 方法,让精灵组中的所有精灵调用各自的update方法更新各自的位置
    * draw(screen)方法,在screen上绘制精灵组中的所有精灵
"""

import pygame
from plane_sprites import *

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
screen.blit(hero, (150, 300))

# 绘制工作完成后,统一update刷新显示
pygame.display.update()

# 创建时钟对象
clock = pygame.time.Clock()

# 1. 定义rect记录飞机的初始位置
hero_rect = pygame.Rect(150, 300, 102, 126)
print(f'这里是hero_rect的值 {hero_rect}')

# 创建敌机的精灵
enemy = GameSprite("./images/enemy1.png")
enemy1 = GameSprite("./images/enemy1.png", 2)
# 创建敌机精灵组
enemy_group = pygame.sprite.Group(enemy, enemy1)

# 这个循环在游戏中叫做"游戏循环",当代码执行到这里才意味着游戏的开始
while True:
    # 调用时钟对象的tick方法,设置刷新频率——帧率,为60次/秒
    clock.tick(60)

    # 捕获事件
    for event in pygame.event.get():
        # 判断事件类型是否是退出事件
        if event.type == pygame.QUIT:
            print("游戏退出...")
            # 调用quit方法卸载所有模块
            pygame.quit()
            # 调用内置函数exit()退出所有程序
            exit()

    # 2. 修改飞机的位置
    hero_rect.y -= 1
    # 判断飞机的位置 y<=0
    if hero_rect.bottom <= 0:
        hero_rect.y = 700

    # 3. 调用blit方法绘制图像,注意这里直接传入了一个矩形对象`hero_rect`
    # 如果不重新绘制背景,会留下飞机残影,所以画飞机之前要先重新绘制背景
    screen.blit(bg, (0, 0))
    screen.blit(hero, hero_rect)

    # 让精灵组调用两个方法
    # update
    enemy_group.update()
    # draw
    enemy_group.draw(screen)

    # 4. 调用update方法刷新显示
    pygame.display.update()

