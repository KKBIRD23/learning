"""
精灵和精灵组
为了简化开发,pygame提供了两个高级类: pygame.sprite.Sprite 和 pygame.sprite.Group
    - pygame.sprite.Sprite —— 存储 图像数据image和 位置rect 的对象
    - pygame.sprite.Group —— 存储精灵对象

pygame.sprite.Sprite类图
————————————————————————————————————
image 记录图像数据
rect 记录在屏幕上的位置
————————————————————————————————————
update(*args): 更新精灵的位置
kill(): 从所有组中删除
————————————————————————————————————

pygame.sprite.Group类图
————————————————————————————————————
__init__(self, *精灵):
add(*sprites): 向组中增加精灵
sprites(): 返回所有精灵列表
update(*args): 染组中所有精灵调用 update 方法
draw(Surface): 将组中所有精灵的image绘制到Surface的rect位置
————————————————————————————————————

派生精灵子类
- 定义 GameSprite 集成自 pygame.sprite.Sprite

注意:
- 如果一个类的父类不是object,在重写`初始化方法`的时候,一定要先super()一下父类的__init__方法
- 以此来保证父类的__init__代码能够被正常的执行
------------------------------------------------------------------------------------
GameSprite类图:
————————————————————————————————————
image
rect
speed
————————————————————————————————————
__init__(self, image_name, speeed=1):
update(self):
------------------------------------
属性:
- image 精灵图像,使用image_name加载
- rect 精灵矩形大小,默认使用图像大小
    - image 的 get_rect()方法,可以返回pygame.Rect(0,0,图像宽,图像高)的对象
- speed 精灵的移动速度,默认值为 1
方法:
- update 每次更新屏幕时在游戏循环内调用
    - 让精灵的 self.rect.y += self.speed ,让精灵在y轴方向运动
"""

import pygame

# 定义屏幕大小的常量
SCREEN_RECT = pygame.Rect(0, 0, 480, 700)
# 刷新帧率
FRAME_PER_SEC = 60


class GameSprite(pygame.sprite.Sprite):
    """游戏精灵"""

    def __init__(self, image_name, speed=1):
        # super()父类的初始化方法
        super().__init__()
        # 定义自身的对象初始化属性
        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    # update方法,让敌人的小灰机在y轴上向下移动
    def update(self):
        self.rect.y += self.speed
