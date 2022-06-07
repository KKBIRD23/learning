"""
pygame.Rect 类定义了一个矩形,Rect的形参分别为:Rect(x, y, width, height)
Rect(对象原点坐标x, 对象原点坐标y, 对象矩形的宽, 对象矩形的高)
可以使用 对象.x 或 对象.y 或 对象.width 或 对象.height 的方式来获取对象的长宽高各属性
pygame.Rect.size方法,可以直接以元组形式输出对象的尺寸
"""

import pygame

hero_rect = pygame.Rect(100, 500, 120, 125)

print(f'英雄的原点{(hero_rect.x, hero_rect.y)}')
print(f'英雄的尺寸{(hero_rect.width, hero_rect.height)}')
# 使用size输出尺寸元组
print(f'{hero_rect.size}')
