import pygame

# pygame.Rect 类定义了一个矩形,Rect的形参分别为:
# Rect(对象原点坐标x, 对象原点坐标y, 对象矩形的宽, 对象矩形的高)
# pygame.Rect.size方法,可以直接以元组形式输出对象的尺寸
hero_rect = pygame.Rect(100, 500, 120, 125)

print(f'英雄的原点{(hero_rect.x, hero_rect.y)}')
print(f'英雄的尺寸{(hero_rect.width, hero_rect.height)}')
# 使用size输出尺寸元组
print(f'{hero_rect.size}')
