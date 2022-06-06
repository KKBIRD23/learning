import pygame

hero_rect = pygame.Rect(100, 500, 120, 125)

print(f'英雄的原点{(hero_rect.x, hero_rect.y)}')
print(f'英雄的尺寸{(hero_rect.width, hero_rect.height)}')
# 使用size输出尺寸元组
print(f'{hero_rect.size}')
