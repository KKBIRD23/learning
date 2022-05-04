"""
重点：一个对象的 属性 可以是另外一个类创建的对象

士兵突击
需求：
1. 士兵 许三多 有一把 AK47
2. 士兵 可以 开火
3. 枪 能够 发射子弹
4. 枪 能够装填子弹
-------------------------
需求分析：
Soldier
name
gun
```````
__init__(self):
fire(self):

Gun
model
bullet_count
```````
__init_(self, model):
add_bullet(self, count):
shoot(self):
"""


class Gun:
    def __init__(self, model):
        # 1. 枪的型号
        self.model = model
        # 2. 子弹的数量
        self.bullet_count = 0

    # 装填子弹方法
    def add_bullet(self, count):
        self.bullet_count += count

    # 开枪(shoot)方法
    def shoot(self):
        # 1. 判断子弹数量
        if self.bullet_count <= 0:
            print(f'{self.model}没有子弹了！')
            return
        # 2. 发射子弹， 子弹数量-1
        self.bullet_count -= 1
        # 3. 提示发射信息
        print(f'{self.model}突突突... {self.bullet_count}')


class Soldier:
    def __init__(self, name):
        self.name = name
        # 枪对象 - 新兵没有枪
        self.gun = None

    def fire(self):
        # 1. 判断有没有枪
        if self.gun is None:
            print(f'{self.name}还他么没有枪！')
            return
        # 2. 高喊口号
        print(f'冲啊！！！ {self.name}')
        # 3. 装填子弹
        self.gun.add_bullet(50)
        # 4. 发射子弹
        self.gun.shoot()


# 1. 创建枪对象
ak47 = Gun("AK47")

# 2. 创建许三多
xusanduo = Soldier("许三多")
xusanduo.gun = ak47
xusanduo.fire()
print(xusanduo.gun)
