"""
和电脑猜拳
1 石头；2 剪刀；3 布
"""
import random as ran

echo = ""
player = int(input("请输入你要出的拳(1:石头、2:剪刀、3:布)  "))
computer = ran.randint(1, 3)

# 我方获胜情况：1、我石头&电脑剪刀 2、我剪刀&电脑布 3、我布&电脑石头
# 双方打平
if player == computer:
    print("不要走！决战到天亮！")
elif ((player == 1 and computer == 2)
      or (player == 2 and computer == 3)
      or (player == 3 and computer == 1)):
    print("电脑弱爆了")
else:
    print("尼玛！")
