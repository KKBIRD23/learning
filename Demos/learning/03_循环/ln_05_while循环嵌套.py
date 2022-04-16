# 在控制台连续输入5行*，每一行的*数量依次递增

"""
# 第一个写法
row = 1

while row<= 5:
    print("*" * row)
    row += 1
"""

row = 1

while row <= 5:
    col = 1
    while col <= row:
        print("*", end="")
        col += 1
    print("")
    row += 1
