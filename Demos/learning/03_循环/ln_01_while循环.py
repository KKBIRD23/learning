# 计算0-100之间所有整数的累计求和结果
import random

a = 0
i = 0
while i <= 100:
    a += i
    i += 1
print(f"a的值为 {a}")
