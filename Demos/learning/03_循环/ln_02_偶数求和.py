# 计算0-100中的偶数求和

i = 0
result = 0

while i <= 100:
    if i % 2 == 0:
        print(i)
        result += i
    i += 1
print(f"最终的结果是 {result}")