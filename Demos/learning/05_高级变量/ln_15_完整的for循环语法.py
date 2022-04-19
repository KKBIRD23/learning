for num in [1, 2, 3]:
    print(num)
    if num == 2:
        break
else:
    # 如果循环内部使用了break退出循环，else的代码就不会被执行
    print("会执行吗")
print("循环结束")
