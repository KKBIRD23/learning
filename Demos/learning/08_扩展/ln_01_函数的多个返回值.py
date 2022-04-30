def measure():
    """测试文档和适度"""

    print("测量开始...")
    temp = 39
    wetness = 50

    print("测量结束...")

    # 元祖|列表 - 可以包含多个数据，因此可以使用元祖让函数返回多个值
    # 如果函数放回的类型是元祖，小括号可以省略
    # return [temp, wetness]
    # 还可以用字典进行返回
    return temp, wetness


# 使用多个变量，一次接受函数的返回结果
gl_temp, gl_wetness = measure()
print(gl_temp)
print(gl_wetness)
