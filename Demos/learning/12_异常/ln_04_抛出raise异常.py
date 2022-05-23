"""
在开发中，除了代码出错会抛异常。还可以根据应用场景主动抛出异常。比如：密码长度不够
Python中，提供了一个Exception异常类可以使用：
1. 创建一个 Exception 的对象
2. 使用 `raise` 关键字抛出异常对象
--------------------------------------------------------------------
需求：
- 定义input_password函数，提示用户输入密码
- 如果用户输入密码长度 < 8，抛出异常
- 如果用户输入长度 >= 8，返回输入的密码
"""


def input_password():
    # 1. 提示用户输入密码
    pwd = input("请输入密码：")

    # 2. 判断密码长度 >= 8,返回用户输入的密码
    if len(pwd) >= 8:
        return pwd
    # 3. 如果 < 8,主动跑异常
    print("主动抛出异常")
    # a. 创建异常对象 - 可以使用错误信息字符串作为参数
    ex = Exception("密码长度不够")
    # b. 主动抛出异常
    raise ex


try:
    # 提示用户输入密码
    print(input_password())
except Exception as result:
    print(result)
