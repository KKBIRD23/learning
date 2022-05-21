"""
异常传递：当 函数/方法 执行出现异常，会将异常传递给调用方。如果主程序依然没有异常处理，程序才会终止
在开发中，可以在主函数中增加 `异常捕获`，可保证代码的整洁
"""


def demo1():
    return int(input("输入整数 "))


def demo2():
    return demo1()


# 在主程序中增加异常捕获
try:
    print(demo2())
except Exception as result:
    print(f'未知错误:{result}')
