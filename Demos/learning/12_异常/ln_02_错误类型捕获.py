"""
错误类型捕获：
在程序执行时，可能会遇到不同类型的异常，并且需要针对不同类型的异常，做出不同的响应时，就需要捕获错误类型
语法为：
try:
    # 尝试执行的代码

except 错误类型1:
    # 针对错误类型1对应的代码处理

except (错误类型2, 错误类型3):
    # 针对错误类型2和错误类型3，对应的代码处理

except Exception as result:
    print(f'未知错误 {result}')

else:
    # 没有异常才会执行的代码

finally:
    # 无论是否有异常，都会执行的代码
"""


try:
    num = int(input("输入一个整数："))

    result = 8 / num
    print(result)

# 一种错误
except ZeroDivisionError:
    print("除0错误")
# 一种错误
except ValueError:
    print("值错误")
# 多种错误
except (ZeroDivisionError, ValueError):
    print("输入错误")
# 未知错误(result是个变量名而已，可以改成其他的)
except Exception as result:
    print(f'未知错误 {result}')
# 没有错误才执行
else:
    print("居然没有错？？")
# 无论是否出错，都执行
finally:
    print("程序和程序员有一个能跑就行！")
