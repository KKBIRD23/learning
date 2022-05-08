"""
海象运算符是在 PEP 572 中提出，并在 Python3.8 版本并入，原名叫Assignment Expresions ，即“赋值表达式”
语法：(variable_name := expression or value)
    即一个变量名后跟一个表达式或者一个值，这个和赋值运算符 = 类似，可以看作是一种新的赋值运算符
"""
# 1 用于 if-else 条件表达式
# 一般写法：
a = 15
if a > 10:
    print('hello, walrus operator!')

# 海象运算符：
if a := 15 > 10:
    print('hello, walrus operator!')

# =====================================
# 用于 while 循环
# （1）n 次循环的一般写法：
n = 5
while n:
    print('hello, walrus operator!')
    n -= 1
# 海象运算符：
n = 5
while (n := n - 1) + 1:  # 需要加1是因为执行输出前n就减1了
    print('hello, walrus operator!')
# ================
# （2） 实现一个密码输入检验的一般写法：
while True:
    psw = input("请输入密码：")
    if psw == "123":
        break
# 更优雅的实现方式：海象运算符。
while (psw := input("请输入密码：")) != "123":
    continue
# （3）读取一个文件的每一行并输出：
# ================
fp = open("test.txt", "r")
while True:
    line = fp.readline()
    if not line:
        break
    print(line.strip())
fp.close()
# 更简洁的实现：
fp = open("test.txt", "r")
while line := fp.readline():
    print(line.strip())
# =====================================
# 用于列表推导式
# 计算元素平方根，并保留平方根大于 5 的值：
# 一共就 4 个数字，但是函数被执行了 7 次。因为有三个数字满足列表推导式的条件，需要再额外计算 3次
nums = [16, 36, 49, 64]


def f(x):
    print('运行了函数f(x)1次。')
    return x ** 0.5


print([f(i) for i in nums if f(i) > 5])

nums = [16, 36, 49, 64]


# ================
# 使用海象海运算符提高执行效率
# 函数只执行了 4 次，函数执行结果被 n 储存，不需要额外计算。性能优于不使用 := 的
def f(x):
    print('运行了函数f(x)1次。')
    return x ** 0.5


print([n for i in nums if (n := f(i)) > 5])
