"""
`eval` 函数功能很强大——它的作用是 将字符串 内的内容当作 有效的表达式来求值 并返回计算结果
"""

input_str = input("输入算术题：")
print(eval(input_str))

# 特别注意！！！在开发时千万不要直接使用eval函数直接转换input的结果！！！
# 否则，将导致用户可以使用 __import__('os').system('终端命令') 的形式进行任何操作
# 终端命令可以为 touch、rm等危险操作
# __import__('os').system('ls') 等价于 import os 然后 os.system("ls")
