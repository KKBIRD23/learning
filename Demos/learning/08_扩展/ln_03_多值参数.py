"""
有时候需要一个函数能处理的参数个数是不确定的，这个时候需要使用多值参数
python中有两种多值参数：
    - 参数名前加一个 * 可以接收 元组
    - 参数名前加两个 * 可以接收 字典
一般在给多值参数命名的时候，习惯使用以下两个名字：
    - *args ———— 存放 元组 参数
    - **kwargs ———— 存放 字典 参数
`args`是`arguments`的缩写，有参数、变量的意思
`kw`是`keyword`的缩写，`kwargs`可以记忆 键值对参数 即 字典
"""


def demo(num, *nums, **person):
    print(num)
    print(nums)
    print(person)


demo(1)
print("-" * 50)

demo(1, 2, 3, 4, 5)
print("-" * 50)

demo(1, 2, 3, 4, 5, name="小米", age="18")
