"""
readline 方法一次可以读取一行内容
方法执行以后，会把 文件指针 移动到下一行
"""

# 读取大文件的正确姿势

file = open("README", "r", encoding="utf-8")

while True:
    text = file.readline()

    # 判断一下是否还能读取到内容
    if not text:
        break

    print(text)

file.close()
