"""
包(Package)
    - 包 是一个包含多个模块的 特殊目录
    - 目录下面有一个特殊的文件 __init__.py
    - 包的命名方式和变量一样， 小写字母 及 _
包的好处是使用 import 可以一次性导入 包中的所有模块
开发中，我们可以把多个相关联的模块打包，方便使用
"""

# 导入已经建立好的 ln_03_message 包
import ln_03_message

# 测试是否可以使用包中模块中的函数
ln_03_message.send_message.send("hello")

message = ln_03_message.receive_message.receive()
print(message)
