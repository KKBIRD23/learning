from distutils.core import setup

# setup()函数接收的参数其实是一个 多值字典参数
setup(name="ln_03_message",  # 包名
      version="1.0",  # 版本号
      description="Douglas.LW's 发送和接受消息模块",  # 描述信息
      long_description="完整的发送和接受消息模块",  # 完整描述信息
      author="Douglas.LW",  # 作者
      author_email="4355089@qq.com",  # 作者邮箱
      url="https://github.com/KKBIRD23",  # 作者主页
      py_modules=["ln_03_message.send_message",
                  "ln_03_message.receive_message"])  # 包内包含的模块

