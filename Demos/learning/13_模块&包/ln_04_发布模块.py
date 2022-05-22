"""
制作发布压缩包的步骤

1. 创建 setup.py , 文件内容如下：
from distutils.core import setup

setup(name="ln_03_message",  # 包名
      version="1.0",  # 版本号
      description="Douglas.LW's 发送和接受消息模块",  # 描述信息
      long_description="完整的发送和接受消息模块",  # 完整描述信息
      author="Douglas.LW",  # 作者
      author_email="4355089@qq.com",  # 作者邮箱
      url="https://github.com/KKBIRD23",  # 作者主页
      py_modules=["ln_03_message.send_message",
                  "ln_03_message.receive_message"])  # 包内包含的模块

有关字典参数的详细信息，可以参阅官网：
https://docs.python.org/zh-cn/2/distutils/apiref.html

2. 构建模块
    - 要构建模块，setup.py只能在终端中执行，而不能在pycharm中执行
    python setup.py build  # 后面要加 `build` 参数

    - 执行之后会在当前目录下创建 ./build/lib/ 两个目录，ln_03_message包会拷贝进lib
    使用`tree`命令可以看到目录结构如下：
        文件夹 PATH 列表
        卷序列号为 0E33-0FC7
        E:.
        ├─build
        │  └─lib
        │      └─ln_03_message
        └─ln_03_message
            └─__pycache__
3. 生成发布压缩包
    - 生成压缩包依然需要在终端中执行
    python setup.py sdist  # 后面加 `sdist` 参数

    - 执行后会在当前目录下生成 `dist`目录，里面会存放我们打好的包 `ln_03_message-1.0.tar.gz`
        文件夹 PATH 列表
        卷序列号为 0E33-0FC7
        E:.
        ├─build
        │  └─lib
        │      └─ln_03_message
        ├─dist
        ├─ln_03_message
        │  └─__pycache__
        └─ln_03_message.egg-info

"""
