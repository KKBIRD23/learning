# -*- coding: utf-8 -*-
"""
需求:
1、 创建检查类
    - 测试连接、取配置
2、 地址及密码管理类
    - 从文件取IP和密码
    - 把因为超连接次数的IP放到临时列表
    - 把放到临时列表的IP剩下的未尝试密码放到单独的列表
——————————————————————————————————————————————————————
Pw_Ip_Manage
--------------------
ip_list
pw_list
--------------------
__init__(self)
err_ip_manage
err_pw_manage

Check
--------------------
ip
pw
--------------------
__init__(self)
__check_config(self)


"""

import os
import sys
import paramiko
import retrying


class Check(object):
    """配置检查"""

    def __init__(self):