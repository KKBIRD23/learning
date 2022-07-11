"""
使用socket(又叫做套接字)有以下几步:
1. 导入模块 socket
2. 创建socket,使用IPv4/UDP方式
3. 进行数据传递
4. 关闭socket
----------------------------------
socket.socket(协议类型, 传输方式)
协议类型: socket.AF_INET —— IPv4 ； socket.AF_INET6 —— IPv6
传输方式: socket.SOCK_DGRAM —— UDP ；socket.SOCK_STREAM —— TCP
"""

import socket

# 创建一个socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 传递数据


# 关闭socket
udp_socket.close()
