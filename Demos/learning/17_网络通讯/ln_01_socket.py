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
# udp_socket.sendto(要发送的数据的二进制格式， 对方IP和端口号)
# 参数一：字符串转二进制格式：字符串.encode()
# 参数二：对端的IP地址和端口号 格式为元组
udp_socket.sendto("着是一条测试信息".encode(), ("192.168.1.99", 8888))

# 关闭socket
udp_socket.close()
