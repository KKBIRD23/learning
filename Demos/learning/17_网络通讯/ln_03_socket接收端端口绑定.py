"""
1、 导入模块
2、 创建套接字
3、 绑定端口
4、 接收对方数据
5、 解码数据
6、 输出
7、 关闭套接字
"""

# 1、 导入模块
import socket

# 2、 创建套接字
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 3、 绑定端口
udp_socket.bind(("", 8080))

# 4、 接收对方数据
# 这里使用了解包的方式,用两个变量分别接收元组中的元素,如 a, b = (1, 2)
recv_data, ip_port = udp_socket.recvfrom(1024)

# 5、 解码数据
print(udp_socket.recvfrom(1024))
print("未解码信息为", recv_data)
print("信息是从 ", ip_port, " 发送来的")
print("解码后的信息为 ", recv_data.decode(encoding="GBK", errors="ignore"))

# 6、 输出
# 7、 关闭套接字
udp_socket.close()
