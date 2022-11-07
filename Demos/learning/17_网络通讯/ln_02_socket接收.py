"""
1、导入模块
2、创建套接字
3、发送数据
4、接收数据
    - udp_socket.recvfrom(接收缓冲器大小)  可设置为1024
    - recvfrom方法会阻塞程序的运行，引起程序的等待
    - recvfrom的返回值由两个元组构成：recv_data[0]是接收到的“二进制信息”，recv_data[1]是信息发送方的IP和端口
5、解码数据得到字符串
    - 由于recvfrom[0]是二进制信息，所以需要解码
    - recvfrom[0].decode()函数对信息进行解码，可以通过decode("GBK")的方式对编码进行指定
6、输出显示接收到的内容
8、关闭套接字

---------------------------------
decode进行解码的时候有两个参数可以选择:
str.decode(encoding="字符集", errors="错误模式")
    - 常用中文字符集如GBK、UTF-8
    - 错误模式有 ignore 忽略错误,和 strict 严格模式.设置为strict的时候,当字符集不对应编码错误,将报错
"""
# 1、导入模块
import socket

# 2、创建套接字
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 3、发送数据
udp_socket.sendto("test".encode(), ("192.168.1.99", 8888))

# 4、接收数据
recv_data = udp_socket.recvfrom(1024)
# 5、解码数据得到字符串
# recv_test = recv_data[0].decode("gbk")
recv_test = recv_data[0].decode(encoding="UTF8",errors="ignore")    # 虽然字符集不正确,但忽略错误
# 6、输出显示接收到的内容
print(recv_data, "\n =======")
print(recv_data[0], "\n =======")
print(recv_data[1], "\n =======")
print(recv_test)
# 8、关闭套接字
udp_socket.close()
