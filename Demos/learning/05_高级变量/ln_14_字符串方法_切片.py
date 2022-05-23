# 切片方法适用于字符串、列表、元组
num_str = "0123456789"

# 2-5 的字符串
print(num_str[2:6])
# 2-末尾
print(num_str[2:])
# 开始-5
print(num_str[:6])
# 完整字符串
print(num_str[:])
# 从开始每隔一个截取
print(num_str[::2])
# 从索引1，每隔一个
print(num_str[1::2])
# 截取2-末尾-1的字符串
print(num_str[2:-1])
# 截取末尾2个字符
print(num_str[-2:])

# 通过切片获取到字符串的逆序
print(num_str[-1::-1])
print(num_str[::-1])
print(num_str[::-2])
