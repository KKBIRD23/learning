# 使用多个键值对，存储描述一个物体的相关信息
# 将多个字典放到一个列表中进行遍历

card_list = [
    {"name": "张三",
     "qq": "12345",
     "phone": "110"},
    {"name": "李四",
     "qq": "54321",
     "phone": "10086"}
]

for card_info in card_list:
    print(card_info)

a = card_list[0]["name"]
print(a)

