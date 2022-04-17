# 字典是一个无需的数据集合
xiaoming ={"name": "小明",
           "age": 18,
           "gender": True,
           "weight": 75.5,
           "height": 1.75}

# 取值
print(xiaoming["name"])

# 增加/修改
xiaoming["cfg"] = 55
xiaoming["name"] = 20
# 删除
xiaoming.pop("age")

print(xiaoming)

# len统计字典
print(len(xiaoming))

# update方法合并字典
# 如果key存在，则会覆盖对应key的value
temp_dict = {"height": 1.75,
             "age": 20,
             "aaa": "bbb"}
xiaoming.update(temp_dict)
print(xiaoming)

# clear方法清空字典
xiaoming.clear()
print(xiaoming)
