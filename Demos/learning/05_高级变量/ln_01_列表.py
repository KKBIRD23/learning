name_list = ["zhangsan", "lisi", "wangwu"]
# 取值
print(name_list[2])
# 取索引
print(name_list.index("lisi"))
# 修改
name_list[1] = "李四"
# 增加
# append方法末尾追加数据
name_list.append("王小二")
# insert指定位置插入
name_list.insert(1, "小妹妹")
# extend把其他列表追加到当前列表末尾
temp_list = ["孙悟空", "猪二哥", "沙师弟"]
name_list.extend(temp_list)
# 删除
# remove方法删除指定数据
name_list.remove("王小二")
# pop方法删除指定索引元素，默认删除最后一个
name_list.pop(3)
# clear清空列表
name_list.clear()
# len(length 长度)函数可以统计列表的元素个数
name_list = ["zhangsan", "lisi", "wangwu", "孙悟空", "猪二哥", "沙师弟"]
list_len = len(name_list)
print(list_len)
# count 方法统计列表中某一元素出现的次数
print(name_list.count("zhangsan"))

# del关键字可以直接从列表或元组中删除元素
# del关键字本质上是删除变量用的
del name_list[0]

# len函数 统计整个列表或元组的元素个数
# count方法 统计某个元素出现的次数
print(len(name_list))


print(name_list)

