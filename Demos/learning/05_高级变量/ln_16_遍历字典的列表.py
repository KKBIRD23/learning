students = [
    {"name": "阿土",
     "age": "19"},
    {"name": "小美",
     "age": "18"}
]
find_name = "小美"
find_age = "19"
for stu_dict in students:
    print(stu_dict)
    if stu_dict["age"] == find_age:
        print(f'找到了 {stu_dict["name"]} , 年龄是{stu_dict["age"]}')
        # 找到就退出
        break
else:
    # 如果遍历实现都没有找到搜索目标，执行以下
    print(f"抱歉没有找到 {find_age}")
print("循环结束")
