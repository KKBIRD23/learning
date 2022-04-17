xiaoming ={"name": "小明",
           "age": 18,
           "gender": True,
           "weight": 75.5,
           "height": 1.75}

# 迭代遍历字典
# for遍历中的k变量，获取到的是字典中的key
for k in xiaoming:
    print(f"{k} - {xiaoming[k]}")
