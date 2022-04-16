hello_str = "hello world"

# 1. 判断是否以指定字符串开始
print(hello_str.startswith("hello"))
print(hello_str.startswith("Hello"))

# 2. 判断是否以指定字符串结束
print(hello_str.endswith("world"))

# 3. 查找指定字符串
# index方法同样可以查找指定的字符串在大字符串中的索引
# index方法找不到就报错，find方法找不到返回-1
print(hello_str.find("llo"))
print(hello_str.index("llo"))
print(hello_str.find("abc"))

# 4. 替换字符串
# replace方法执行执行完后会返回一个新的字符串
# 注意：不会修改原字符串
print(hello_str.replace("world", "WORLD"))
print(hello_str)
