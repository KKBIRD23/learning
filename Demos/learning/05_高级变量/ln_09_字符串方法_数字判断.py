# 判断是否是数字
# 1. 三个方法都不能判断小数
# 2. isdecimal 只能判断十是否包含全角数字
# 3. isdigit 包含isdecimal的范围，增加了(1)--小括号数字和\u00b2--Unicode码数字
# 4. isnumeric 包含isdigit的范围，增加了中文数字
# 5. 开发中尽量选择isdecimal方法

num_str = "1.1"
num_str1 = "①"
num_str2 = "一二"

print(num_str, end="\t\t")
print(num_str.isdecimal())
print(num_str1, end="\t\t")
print(num_str1.isdigit())
print(num_str2, end="\t\t")
print(num_str2.isnumeric())
