# 假设该内容是网上抓取的
# 按顺序并且居中对齐输出以下内容
poem = ["登鹳雀楼",
        "王之涣",
        "白日依山尽",
        "黄河入海流",
        "欲穷千里目",
        "更上一层楼"]

for poem_str in poem:
    print(f"|{poem_str.center(10, '*')}|")

for poem_str in poem:
    print(f"|{poem_str.ljust(10, '*')}|")

for poem_str in poem:
    print(f"|{poem_str.rjust(10, '*')}|")
