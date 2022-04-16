# \t \n算空白字符

poem = ["\t\n登鹳雀楼",
        "王之涣",
        "白日依山尽\t\n",
        "黄河入海流",
        "欲穷千里目",
        "更上一层楼"]

for poem_str in poem:
    print(f"|{poem_str.strip().center(10, '　')}|")

