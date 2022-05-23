def multiple_table():

    """
    九九乘法表
    1. 行号为1-9
    2. 被乘数即为行号
    3. 乘数为1--行号
    """
    # x为乘数，y为被乘数
    
    y = 1
    while y <= 9:
        x = 1
        while x <= y:
            print(f"{x} * {y} =", x * y, end="\t")
            x += 1
        print("\r")
        y += 1
    