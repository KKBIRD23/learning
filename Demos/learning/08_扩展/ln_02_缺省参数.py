def print_info(name, height=150, gender=True):
    """

    :param height: 乱来的体重
    :param name: 名字
    :param gender: True是男生，False是女生
    :return:
    """
    gender_text = "男生"
    if not gender:
        gender_text = "女生"

    print(f'{name} 是 {gender_text} 体重还他妈的是 {height}')


print_info("老王")
print_info("小美")
# 在对有多个默认参数的函数调用时，要给具体哪个参数传递值，需要指定参数名
print_info("大胖", 190, False)
