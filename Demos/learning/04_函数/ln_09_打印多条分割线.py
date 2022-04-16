def print_line(char, times):
    """打印单行分割线

    :param char:打印的字符
    :param times:打印的次数
    :return:
    """
    # print(char * 50)
    return print(char * times)


def print_lines(i, char, times):
    """打印多行分割线

    :param i:打印的行数
    :param char:分割线使用的字符
    :param times:打印的次数
    """
    row = 0
    while row < i:
        print_line(char, times)
        row += 1


name = "黑马程序员"
