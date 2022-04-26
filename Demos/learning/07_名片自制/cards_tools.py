# 定义全局变量存储名片
cards_list = []


# 菜单
def show_menu():
    print("=" * 50)
    print("欢迎使用名片管理系统！请输入对应数字选择功能：")
    for menu in ["[1]新增名片", "[2]查询名片", "[3]查看所有", "----------", "[0]退出系统"]:
        print(menu)
    print("=" * 50)


# 1. 新增名片
def create_card():
    name_str = input("请输入姓名：")
    phone_str = input("请输入电话：")
    qq_str = input("请输入QQ：")
    email_str = input("请输入邮箱：")
    cards_dict = {"name": name_str,
                  "phone": phone_str,
                  "qq": qq_str,
                  "email": email_str}
    cards_list.append(cards_dict)
    print(cards_dict)
    print("您的信息已提交成功。")


# 2. 名片查询
def find_card():
    search_name = input("请输入您要查询的姓名： ")
    for card_dict in cards_list:
        if card_dict["name"] == search_name:
            print("姓名\t\t电话\t\tQQ\t\t邮箱")
            print("=" * 50)
            print(f'{card_dict["name"]}\t\t{card_dict["phone"]}\t\t{card_dict["qq"]}\t\t{card_dict["email"]}')

        # 找到以后调用自定义函数对卡后续操作——修改|删除
        deal_card(card_dict)
        break
    else:
        print(f"抱歉，没有找到 {search_name}")


# 3. 删除|修改
def deal_card(card_dict):
    """调用input_info_judge对名片进行处理

    :param card_dict: 被查找到的名片字典
    """
    print("=" * 50)

    for menu in ["[1]修改名片", "[2]删除名片", "[3]返回上级菜单"]:
        print(menu)
    print("=" * 50)

    choose_str = input("请输入数字进行选择： ")
    if choose_str in ["1", "2", "3"]:
        if choose_str == "1":
            # 修改名片
            card_dict["name"] = input_info_judge(card_dict["name"], "请输入名字，回车不修改")
            card_dict["phone"] = input_info_judge(card_dict["phone"], "请输入电话，回车不修改")
            card_dict["qq"] = input_info_judge(card_dict["qq"], "请输入QQ，回车不修改")
            card_dict["email"] = input_info_judge(card_dict["email"], "请输入邮箱，回车不修改")
            print("修改成功")
            print("-" * 50)
            print(card_dict)
            print("-" * 50)

        elif choose_str == "2":
            # 删除名片
            cards_list.remove(card_dict)
            print("名片删除成功")
    else:
        print("回到主菜单")


# 4. 提示用户输入，并对输入信息进行判断处理
def input_info_judge(ole_info_str, tip_message):
    """处理被找到的名片

    :param ole_info_str: 字典中原有名片信息
    :param tip_message: 提示用户输入的文字
    :return: 如果用户有输入内容，就传入输入的内容，否则返回字典原有信息
    """
    result_str = input(tip_message)
    if len(result_str) > 0:
        return result_str
    else:
        return ole_info_str


# 5. 全部显示
def show_all():
    for name in ["姓名", "电话", "QQ", "邮箱"]:
        print(name, end="\t\t")
    print("")
    print("-" * 50)
    for card_dict in cards_list:
        print(f'{card_dict["name"]}\t\t'
              f'{card_dict["phone"]}\t\t'
              f'{card_dict["qq"]}\t\t'
              f'{card_dict["email"]}')
