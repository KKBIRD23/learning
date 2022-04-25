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
    search_name = input("请输入您要查询的姓名")
    for card_dict in cards_list:
        if card_dict["name"] == search_name:
            print("姓名\t\t电话\t\tQQ\t\t邮箱")
            print("=" * 50)
            print(f'{card_dict["name"]}\t\t{card_dict["phone"]}\t\t{card_dict["qq"]}\t\t{card_dict["email"]}')
        # 找到以后调用自定义函数对卡后续操作——修改|删除
        # TODO 删除|修改
    else:
        print(f"抱歉，没有找到 {search_name}")


# 3. 全部显示
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


# 4. 删除|修改
def deal_card(card_dict):
    print("=" * 50)
    print("欢迎使用名片管理系统！请输入对应数字选择功能：")
    for menu in ["[1]修改名片", "[2]删除名片", "[3]返回上级菜单"]:
        print(menu)
    print("=" * 50)

    choose_str = input("请输入数字进行选择")
    if choose_str in ["1", "2", "3"]:
        if choose_str == "1":
            # 修改名片
            card_dict["name"] = input("请输入名字，回车不修改")
            card_dict["phone"] = input("请输入电话，回车不修改")
            card_dict["qq"] = input("请输入QQ，回车不修改")
            card_dict["email"] = input("请输入邮箱，回车不修改")

        elif choose_str == "2":
            # 删除名片
            cards_list.remove(card_dict)
            print("名片删除成功")
    else:
        print("回到主菜单")

# 5. 处理名片修改
