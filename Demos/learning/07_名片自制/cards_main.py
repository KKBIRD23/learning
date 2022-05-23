# 名片管理系统
import cards_tools

while True:
    cards_tools.show_menu()
    action_str = input("您选择的是： ")
    if action_str in ["1", "2", "3"]:
        # 执行对应功能
        # 功能1：新增名片
        if action_str == "1":
            cards_tools.create_card()

        # 功能2：名片查询
        elif action_str == "2":
            cards_tools.find_card()

        # 功能3：全部显示
        elif action_str == "3":
            cards_tools.show_all()

    # 0,退出程序
    elif action_str == "0":
        print("感谢使用名片管理系统，再见！")
        break

    else:
        print("您的输入错误，请重新输入！")
