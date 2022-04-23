import cards_tools

# 无限循环，由用户主动决定什么时候退出循环
while True:
    # 显示功能菜单
    cards_tools.show_menu()
    action_str = input("请选择希望执行的操作：")
    print(f"您选择的操作是 【{action_str}】")

    # 1，2,3，是针对名片的操作
    if action_str in ["1", "2", "3"]:
        # 新增
        if action_str == "1":
            cards_tools.new_card()
        # 显示全部
        elif action_str == "2":
            cards_tools.show_all()
        # 查询
        elif action_str == "3":
            cards_tools.search_card()
    # 0 退出系统
    elif action_str == "0":
        print("欢迎再次使用【名片管理系统】")
        break
        # 如果在开发过程中，不希望立刻编写分支内部的代码
        # 可以使用pass关键字，表示一个占位符，来保证程序的代码结构正确
        # pass不会执行任何操作
        # pass
    # 其他内容输入错误，需要提示用户
    else:
        print("您输入得不正确，请重新选择")
