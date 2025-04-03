import uiautomation as auto
from typing import List, Dict, Union

def find_control_by_name(root, control_type, name):
    """递归查找指定控件类型和名称的控件"""
    if root.ControlTypeName == control_type and root.Name.strip() == name:
        return root
    for child in root.GetChildren():
        found = find_control_by_name(child, control_type, name)
        if found:
            return found
    return None

def get_all_descendants(control):
    """递归获取所有后代控件"""
    descendants = []
    for child in control.GetChildren():
        descendants.append(child)
        descendants.extend(get_all_descendants(child))
    return descendants

def get_wechat_messages(
        include_system: bool = False,
        include_self: bool = False,
        include_peer: bool = True,
        message_count: int = 0,
        window_title: str = "微信"
    ) -> List[Dict[str, Union[int, str]]]:
    """
    获取微信窗口聊天记录

    :param include_system: 是否包含系统消息
    :param include_self: 是否包含自己发送的消息
    :param include_peer: 是否包含他人发送的消息
    :param message_count: 要获取的消息数量(0表示获取所有)
    :param window_title: 微信窗口标题
    :return: 消息列表，每条消息包含ID、名称、昵称(如果有)和内容
    """
    # 连接微信窗口
    win = auto.WindowControl(searchDepth=1, Name=window_title)
    if not win.Exists(0):
        raise Exception(f"窗口 '{window_title}' 未找到，请确认微信是否打开。")
    
    # 根据结构，先查找名称为 "消息" 的列表控件
    list_items = find_control_by_name(win, "ListControl", "消息")
    if not list_items or not list_items.Exists(0):
        raise Exception("未找到聊天消息列表，请确保聊天窗口已打开。")
    
    # 打印所有消息项的控件类型、名称和类名————用于调试
    # for item in list_items.GetChildren():
    #     print(f"ControlTypeName: {item.ControlTypeName} | Name: '{item.Name}' | ClassName: {item.ClassName or '无'}")

    # 获取所有消息项
    messages = []
    ids = []
    if include_system:
        ids.append(0)
    if include_self:
        ids.append(1)
    if include_peer:
        ids.append(2)

    # 遍历 [11] ListItemControl 消息项
    for item in list_items.GetChildren():
        try:
            # 获取消息内容
            msg_content = item.Name.strip()

            # 获取消息类型
            # 检查是否是系统消息(第一个子元素有文本)
            first_child = item.GetChildren()[0]
            if first_child.Name.strip():
                # print(f"系统消息: {first_child.Name.strip()}")
                messages.append({
                    "ID": 0,
                    "name": "系统消息",
                    "content": msg_content,
                })
                continue

            # 检查是否是他人消息(第一个子元素的第一个子元素有文本)
            first_child_first_child = first_child.GetChildren()[0]
            if first_child_first_child.Name:
                other_name = first_child_first_child.Name.strip()
                # 尝试获取昵称
                aka_name = ""
                try:
                    aka_element = first_child.GetChildren()[1].GetChildren()[0].GetChildren()[0]
                    aka_name = aka_element.Name.strip()
                except Exception:
                    pass
                # print(f"他人消息: {other_name} - {aka_name} - {msg_content}")

                messages.append({
                    "ID": 2,
                    "name": other_name,
                    "AKA": aka_name,
                    "content": msg_content,
                })
            else:
                # 自己发送的消息
                my_name = first_child.GetChildren()[2].Name.strip()
                # print(f"自己消息: {my_name} - {msg_content}")
                messages.append({
                    "ID": 1,
                    "name": my_name,
                    "AKA": "",
                    "content": msg_content,
                })


        except Exception as e:
            # print(f"处理消息时出错: {e}")
            continue

    filtered_messages = [msg for msg in messages if msg["ID"] in ids]
    if message_count > 0 and message_count <= len(filtered_messages):
        filtered_messages = filtered_messages[-message_count:]
    return filtered_messages

if __name__ == "__main__":
    messages = get_wechat_messages(
        include_system=True,
        include_self=True,
        include_peer=True,
        message_count=10,
        window_title="微信"
    )

    for msg in messages:
        msg_type = '系统' if msg['ID'] == 0 else '自己' if msg['ID'] == 1 else '他人'
        print(f"类型: {msg_type}")
        print(f"名称: {msg['name']}")
        if msg['ID'] == 2 and 'AKA' in msg:
            print(f"昵称: {msg['AKA']}")
        print(f"内容: {msg['content']}")
        print("-" * 40)