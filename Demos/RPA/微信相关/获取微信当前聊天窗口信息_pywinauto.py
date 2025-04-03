from pywinauto import Application
from pywinauto.controls.uia_controls import ListItemWrapper
from typing import List, Dict, Union

def get_wechat_messages(
        include_system: bool = False,
        include_self: bool = False,
        include_peer: bool = True,
        message_count: int = 0,
        window_title: str = "微信"
    ) -> List[Dict[str, Union[int, str, ListItemWrapper]]]:
    """
    获取微信窗口聊天记录
    
    :param include_system: 是否包含系统消息
    :param include_self: 是否包含自己发送的消息
    :param include_peer: 是否包含他人发送的消息
    :param message_count: 要获取的消息数量(0表示获取所有)
    :param window_title: 微信窗口标题
    :return: 消息列表，每条消息包含ID、名称、昵称(如果有)、内容和元素对象
    """
    # 连接微信窗口
    app = Application(backend="uia").connect(title=window_title, timeout=10)
    win = app.window(title=window_title)
    
    # 确定要获取的消息类型ID
    ids = []
    if include_system:
        ids.append(0)  # 系统消息ID
    if include_self:
        ids.append(1)  # 自己消息ID
    if include_peer:
        ids.append(2)  # 他人消息ID
    
    messages = []
    
    # 获取聊天列表项
    list_items = win.descendants(control_type="ListItem")
    
    for item in list_items:
        try:
            # 获取消息内容
            msg_content = item.window_text()
            
            # 检查是否是系统消息(第一个子元素有文本)
            first_child = item.children()[0]
            if first_child.window_text():
                messages.append({
                    "ID": 0,
                    "name": "系统消息",
                    "content": msg_content,
                    # "element": item
                })
                continue
                
            # 检查是否是他人消息(第一个子元素的第一个子元素有文本)
            first_grandchild = first_child.children()[0]
            if first_grandchild.window_text():
                other_name = first_grandchild.window_text()
                
                # 尝试获取昵称
                aka_name = ""
                try:
                    # 这个路径可能需要根据实际微信UI结构调整
                    aka_element = first_child.children()[1].children()[0].children()[0]
                    aka_name = aka_element.window_text()
                except IndexError:
                    pass
                    
                messages.append({
                    "ID": 2,
                    "name": other_name,
                    "AKA": aka_name,
                    "content": msg_content,
                    # "element": item
                })
            else:
                # 自己发送的消息
                my_name = first_child.children()[2].window_text()
                messages.append({
                    "ID": 1,
                    "name": my_name,
                    "AKA": "",
                    "content": msg_content,
                    # "element": item
                })
                
        except Exception as e:
            # 忽略处理失败的项目
            continue
    
    # 过滤指定类型的消息
    filtered_messages = [msg for msg in messages if msg["ID"] in ids]
    
    # 截取指定数量的消息
    if message_count > 0 and message_count <= len(filtered_messages):
        filtered_messages = filtered_messages[-message_count:]
    
    return filtered_messages


if __name__ == "__main__":
    # 示例用法
    messages = get_wechat_messages(
        include_system=False,
        include_self=True,
        include_peer=True,
        message_count=10,
        window_title="微信"
    )
    
    for msg in messages:
        print(f"类型: {'系统' if msg['ID'] == 0 else '自己' if msg['ID'] == 1 else '他人'}")
        print(f"名称: {msg['name']}")
        if msg['ID'] == 2 and msg['AKA']:
            print(f"昵称: {msg['AKA']}")
        print(f"内容: {msg['content']}")
        print("-" * 40)