import uiautomation as auto

def explore_wechat_structure(window_title="微信"):
    """兼容旧版uiautomation的微信窗口探查工具"""
    auto.uiautomation.SetGlobalSearchTimeout(10)
    
    # 连接微信窗口
    wechat = auto.WindowControl(
        searchDepth=1,
        Name=window_title,
        ClassName='WeChatMainWndForPC'
    )
    
    if not wechat.Exists(3, 1):
        raise Exception("微信窗口未找到，请检查：\n1. 窗口标题是否为'微信'\n2. 窗口未被最小化")
    
    print("=== 窗口基本信息 ===")
    print(f"窗口标题: {wechat.Name}")
    print(f"类名: {wechat.ClassName}")
    print(f"坐标: {wechat.BoundingRectangle}")
    
    print("\n=== 所有直接子控件 ===")
    children = wechat.GetChildren()
    for i, child in enumerate(children):
        print(f"[{i}] {child.ControlTypeName} | 名称: '{child.Name}' | 类名: {child.ClassName or '无'}")
        
    print("\n=== 查找所有列表控件 ===")
    list_count = 0
    for child in children:
        if child.ControlType == auto.ControlType.ListControl:
            list_count += 1
            print(f"\n列表{list_count}:")
            print(f"名称: '{child.Name}'")
            print(f"类名: {child.ClassName or '无'}")
            print(f"包含子项数: {child.GetChildCount()}")
            
            # 显示前3项示例
            list_items = child.GetChildren()
            for j in range(min(3, len(list_items))):
                item = list_items[j]
                print(f"  [{j}] {item.ControlTypeName} | 名称: '{item.Name}'")
    
    if list_count == 0:
        print("未找到任何列表控件，尝试深度搜索...")
        deep_search_controls(wechat)

def deep_search_controls(parent, depth=0, max_depth=20):
    """递归搜索控件结构"""
    if depth > max_depth:
        return
        
    children = parent.GetChildren()
    for child in children:
        prefix = "  " * depth
        print(f"{prefix}[{depth}] {child.ControlTypeName} | 名称: '{child.Name}' | 类名: {child.ClassName or '无'}")
        
        # 特别关注可能的消息容器
        if child.ControlType == auto.ControlType.ListControl:
            print(f"{prefix}  ^^^ 发现列表控件 ^^^")
        
        deep_search_controls(child, depth + 1, max_depth)

if __name__ == "__main__":
    print("正在探查微信窗口结构...")
    try:
        explore_wechat_structure()
        print("\n探查完成！请根据输出调整消息获取代码")
    except Exception as e:
        print(f"探查失败: {str(e)}")
        print("建议操作:")
        print("1. 确保微信窗口在最前面")
        print("2. 尝试更新uiautomation: pip install --upgrade uiautomation")
        print("3. 使用Inspect.exe工具手动查看控件结构")