import uiautomation as auto

def find_window_recursively(root, name, class_name=None, depth=0, max_depth=15):
    if depth > max_depth:
        return None

    if root.Name == name and (class_name is None or root.ClassName == class_name):
        return root

    for child in root.GetChildren():
        found = find_window_recursively(child, name, class_name, depth + 1, max_depth)
        if found:
            return found
    return None

def explore_window_structure(window_title, class_name=None, max_depth=20):
    auto.uiautomation.SetGlobalSearchTimeout(10)

    root = auto.GetRootControl()
    class_name = None if class_name == "" else class_name
    window = find_window_recursively(root, window_title, class_name, max_depth=max_depth)

    if not window:
        raise Exception(f"未找到标题为'{window_title}'的窗口，请确认窗口存在且未被最小化")

    window.SetActive()  # 激活窗口

    print("=== 窗口基本信息 ===")
    print(f"窗口标题: {window.Name}")
    print(f"类名: {window.ClassName}")
    print(f"坐标: {window.BoundingRectangle}")

    print("\n=== 深度递归子控件结构 ===")
    deep_search_controls(window, max_depth=max_depth)

def deep_search_controls(parent, depth=0, max_depth=20):
    if depth > max_depth:
        return
    children = parent.GetChildren()
    for child in children:
        prefix = "  " * depth
        print(f"{prefix}[{depth}] {child.ControlTypeName} | 名称: '{child.Name}' | 类名: {child.ClassName or '无'}")
        deep_search_controls(child, depth + 1, max_depth)

if __name__ == "__main__":
    print("=== 通用 GUI 结构探查工具 ===\n")

    # 必填：窗口标题
    while True:
        window_title = input("请输入窗口标题（必须项）: ").strip()
        if window_title:
            break
        print("⚠️  窗口标题是必须的，请重新输入。\n")

    # 选填：窗口类名
    class_name = input("请输入窗口类名（可选，留空回车表示不限制）: ").strip()

    # 选填：最大深度
    max_depth_input = input("请输入最大递归深度（可选，默认为 20）: ").strip()
    max_depth = int(max_depth_input) if max_depth_input.isdigit() else 20

    print(f"\n正在探查标题为 '{window_title}' 的窗口...")

    try:
        explore_window_structure(window_title, class_name, max_depth)
        print("\n✅ 探查完成！")
    except Exception as e:
        print(f"❌ 探查失败: {str(e)}")
