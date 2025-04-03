import pyautogui



# 1. 定义目标图像路径和搜索参数
template_path = "e://1.bmp"  # 目标图像路径
confidence = 0.8  # 匹配精度（0~1，值越低速度越快）
region = None  # 搜索区域（左,上,宽,高），None表示全屏

# 2. 批量查找所有匹配位置
instances = list(pyautogui.locateAllOnScreen(template_path, confidence=confidence, region=region, grayscale=False))

# 3. 输出结果
print(f"找到 {len(instances)} 个实例，坐标如下：{instances}")
for idx, loc in enumerate(instances):
    print(f"实例 {idx+1}: 左上角坐标 {loc}, 宽高 {loc[2]}x{loc[3]}")

# return [[loc.left, loc.top] for loc in instances]