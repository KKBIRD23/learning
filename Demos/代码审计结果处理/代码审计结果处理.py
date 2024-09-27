import pandas as pd
import os
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename

# 创建Tkinter根窗口
Tk().withdraw()  # 隐藏主窗口

# 弹出使用说明
messagebox.showinfo("使用说明", "把每一个代码块的扫描结果存为data.txt 文件。\n"
                                  "程序将提取信息并生成 warnings.xlsx 文件。")

# 打开文件选择对话框，让用户选择 data.txt 文件
file_path = askopenfilename(title="选择数据文件", filetypes=[("Text files", "*.txt")])

if not file_path:  # 检查是否选择了文件
    print("没有选择文件，程序退出。")
else:
    # 获取文件的目录
    directory = os.path.dirname(os.path.abspath(file_path))

    # 初始化一个列表来存储每个数据块
    data_blocks = []

    with open(file_path, 'r', encoding='utf-8') as file:
        block = []  # 临时存储每个信息块
        for line in file:
            stripped_line = line.strip()  # 去除首尾空白字符

            if stripped_line:  # 如果该行不是空行
                # 检查是否开始新的信息块
                if stripped_line.startswith("中等") or stripped_line.startswith("高危"):
                    # 如果已有内容，保存之前的块
                    if len(block) > 0:
                        # 创建数据字典
                        data = {
                            "A": block[0],  # 第一行
                            "B": block[1],  # 第二行
                            "C": block[2],  # 第三行
                            "D": block[3],  # 第四行
                        }
                        if block[0].startswith("高危"):  # 如果是高危，检查行数
                            if len(block) >= 6:
                                data["E"] = block[4]  # 第五行
                                data["F"] = block[5]  # 第六行
                            else:
                                data["E"] = ""
                                data["F"] = ""
                        else:  # 中等类型
                            data["E"] = block[4]  # 第五行
                        data_blocks.append(data)  # 保存数据块
                        block = []  # 清空 block，准备处理下一个信息块

                block.append(stripped_line)  # 添加当前行到 block

        # 处理最后一个信息块（如果没有空行结尾）
        if block:
            data = {
                "A": block[0],  # 第一行
                "B": block[1],  # 第二行
                "C": block[2],  # 第三行
                "D": block[3],  # 第四行
            }
            if block[0].startswith("高危"):  # 如果是高危，检查行数
                if len(block) >= 6:
                    data["E"] = block[4]  # 第五行
                    data["F"] = block[5]  # 第六行
                else:
                    data["E"] = ""
                    data["F"] = ""
            else:  # 中等类型
                data["E"] = block[4]  # 第五行
            data_blocks.append(data)  # 保存最后的数据块

    # 创建 DataFrame
    df = pd.DataFrame(data_blocks)

    # 保存为 Excel 文件到相同的文件夹
    output_file_path = os.path.join(directory, 'warnings.xlsx')
    df.to_excel(output_file_path, index=False)

    print(f"数据已成功保存为 {output_file_path}")
