import pandas as pd
import os
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename

# 创建Tkinter根窗口
Tk().withdraw()  # 隐藏主窗口

# 弹出使用说明
messagebox.showinfo("使用说明", "请选择包含数据块的 TXT 文件。\n"
                                  "程序将提取信息并生成与输入文件同名的 XLSX 文件。")

# 打开文件选择对话框，让用户选择 TXT 文件
file_path = askopenfilename(title="选择数据文件", filetypes=[("Text files", "*.txt")])

if not file_path:  # 检查是否选择了文件
    print("没有选择文件，程序退出。")
else:
    # 获取文件的目录和文件名（不带扩展名）
    directory = os.path.dirname(os.path.abspath(file_path))
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（去除扩展名）

    # 初始化一个列表来存储每个数据块
    data_blocks = []

    # 尝试使用 utf-8 编码读取文件
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            block = []  # 临时存储每个信息块
            for line in file:
                stripped_line = line.strip()  # 去除首尾空白字符

                if stripped_line:  # 如果该行不是空行
                    block.append(stripped_line)  # 添加当前行到 block
                else:  # 遇到空行，处理当前信息块
                    if block:  # 如果 block 不为空
                        # 创建数据字典
                        data = {}
                        data["A"] = block[0]  # 第一行
                        if len(block) > 1:
                            data["B"] = block[1]  # 第二行
                        if len(block) > 2:
                            data["C"] = block[2]  # 第三行
                        if len(block) > 3:
                            data["D"] = block[3]  # 第四行
                        if len(block) > 4:
                            data["E"] = block[4]  # 第五行
                        if len(block) > 5:
                            data["F"] = block[5]  # 第六行
                        data_blocks.append(data)  # 保存数据块
                        block = []  # 清空 block，准备处理下一个信息块

            # 处理最后一个信息块（如果没有空行结尾）
            if block:
                data = {}
                data["A"] = block[0]  # 第一行
                if len(block) > 1:
                    data["B"] = block[1]  # 第二行
                if len(block) > 2:
                    data["C"] = block[2]  # 第三行
                if len(block) > 3:
                    data["D"] = block[3]  # 第四行
                if len(block) > 4:
                    data["E"] = block[4]  # 第五行
                if len(block) > 5:
                    data["F"] = block[5]  # 第六行
                data_blocks.append(data)  # 保存最后的数据块

    except UnicodeDecodeError:
        # 如果 utf-8 编码失败，尝试使用 latin1 编码
        with open(file_path, 'r', encoding='latin1') as file:
            block = []  # 临时存储每个信息块
            for line in file:
                stripped_line = line.strip()  # 去除首尾空白字符

                if stripped_line:  # 如果该行不是空行
                    block.append(stripped_line)  # 添加当前行到 block
                else:  # 遇到空行，处理当前信息块
                    if block:  # 如果 block 不为空
                        # 创建数据字典
                        data = {}
                        data["A"] = block[0]  # 第一行
                        if len(block) > 1:
                            data["B"] = block[1]  # 第二行
                        if len(block) > 2:
                            data["C"] = block[2]  # 第三行
                        if len(block) > 3:
                            data["D"] = block[3]  # 第四行
                        if len(block) > 4:
                            data["E"] = block[4]  # 第五行
                        if len(block) > 5:
                            data["F"] = block[5]  # 第六行
                        data_blocks.append(data)  # 保存数据块
                        block = []  # 清空 block，准备处理下一个信息块

            # 处理最后一个信息块（如果没有空行结尾）
            if block:
                data = {}
                data["A"] = block[0]  # 第一行
                if len(block) > 1:
                    data["B"] = block[1]  # 第二行
                if len(block) > 2:
                    data["C"] = block[2]  # 第三行
                if len(block) > 3:
                    data["D"] = block[3]  # 第四行
                if len(block) > 4:
                    data["E"] = block[4]  # 第五行
                if len(block) > 5:
                    data["F"] = block[5]  # 第六行
                data_blocks.append(data)  # 保存最后的数据块

    # 创建 DataFrame
    df = pd.DataFrame(data_blocks)

    # 保存为 Excel 文件到相同的文件夹，使用 openpyxl 引擎
    output_file_path = os.path.join(directory, f'{file_name}.xlsx')
    df.to_excel(output_file_path, index=False, engine='openpyxl')  # 指定引擎为 openpyxl

    print(f"数据已成功保存为 {output_file_path}")
