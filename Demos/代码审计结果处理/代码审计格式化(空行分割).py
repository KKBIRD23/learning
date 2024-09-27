import pandas as pd
import os
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename


def read_file_by_empty_lines(file_path):
    """
    读取文件，按空行分割内容，并存储到列表中

    :param file_path: 文件的路径
    :return: 包含分割后内容的列表
    """
    # 初始化一个空列表来存储分割后的内容块
    blocks = []
    # 初始化一个空字符串来存储当前内容块
    current_block = ""

    # 打开文件
    with open(file_path, 'r') as file:
        for line in file:
            # 去除每行末尾的换行符
            stripped_line = line.strip()

            # 如果当前行不是空行，则将其添加到当前内容块
            if stripped_line:
                current_block += stripped_line + "\n"
            else:
                if current_block:  # 确保不是空的开始或结束
                    blocks.append(current_block.strip())
                    current_block = ""

    # 将最后一个内容块（如果有的话）添加到列表中
    if current_block:
        blocks.append(current_block.strip())

    return blocks


# 创建Tkinter根窗口
Tk().withdraw()  # 隐藏主窗口

# 弹出使用说明
messagebox.showinfo("使用说明", "把每一个代码块的扫描结果存为单独的 TXT 文件.\n"
                                "程序将提取信息并生成与输入文件同名的 XLSX 文件.")

# 打开文件选择对话框，让用户选择 data.txt 文件
file_path = askopenfilename(title="选择数据文件", filetypes=[("Text files", "*.txt")])
file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名（去除扩展名）

if not file_path:  # 检查是否选择了文件
    print("没有选择文件，程序退出。")
    exit()
else:
    # 获取文件的目录
    directory = os.path.dirname(os.path.abspath(file_path))

# 使用示例
content_blocks = read_file_by_empty_lines(file_path)
datas = []  # 移除列标题

# 打印结果
for block in content_blocks:
    blocks = block.split('\n')
    # 检查长度以避免索引错误
    part1 = blocks[0] if len(blocks) > 0 else ''
    part2 = blocks[1] if len(blocks) > 1 else ''
    part3 = '\n'.join(blocks[2:]) if len(blocks) > 2 else ''

    # 将数据追加到 datas 中
    datas.append([part1, part2, part3])

# 添加列标题
datas.insert(0, ['类型', '位置', '描述'])

# 创建 DataFrame
df = pd.DataFrame(datas[1:], columns=datas[0])  # 使用第一行作为列名

# 保存为 Excel 文件到相同的文件夹
output_file_path = os.path.join(directory, f'{file_name}.xlsx')
df.to_excel(output_file_path, index=False)

print(f"数据已成功保存为 {output_file_path}")
