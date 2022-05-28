# 大文件的复制操作可以使用readline的方式进行

# 1. 打开文件
file_read = open("README", "r")
file_write = open("README[附件2]", "w")

# 2. 复制文件
while True:
    text = file_read.readline()
    if not text:
        break
    file_write.write(text)
    # if not file_read.readline():
    #     break
    # file_write.write(file_read.readline())
    # 为啥这个TMD会跳着读？？？

# 3. 关闭文件
file_read.close()
file_write.close()
