# 1. 打开
file_read = open("README", "r", encoding="utf-8")
file_write = open("README[附件]", "w", encoding="utf-8")

# 2. 读、写
text = file_read.read()
file_write.write(text)

# 3. 关闭
file_read.close()
file_write.close()
