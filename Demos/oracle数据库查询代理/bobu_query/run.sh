#!/bin/bash

# 设置Oracle客户端字符集环境变量
export NLS_LANG="SIMPLIFIED CHINESE_CHINA.ZHS16GBK"

# ---------------------------
# 配置区
# ---------------------------
SERVER_SCRIPT="server.py"           # 服务端主程序

# ---------------------------
# 预检查
# ---------------------------
# 检查server.py是否存在
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "错误：找不到服务端脚本 ${SERVER_SCRIPT}！"
    exit 1
fi

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误：未找到Python解释器，请确保已安装Python！"
    exit 1
fi

# ---------------------------
# 启动服务（前台运行，日志直接输出到Byobu窗口）
# ---------------------------
echo "启动服务（NLS_LANG=$NLS_LANG）..."
python "$SERVER_SCRIPT"
