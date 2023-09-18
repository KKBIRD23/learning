#!/bin/bash

# 检查是否已经具有 root 权限
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：脚本需要 root 权限来配置串口号。"
    exit 1
fi

# 获取脚本所在目录
script_dir="$(cd "$(dirname "$0")" && pwd)"

# 配置文件路径
config_file="$script_dir/rapicomm.ini"

# 检查配置文件是否存在
if [ ! -e "$config_file" ]; then
    echo "错误：未找到配置文件 rapicomm.ini。"
    exit 1
fi

# 获取系统中存在的串口设备
available_ports=$(ls /dev/ttyUSB* 2>/dev/null)

# 检查是否找到串口设备
if [ -z "$available_ports" ]; then
    echo "错误：未找到串口设备。请确保串口设备已连接并正确识别。"
    exit 1
fi

# 清空旧的串口配置信息
sed -i '/^\[COMMS\]/,/^$/d' "$config_file"

# 生成新的串口配置信息并写入配置文件
new_comms="[COMMS]"
count=0
for port in $available_ports; do
    # 去除串口名称中的 "/dev/" 部分
    port_name="${port##*/}"
    com_entry="COM$count=$port_name:AUTO,N,8,1"
    new_comms="$new_comms\n$com_entry"
    count=$((count+1))
done

# 更新配置文件中的 [COMMS] 部分
sed -i "s/^\[SYSTEM\]/$new_comms\n\n[SYSTEM]/" "$config_file"

echo "已更新串口配置信息到 $config_file。"

# 使用 expect 执行 testDll，并将输出保存到临时文件
expect -c "
spawn $script_dir/testDll
expect {
    \"-> OK handle\" {
        exit 0
    }
    default {
        puts \"testDll 执行失败，返回结果：\$expect_out(buffer)\"
        exit 1
    }
}
"

# 获取 expect 的退出码
expect_exit_code=$?

# 根据 expect 的退出码输出相应消息
if [ $expect_exit_code -eq 0 ]; then
    echo "testDll 执行成功！"
    exit 0
else
    echo "testDll 执行失败，返回结果：$expect_exit_code"
    exit 1
fi
