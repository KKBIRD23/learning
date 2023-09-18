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

# 提取串口配置信息
com_config=$(grep -E "^\[COMMS\]" -A 1 "$config_file" | grep "^COM[0-9]=" | sed 's/.*=//')

# 检查是否成功提取串口配置信息
if [ -z "$com_config" ]; then
    echo "错误：无法提取串口配置信息。请检查配置文件格式。"
    exit 1
fi

# 遍历串口配置信息
for com_entry in $com_config; do
    # 分割串口配置信息
    IFS=',' read -r -a com_parts <<< "$com_entry"

    # 检查配置信息是否有效
    if [ "${#com_parts[@]}" -eq 4 ]; then
        port="${com_parts[0]}"
        # 将串口配置中的 ":AUTO" 移除，以获取正确的串口设备名称
        port="${port/:AUTO/}"
        echo $port

        # 检查串口设备是否存在
        if [ -e "/dev/$port" ]; then
            # 配置串口设备
            stty -F "/dev/$port" "${com_parts[1]},${com_parts[2]}"
            echo "配置串口成功"

            # 执行 testDll
            echo "配置串口 /dev/$port 成功。"
            echo "执行 testDll..."
            # 使用 expect 来模拟按下 "q" 键并等待程序退出
            expect -c "spawn \"$script_dir/testDll\"; expect \"Press 'q' to quit\"; send \"q\"; interact"
            
            echo "testDll 执行完成。"
            exit 0
        else
            echo "错误：串口设备 /dev/$port 不存在。"
        fi
    else
        echo "错误：无效的串口配置信息：$com_entry"
    fi
done

# 如果没有找到可用的串口或配置失败，显示错误消息
echo "错误：无法配置正确的串口。请检查配置文件和串口设备。"
exit 1
