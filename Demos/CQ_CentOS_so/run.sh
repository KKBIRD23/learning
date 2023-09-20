#!/bin/bash

chmod +775 testDll

echo ""
echo "======================================"
echo " 请先使用sudo dmesg |grep tty查看串口"
echo "然后配置rapicomm.ini中串口号为正确串口"
echo "======================================"
echo ""

# 获取脚本所在目录的完整路径
script_dir="$(cd "$(dirname "$0")" && pwd)"
# 切换工作目录到脚本所在目录
cd "$script_dir"
# 构建完整路径，假设testDLL与脚本在同一目录下
testDll="$script_dir/testDll"

# 检查当前用户是否已经具有 root 权限
if [ "$(id -u)" -ne 0 ]; then
    echo "当前用户不具有 root 权限。"

    # 提示用户输入密码以提升权限
    echo "请输入您的密码以获得 root 权限："
    read -s password

    # 使用sudo提升权限并检查是否成功
    if echo "$password" | sudo -S id >/dev/null 2>&1; then
        echo "权限提升成功。"
        # 使用绝对路径来运行 testDll
        sudo "$testDll"
        exit 1
    else
        echo "权限提升失败，请检查密码或用户是否具有 sudo 权限。"
        exit 1
    fi

    # 清除密码变量
    unset password
fi

# 如果脚本运行到这里，说明已经获得了 root 权限
echo "现在具有 root 权限，可以执行需要 root 权限的操作。"
echo "执行 testDll 以 root 权限..."
"$testDll"

# 脚本结束
echo "脚本执行完毕。"
