#!/bin/bash

# ==========================================================
# OBU-OCR 服务 - 交互式运维菜单脚本
# ==========================================================

# 在脚本开头设置-e，确保任何一步出错，脚本都会立刻停止
set -e

# --- 变量定义 ---
COMPOSE_FILE="docker-compose.prod.yml"
IMAGE_NAME="obu-ocr-service"
DEPLOY_PACKAGE_PATTERN="OBU-OCR-DEPLOY-v*.tar.gz"
IMAGE_FILE_PATTERN="obu-ocr-service-v*-temp.tar"

# --- 颜色定义 ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- 在执行任何操作前，检查脚本是否以root权限运行 ---
if [[ $EUID -ne 0 ]]; then
   echo -e "${YELLOW}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}" 
   echo -e "${YELLOW}请尝试使用: sudo ./ops_menu.sh${NC}"
   exit 1
fi

# --- 功能函数定义 ---

# 显示主菜单
display_menu() {
    clear
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${GREEN}        OBU-OCR 服务 - 运维操作菜单 (v2.0)           ${NC}"
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${CYAN}  1. 解包并加载新版本 (智能检测部署包)            ${NC}"
    echo -e "${CYAN}  2. 停止服务 (docker compose down)                 ${NC}"
    echo -e "${CYAN}  3. 启动服务 (docker compose up)                 ${NC}"
    echo -e "${CYAN}  4. 清理无用镜像 (docker image prune)            ${NC}"
    echo -e "-----------------------------------------------------"
    echo -e "${CYAN}  5. 查看当前服务状态                             ${NC}"
    echo -e "${CYAN}  6. 查看实时日志                                 ${NC}"
    echo -e "-----------------------------------------------------"
    echo -e "${YELLOW}  0. 退出菜单                                     ${NC}"
    echo -e "${GREEN}=====================================================${NC}"
}

# 1. 智能部署新版本
deploy_new_version() {
    echo "--- 1. 解包、加载并配置新版本 ---"

    # --- 阶段一: 选择并解压部署包 ---
    echo "正在搜索部署包 (${DEPLOY_PACKAGE_PATTERN})..."
    mapfile -t packages < <(find . -maxdepth 1 -name "${DEPLOY_PACKAGE_PATTERN}" | sort -V)
    local package_count=${#packages[@]}
    local chosen_package=""

    if (( package_count == 0 )); then
        echo -e "${YELLOW}错误: 在当前目录下未找到任何部署包 (${DEPLOY_PACKAGE_PATTERN})。${NC}"
        return 1
    elif (( package_count == 1 )); then
        chosen_package=${packages[0]}
        echo -e "已自动选择唯一的部署包: ${GREEN}${chosen_package}${NC}"
    else
        echo "找到多个部署包，请选择一个进行解压:"
        for i in "${!packages[@]}"; do
            echo "  $(($i + 1))) ${packages[$i]}"
        done
        read -p "请输入您的选择 [1-${package_count}]: " choice
        if [[ $choice =~ ^[0-9]+$ ]] && (( choice > 0 && choice <= package_count )); then
            chosen_package=${packages[$((choice - 1))]}
        else
            echo -e "${YELLOW}无效的选择。操作已取消。${NC}"
            return 1
        fi
    fi

    echo "准备解压部署包: ${chosen_package}..."
    if tar -xzvf "${chosen_package}" -m; then
        echo -e "${GREEN}部署包解压成功！文件已更新。${NC}"
    else
        echo -e "${YELLOW}错误: 解压部署包失败！${NC}"
        return 1
    fi

    # --- 阶段二: 读取新版本号 ---
    echo ""
    echo "正在读取版本文件 (version.txt)..."
    local version_file="version.txt"
    if [ ! -f "$version_file" ]; then
        echo -e "${YELLOW}错误: 解压后未找到版本文件 'version.txt'。${NC}"
        return 1
    fi
    # 读取版本号，并去除所有可能的空格或换行符
    local new_version=$(cat "$version_file" | tr -d '[:space:]')
    if [ -z "$new_version" ]; then
        echo -e "${YELLOW}错误: 'version.txt' 文件为空或格式不正确。${NC}"
        return 1
    fi
    echo -e "检测到新版本号: ${GREEN}${new_version}${NC}"

    # --- 阶段三: 更新 docker-compose.prod.yml ---
    echo "正在更新 ${COMPOSE_FILE} 中的镜像版本..."
    # 使用sed进行安全、精准的替换
    # s|image: obu-ocr-service:.*|image: obu-ocr-service:new_version|g
    if sed -i "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${new_version}|g" "${COMPOSE_FILE}"; then
        echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"
    else
        echo -e "${YELLOW}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"
        return 1
    fi

    # --- 阶段四: 选择并加载镜像文件 ---
    echo ""
    echo "正在从解压后的文件中搜索镜像文件 (${IMAGE_FILE_PATTERN})..."
    mapfile -t images < <(find . -maxdepth 1 -name "${IMAGE_FILE_PATTERN}" | sort -V)
    local image_count=${#images[@]}
    local chosen_image=""

    if (( image_count == 0 )); then
        echo -e "${YELLOW}错误: 解压后未找到任何镜像文件 (${IMAGE_FILE_PATTERN})。${NC}"
        return 1
    elif (( image_count == 1 )); then
        chosen_image=${images[0]}
        echo -e "已自动选择唯一的镜像文件: ${GREEN}${chosen_image}${NC}"
    else
        echo "找到多个镜像文件，请选择一个进行加载:"
        for i in "${!images[@]}"; do
            echo "  $(($i + 1))) ${images[$i]}"
        done
        read -p "请输入您的选择 [1-${image_count}]: " choice
        if [[ $choice =~ ^[0-9]+$ ]] && (( choice > 0 && choice <= image_count )); then
            chosen_image=${images[$((choice - 1))]}
        else
            echo -e "${YELLOW}无效的选择。操作已取消。${NC}"
            return 1
        fi
    fi

    echo "准备加载Docker镜像: ${chosen_image}，请稍候..."
    if sudo docker load -i "${chosen_image}"; then
        echo -e "${GREEN}镜像加载成功！新版本部署准备就绪。${NC}"
        echo -e "${YELLOW}请注意: 服务当前可能仍在运行旧版本。请使用菜单选项 '2. 停止服务' 和 '3. 启动服务' 来应用新版本。${NC}"
    else
        echo -e "${YELLOW}错误: 加载Docker镜像失败！${NC}"
        return 1
    fi
}

# --- 其他函数 (stop_service, start_service 等) 保持不变 ---
stop_service() {
    echo "--- 2. 停止服务 ---"
    echo "正在执行 'docker compose down'..."
    sudo docker compose -f ${COMPOSE_FILE} down
    echo -e "${GREEN}服务已成功停止。${NC}"
}

start_service() {
    echo "--- 3. 启动服务 ---"
    echo "正在执行 'docker compose up -d'..."
    sudo docker compose -f ${COMPOSE_FILE} up -d
    echo -e "${GREEN}服务已成功启动！${NC}"
    echo "当前服务状态："
    sudo docker compose -f ${COMPOSE_FILE} ps
}

prune_images() {
    echo "--- 4. 清理无用镜像 ---"
    echo "以下是将被清理的悬空(dangling)镜像："
    sudo docker images -f "dangling=true"
    read -p "确认要清理以上所有镜像吗? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在执行 'docker image prune'..."
        sudo docker image prune -f
        echo -e "${GREEN}清理完成。${NC}"
    else
        echo "操作已取消。"
    fi
}

display_status() {
    echo "--- 5. 查看当前服务状态 ---"
    echo ">>> 容器运行状态 (ps):"
    sudo docker compose -f ${COMPOSE_FILE} ps
    echo ""
    echo ">>> 相关镜像列表 (images):"
    sudo docker images ${IMAGE_NAME}
}

view_logs() {
    echo "--- 6. 查看实时日志 (按 Ctrl+C 退出) ---"
    sudo docker compose -f ${COMPOSE_FILE} logs -f
}


# --- 主循环 ---
while true; do
    display_menu
    read -p "请输入您的选择 [0-6]: " choice

    case $choice in
        1) deploy_new_version ;;
        2) stop_service ;;
        3) start_service ;;
        4) prune_images ;;
        5) display_status ;;
        6) view_logs ;;
        0) echo "感谢使用，再见！"; break ;;
        *) echo -e "${YELLOW}无效的输入，请输入 0-6 之间的数字。${NC}" ;;
    esac

    if [[ "$choice" != "6" && "$choice" != "0" ]]; then
      echo ""
      read -p "按回车键返回主菜单..."
    fi
done
