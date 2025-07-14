#!/bin/bash

# ==============================================================================
# OBU-OCR 服务 - 交互式运维菜单脚本 (v2.4 - 最终交付版)
# ==============================================================================

# 脚本出错时立刻停止
set -e

# --- 变量与颜色定义 ---
COMPOSE_FILE="docker-compose.prod.yml"
IMAGE_NAME="obu-ocr-service"
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- 权限检查 ---
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}"
   echo -e "${YELLOW}请尝试使用: sudo $0${NC}"
   exit 1
fi

# --- 新旧命令兼容性检测 ---
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}错误: 未找到 'docker-compose' 或 'docker compose' 命令。${NC}"
    exit 1
fi

# ==============================================================================
# 【智能预警】CPU 配置检查函数 (只检查，不修改)
# ==============================================================================
check_cpu_configuration() {
    echo "--- 正在进行CPU配置预警检查 ---"
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${YELLOW}警告: 未找到 ${COMPOSE_FILE} 文件，跳过CPU检查。${NC}"
        return
    fi
    local configured_cpuset=$(grep 'cpuset:' "$COMPOSE_FILE" | grep -v '^[[:space:]]*#' | awk -F"'" '{print $2}')
    if [ -z "$configured_cpuset" ]; then
        echo -e "${GREEN}检查通过: ${COMPOSE_FILE} 中未设置cpuset，将使用所有可用CPU。${NC}"
        return
    fi
    local host_cores=$(nproc)
    if ! [[ $configured_cpuset =~ ^0-([0-9]+)$ ]]; then
        echo -e "${RED}严重格式错误: ${COMPOSE_FILE} 中的 cpuset 值 '${configured_cpuset}' 格式不正确。${NC}"
        echo -e "${YELLOW}建议格式为 '0-X'。请手动修正后重试。${NC}"
        exit 1
    fi
    local max_cpu_index=${BASH_REMATCH[1]}
    if (( max_cpu_index >= host_cores )); then
        local last_valid_index=$((host_cores - 1))
        local recommended_app_cores=$((host_cores > 1 ? host_cores - 1 : 1))
        local recommended_max_index=$((recommended_app_cores - 1))
        echo -e "${RED}======================= 严重配置错误！ =======================${NC}"
        echo -e "${RED}原因: 您配置的 cpuset 为 '${configured_cpuset}', 但服务器只有 ${host_cores} 个核心 (0-${last_valid_index})。${NC}"
        echo -e "${CYAN}--- 修正建议 ---${NC}"
        echo -e "请手动编辑 ${GREEN}${COMPOSE_FILE}${NC} 文件，将 cpuset 的值修改为: ${GREEN}'0-${recommended_max_index}'${NC}"
        echo -e "${RED}================================================================${NC}"
        exit 1
    else
        echo -e "${GREEN}检查通过: cpuset 配置 '${configured_cpuset}' 在服务器可用范围内。${NC}"
    fi
}

# --- 在脚本启动时，立刻执行一次预警检查 ---
check_cpu_configuration
echo "-----------------------------------------------------"
sleep 1

# --- 功能函数 ---

# 显示主菜单 (完全恢复您的美观设计)
display_menu() {
    clear
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${GREEN}        OBU-OCR 服务 - 运维操作菜单 (v2.4)           ${NC}"
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${CYAN}  1. 解包并加载新版本                             ${NC}"
    echo -e "${CYAN}  2. 停止服务 (down)                              ${NC}"
    echo -e "${CYAN}  3. 启动/更新服务 (up)                           ${NC}"
    echo -e "${CYAN}  4. 清理无用镜像 (prune)                         ${NC}"
    echo -e "-----------------------------------------------------"
    echo -e "${CYAN}  5. 查看当前服务状态 (ps)                        ${NC}"
    echo -e "${CYAN}  6. 查看实时日志 (logs)                          ${NC}"
    echo -e "-----------------------------------------------------"
    echo -e "${YELLOW}  0. 退出菜单                                     ${NC}"
    echo -e "${GREEN}=====================================================${NC}"
}

# 1. 部署新版本 (v2.5 - 增加了自动清理功能)
deploy_new_version() {
    echo "--- 1. 解包、加载并配置新版本 ---"
    local deploy_package_pattern="OBU-OCR-DEPLOY-v*.tar.gz"
    local image_file_pattern="obu-ocr-service-v*.tar"
    local env_backup_file=".env.prod_backup_$(date +%s)" # 增加时间戳，确保备份唯一

    # --- 阶段一: 备份现有的生产 .env 文件 ---
    if [ -f ".env" ]; then
        echo "检测到现有的 .env 文件，正在进行安全备份..."
        cp -f .env "${env_backup_file}"
        echo -e "${GREEN}.env 文件已成功备份到 ${env_backup_file}${NC}"
    fi

    # --- 阶段二: 选择并解压部署包 ---
    echo "正在搜索部署包 (${deploy_package_pattern})..."
    mapfile -t packages < <(find . -maxdepth 1 -name "${deploy_package_pattern}" | sort -V)
    if (( ${#packages[@]} == 0 )); then
        echo -e "${RED}错误: 未找到部署包。${NC}"; return 1;
    fi
    local chosen_package=${packages[-1]}
    echo -e "已自动选择最新的部署包: ${GREEN}${chosen_package}${NC}"
    echo "准备解压..."
    if ! tar -xzvf "${chosen_package}" -m; then
        echo -e "${RED}错误: 解压部署包失败！${NC}"; return 1;
    fi
    echo -e "${GREEN}部署包解压成功！${NC}"

    # --- 【核心修正】在解包后，立刻将所有权交还给操作员 ---
    echo "正在修正解压后文件的所有权..."
    sudo chown -R "${original_user}:${original_user}" .
    echo -e "${GREEN}文件所有权已交还给用户 ${original_user}。${NC}"

    # --- 阶段三: 从备份恢复生产 .env 文件 ---
    if [ -f "${env_backup_file}" ]; then
        echo "正在从备份恢复生产环境的 .env 文件..."
        cp -f "${env_backup_file}" .env
        rm -f "${env_backup_file}" # 清理临时的备份文件
        echo -e "${GREEN}.env 文件已成功恢复。${NC}"
    fi

    # --- 阶段四: 精确替换版本号 ---
    local version_file="version.txt"
    if [ ! -f "$version_file" ]; then
        echo -e "${RED}错误: 未找到版本文件 'version.txt'。${NC}"; return 1;
    fi
    local new_version=$(cat "$version_file" | tr -d '[:space:]')
    if [ -z "$new_version" ]; then
        echo -e "${RED}错误: 'version.txt' 文件为空。${NC}"; return 1;
    fi
    echo -e "检测到新版本号: ${GREEN}${new_version}${NC}"
    echo "正在自动更新 ${COMPOSE_FILE} 中的镜像版本..."

    # 构造一个100%格式正确的新行
    new_image_line="    image: obu-ocr-service:${new_version}"

    # 使用'c\'命令，找到以'image:'开头的行，并用新行将其完全替换
    # 这是最健壮、最不依赖于原始文件格式的方法
    if ! sudo sed -i "/^[[:space:]]*image:/c\\${new_image_line}" "${COMPOSE_FILE}"; then
        echo -e "${RED}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"; return 1;
    fi
    echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"

    # --- 阶段五: 加载镜像文件 ---
    mapfile -t images < <(find . -maxdepth 1 -name "${image_file_pattern}" | sort -V)
    if (( ${#images[@]} == 0 )); then
        echo -e "${RED}错误: 解压后未找到镜像文件。${NC}"; return 1;
    fi
    local chosen_image=${images[-1]}
    echo -e "准备加载Docker镜像: ${GREEN}${chosen_image}${NC}，请稍候..."
    if ! sudo docker load -i "${chosen_image}"; then
        echo -e "${RED}错误: 加载Docker镜像失败！${NC}"; return 1;
    fi
    echo -e "${GREEN}镜像加载成功！新版本部署准备就绪。${NC}"

    # ==============================================================================
    # 【核心新增】部署成功后，进行交互式清理
    # ==============================================================================
    echo "-----------------------------------------------------"
    read -p "新版本已成功加载。是否要清理本次使用的部署包和镜像包? (y/n): " -n 1 -r
    echo # 换行
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在清理文件: ${chosen_package} 和 ${chosen_image}..."
        rm -f "${chosen_package}" "${chosen_image}"
        echo -e "${GREEN}清理完成。${NC}"
    else
        echo -e "${YELLOW}已跳过清理。部署包和镜像包仍保留在当前目录。${NC}"
    fi
    # ==============================================================================

    echo -e "${YELLOW}请使用菜单选项 '3. 启动/更新服务' 来应用新版本。${NC}"
}

# 2. 停止服务
stop_service() {
    echo "--- 2. 停止服务 ---"; sudo $COMPOSE_CMD -f ${COMPOSE_FILE} down --remove-orphans; echo -e "${GREEN}服务已成功停止。${NC}";
}

# 3. 启动/更新服务
start_service() {
    echo "--- 3. 启动/更新服务 ---"; sudo $COMPOSE_CMD -f ${COMPOSE_FILE} up --remove-orphans; echo -e "${GREEN}启动指令已发送！${NC}"; sleep 3; echo "当前服务状态："; sudo $COMPOSE_CMD -f ${COMPOSE_FILE} ps;
}

# 4. 清理镜像
prune_images() {
    echo "--- 4. 清理无用镜像 ---"; read -p "确认清理所有未使用镜像? (y/n): " -n 1 -r; echo; if [[ $REPLY =~ ^[Yy]$ ]]; then echo "正在清理..."; sudo docker image prune -a -f; echo -e "${GREEN}清理完成。${NC}"; else echo "操作已取消。"; fi
}

# 5. 查看状态
display_status() {
    echo "--- 5. 查看当前服务状态 ---"; sudo $COMPOSE_CMD -f ${COMPOSE_FILE} ps;
}

# 6. 查看日志
view_logs() {
    echo "--- 6. 查看实时日志 (按 Ctrl+C 退出) ---"; sudo $COMPOSE_CMD -f ${COMPOSE_FILE} logs -f --tail="200";
}

# --- 主循环 (恢复您的经典设计) ---
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
        *) echo -e "${RED}无效的输入，请输入 0-6 之间的数字。${NC}" ;;
    esac
    if [[ "$choice" != "6" && "$choice" != "0" ]]; then
      echo ""; read -p "按回车键返回主菜单...";
    fi
done
