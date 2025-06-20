#!/bin/bash
# start_prod.sh (v27.1 - Final Professional Edition)

# =================================================================
# OBU-OCR 服务生产环境启动脚本
# 架构: 自动化、健壮、可预测
# 特性:
# 1. 严格的版本检查与配置文件的物理更新。
# 2. 自动化的资源限制建议（以注释形式写入配置文件）。
# 3. 统一的、可靠的启动流程。
# =================================================================

# 在脚本开头设置-e，确保任何一步出错，脚本都会立刻停止
set -e

# --- 变量与颜色定义 ---
COMPOSE_FILE="docker-compose.prod.yml"
IMAGE_NAME="obu-ocr-service"
VERSION_FILE="version.txt"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "--- OBU-OCR Production Start Script (Final Edition) ---"

# --- 在执行任何操作前，检查脚本是否以root权限运行 ---
if [[ $EUID -ne 0 ]]; then
   echo -e "${YELLOW}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}"
   echo -e "${YELLOW}请尝试使用: sudo ./ops_menu.sh${NC}"
   exit 1
fi

# --- 步骤 1: 严格的版本检查 ---
echo "正在检查版本文件: ${VERSION_FILE}..."
if [ ! -f "$VERSION_FILE" ]; then
    echo -e "${YELLOW}错误: 版本文件 '${VERSION_FILE}' 未找到！${NC}"
    exit 1
fi
NEW_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
if [ -z "$NEW_VERSION" ]; then
    echo -e "${YELLOW}错误: 版本文件 '${VERSION_FILE}' 为空或格式不正确！${NC}"
    exit 1
fi
echo -e "版本检查通过，目标版本: ${GREEN}${NEW_VERSION}${NC}"

# --- 步骤 2: 检查目标镜像是否存在 ---
TARGET_IMAGE="${IMAGE_NAME}:${NEW_VERSION}"
echo "正在检查目标镜像: ${TARGET_IMAGE}..."
if ! sudo docker image inspect "${TARGET_IMAGE}" >/dev/null 2>&1; then
  echo -e "${YELLOW}错误: 目标镜像 '${TARGET_IMAGE}' 未找到。请先使用ops_menu.sh加载镜像。${NC}"
  exit 1
fi
echo "镜像检查通过。"

# --- 步骤 3: 核心修正 - 自动更新 docker-compose.prod.yml ---
echo "正在更新 ${COMPOSE_FILE} 中的镜像版本..."
# 使用sed进行安全、精准的替换
if sudo sed -i "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${NEW_VERSION}|g" "${COMPOSE_FILE}"; then
    echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"
else
    echo -e "${YELLOW}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"
    exit 1
fi

# --- 步骤 4: 动态资源建议 (可选，但非常专业) ---
# 这部分不再导出环境变量，而是可以用于提醒运维人员
TOTAL_CORES=$(nproc)
TOTAL_MEM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
RECOMMENDED_CORES=$((TOTAL_CORES - 1))
RECOMMENDED_MEM_RESERVATION=$(( (TOTAL_MEM_MB - 1024) * 8 / 10 ))
RECOMMENDED_MEM_LIMIT=$((TOTAL_MEM_MB - 1024))

echo "-----------------------------------------------------"
echo -e "${CYAN}系统资源检测与建议:${NC}"
echo "  服务器总核心数: ${TOTAL_CORES}"
echo "  服务器总内存: ${TOTAL_MEM_MB} MB"
echo "  ${YELLOW}建议在 ${COMPOSE_FILE} 中配置的 cpuset 为: '0-${RECOMMENDED_CORES}'${NC}"
echo "  ${YELLOW}建议配置的 mem_reservation/mem_limit 为: ${RECOMMENDED_MEM_RESERVATION}m / ${RECOMMENDED_MEM_LIMIT}m${NC}"
echo "  (当前脚本将使用 ${COMPOSE_FILE} 中已有的固化配置)"
echo "-----------------------------------------------------"

# --- 步骤 5: 启动服务 ---
echo "准备启动服务... 在 byobu/tmux 中运行，按 Ctrl+C 可安全停止。"
# 我们不再依赖任何外部环境变量，只信任被修改后的compose文件
sudo docker compose -f ${COMPOSE_FILE} up --remove-orphans
