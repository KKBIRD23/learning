#!/bin/bash
# start_prod.sh (v27.2 - Final Compatible Edition)

# =================================================================
# OBU-OCR 服务生产环境启动脚本
# 架构: 自动化、健壮、可预测、跨版本兼容
# =================================================================

set -e

# --- 变量与颜色定义 ---
COMPOSE_FILE="docker-compose.prod.yml"
IMAGE_NAME="obu-ocr-service"
VERSION_FILE="version.txt"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "--- OBU-OCR Production Start Script (Final Compatible Edition) ---"

# --- 权限检查 ---
#if [[ $EUID -ne 0 ]]; then
#   echo -e "${RED}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}"
#   exit 1
#fi

# --- 【核心修正】智能检测并设置正确的 docker compose 命令 ---
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}错误: 未找到 'docker-compose' 或 'docker compose' 命令。${NC}"
    exit 1
fi
echo -e "检测到并使用命令: ${GREEN}${COMPOSE_CMD}${NC}"
# --- 修正结束 ---

# --- 步骤 1: 严格的版本检查 ---
echo "正在检查版本文件: ${VERSION_FILE}..."
if [ ! -f "$VERSION_FILE" ]; then
    echo -e "${RED}错误: 版本文件 '${VERSION_FILE}' 未找到！${NC}"
    exit 1
fi
NEW_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}错误: 版本文件 '${VERSION_FILE}' 为空或格式不正确！${NC}"
    exit 1
fi
echo -e "版本检查通过，目标版本: ${GREEN}${NEW_VERSION}${NC}"

# --- 步骤 2: 检查目标镜像是否存在 ---
TARGET_IMAGE="${IMAGE_NAME}:${NEW_VERSION}"
echo "正在检查目标镜像: ${TARGET_IMAGE}..."
if ! sudo docker image inspect "${TARGET_IMAGE}" >/dev/null 2>&1; then
  echo -e "${RED}错误: 目标镜像 '${TARGET_IMAGE}' 未找到。请先使用load_img.sh加载镜像。${NC}"
  exit 1
fi
echo "镜像检查通过。"

# --- 3.精确替换版本号 ---
echo "正在更新 ${COMPOSE_FILE} 中的镜像版本..."
# 构造一个100%格式正确的新行
new_image_line="    image: obu-ocr-service:${NEW_VERSION}"
echo ${new_image_line}
# 使用'c\'命令，找到以'image:'开头的行，并用新行将其完全替换
# 这是最健壮、最不依赖于原始文件格式的方法
if ! sudo sed -i "/^[[:space:]]*image:/c\\${new_image_line}" "${COMPOSE_FILE}"; then
    echo -e "${RED}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"; return 1;
fi
echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"

# --- 步骤 4: 动态资源建议 ---
TOTAL_CORES=$(nproc)
TOTAL_MEM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
RECOMMENDED_CORES=$((TOTAL_CORES > 1 ? TOTAL_CORES - 1 : 1))
RECOMMENDED_MEM_RESERVATION=$(( (TOTAL_MEM_MB - 1024) * 8 / 10 ))
RECOMMENDED_MEM_LIMIT=$((TOTAL_MEM_MB - 1024))
echo "-----------------------------------------------------"
echo -e "${CYAN}系统资源检测与建议:${NC}"
echo -e " ${GREEN} 服务器总核心数:${NC} ${TOTAL_CORES}"
echo -e " ${GREEN} 服务器总内存:${NC} ${TOTAL_MEM_MB} MB"
echo -e "  ${YELLOW}建议在 ${COMPOSE_FILE} 中配置的 cpuset 为: '0-$((RECOMMENDED_CORES - 1))'${NC}"
echo -e "  ${YELLOW}建议配置的 mem_reservation/mem_limit 为: ${RECOMMENDED_MEM_RESERVATION}m / ${RECOMMENDED_MEM_LIMIT}m${NC}"
echo -e "  (当前脚本将使用 ${COMPOSE_FILE} 中已有的固化配置)"
echo "-----------------------------------------------------"

# --- 步骤 5: 启动服务 ---
echo "准备启动服务... 在 byobu/tmux 中运行，按 Ctrl+C 可安全停止。"
# 使用检测到的命令
sudo $COMPOSE_CMD -f ${COMPOSE_FILE} up --remove-orphans
