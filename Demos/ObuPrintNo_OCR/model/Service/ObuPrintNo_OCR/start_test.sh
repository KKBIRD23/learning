#!/bin/bash
# start_test.sh (v7.3 - Final Compatible Edition)

# =================================================================
# OBU-OCR 服务开发测试环境启动脚本
# 特性: 跨版本兼容, 自动更新版本, 自动清理, 强制重新构建
# =================================================================

set -e

# --- 变量与颜色定义 ---
COMPOSE_FILE="docker-compose.test.yml"
IMAGE_NAME="obu-ocr-service-dev"
VERSION_FILE="version.txt"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "--- OBU-OCR Development & Test Start Script ---"

# --- 权限检查 ---
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}"
   exit 1
fi

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

# --- 1. 严格的版本检查 ---
echo "正在检查版本文件: ${VERSION_FILE}..."
if [ ! -f "$VERSION_FILE" ]; then
    echo -e "${RED}错误: 版本文件 '${VERSION_FILE}' 未找到！${NC}"
    exit 1
fi
VERSION_FROM_FILE=$(cat "$VERSION_FILE" | tr -d '[:space:]')
if [ -z "$VERSION_FROM_FILE" ]; then
    echo -e "${RED}错误: 版本文件 '${VERSION_FILE}' 为空或格式不正确！${NC}"
    exit 1
fi
echo -e "版本检查通过，目标版本: ${GREEN}${VERSION_FROM_FILE}${NC}"

# --- 2. 自动更新 docker-compose.test.yml ---
echo "正在更新 ${COMPOSE_FILE} 中的镜像版本..."
if ! sed -i "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${VERSION_FROM_FILE}|g" "${COMPOSE_FILE}"; then
    echo -e "${RED}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"
    exit 1
fi
echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"

# --- 3. 自动清理过期的悬空镜像 ---
echo "-----------------------------------------------------"
echo "正在自动清理过期的(dangling)镜像..."
DANGLING_IMAGES=$(sudo docker images -f "dangling=true" -q)
if [ -n "$DANGLING_IMAGES" ]; then
    echo "找到以下过期镜像，将进行清理："
    sudo docker images -f "dangling=true"
    sudo docker image prune -f
    echo -e "${GREEN}清理完成。${NC}"
else
    echo "没有需要清理的过期镜像。"
fi
echo "-----------------------------------------------------"

# --- 4. 启动服务 ---
echo -e "准备构建并启动镜像: ${GREEN}${IMAGE_NAME}:${VERSION_FROM_FILE}${NC}"
# 使用检测到的命令
sudo $COMPOSE_CMD -f ${COMPOSE_FILE} up --build --remove-orphans