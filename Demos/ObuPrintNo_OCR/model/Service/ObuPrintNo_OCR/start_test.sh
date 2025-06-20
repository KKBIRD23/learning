#!/bin/bash

# =================================================================
# OBU-OCR 服务开发测试环境启动脚本 (v7.2 - Final Professional Edition)
# =================================================================

set -e

# --- 变量与颜色定义 ---
COMPOSE_FILE="docker-compose.test.yml"
IMAGE_NAME="obu-ocr-service-dev"
VERSION_FILE="version.txt"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "--- OBU-OCR Development & Test Start Script ---"

# --- 在执行任何操作前，检查脚本是否以root权限运行 ---
if [[ $EUID -ne 0 ]]; then
   echo -e "${YELLOW}错误: 此脚本需要使用 sudo 或以 root 用户身份运行。${NC}"
   echo -e "${YELLOW}请尝试使用: sudo ./ops_menu.sh${NC}"
   exit 1
fi

# --- 1. 严格的版本检查 ---
echo "正在检查版本文件: ${VERSION_FILE}..."
if [ ! -f "$VERSION_FILE" ]; then
    echo -e "${YELLOW}错误: 版本文件 '${VERSION_FILE}' 未找到！${NC}"
    exit 1
fi
VERSION_FROM_FILE=$(cat "$VERSION_FILE" | tr -d '[:space:]')
if [ -z "$VERSION_FROM_FILE" ]; then
    echo -e "${YELLOW}错误: 版本文件 '${VERSION_FILE}' 为空或格式不正确！${NC}"
    exit 1
fi

echo -e "版本检查通过，目标版本: ${GREEN}${VERSION_FROM_FILE}${NC}"

# --- 2. 核心修正：自动更新 docker-compose.test.yml ---
echo "正在更新 ${COMPOSE_FILE} 中的镜像版本..."
# 使用sed进行安全、精准的替换
if sed -i "s|image: ${IMAGE_NAME}:.*|image: ${IMAGE_NAME}:${VERSION_FROM_FILE}|g" "${COMPOSE_FILE}"; then
    echo -e "${GREEN}${COMPOSE_FILE} 更新成功！${NC}"
else
    echo -e "${YELLOW}错误: 更新 ${COMPOSE_FILE} 失败！${NC}"
    exit 1
fi

# --- 3. 新增：自动清理过期的悬空镜像 ---
echo "-----------------------------------------------------"
echo "正在自动清理过期的(dangling)镜像..."
# 使用-q参数只输出ID，如果没有悬空镜像，则什么都不输出
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

echo "-----------------------------------------------------"
echo -e "准备构建并启动镜像: ${GREEN}${IMAGE_NAME}:${VERSION_FROM_FILE}${NC}"
echo "-----------------------------------------------------"

# --- 4. 启动服务 ---
# --build: 强制每次都重新构建，确保代码最新
# --remove-orphans: 清理旧的、不再使用的容器
sudo docker compose -f ${COMPOSE_FILE} up --build --remove-orphans
