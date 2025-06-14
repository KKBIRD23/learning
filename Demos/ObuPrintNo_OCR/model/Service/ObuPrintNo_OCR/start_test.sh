#!/bin/bash

# =================================================================
# OBU-OCR 服务开发测试环境启动脚本 (v7.0 - 钻石版)
# =================================================================

echo "--- OBU-OCR Development & Test Start Script ---"

# --- 版本控制中心 (带自动回退) ---
# 1. 定义一个默认的回退版本号
DEFAULT_IMAGE_VERSION="v2.0"
# 2. 尝试从 version.txt 读取版本号
if [ -f "version.txt" ]; then
    # 读取文件第一行并移除所有空白字符
    VERSION_FROM_FILE=$(head -n 1 version.txt | tr -d '[:space:]')
    if [ -n "$VERSION_FROM_FILE" ]; then
        echo "找到 version.txt, 使用版本: $VERSION_FROM_FILE"
        export IMAGE_VERSION=$VERSION_FROM_FILE
    else
        echo "警告: version.txt 为空, 使用默认版本: $DEFAULT_IMAGE_VERSION"
        export IMAGE_VERSION=$DEFAULT_IMAGE_VERSION
    fi
else
    echo "未找到 version.txt, 使用默认版本: $DEFAULT_IMAGE_VERSION"
    export IMAGE_VERSION=$DEFAULT_IMAGE_VERSION
fi

echo "准备构建并测试镜像: obu-ocr-service:${IMAGE_VERSION}"
docker compose -f docker-compose.test.yml up --build --remove-orphans