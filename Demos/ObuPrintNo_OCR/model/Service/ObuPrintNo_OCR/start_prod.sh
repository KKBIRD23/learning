#!/bin/bash
# start_prod.sh (FINAL - Original Architecture with Dynamic Resources)

# =================================================================
# OBU-OCR 服务生产环境启动脚本 (v27.0 - 最终稳定版)
# 架构: Waitress 单进程 + 内部多进程OCR池
# 特性: 动态计算并分配CPU和内存资源，以保护主机并保障性能
# =================================================================

echo "--- OBU-OCR Production Start Script (Stable Architecture) ---"

# --- 步骤 1: 从 version.txt 读取版本号 ---
if [ ! -s "version.txt" ]; then
    echo "错误: version.txt 文件未找到或为空。"
    exit 1
fi
export IMAGE_VERSION=$(head -n 1 version.txt | tr -d '[:space:]')
echo "目标镜像: obu-ocr-service:${IMAGE_VERSION}"

# --- 步骤 2: 检查镜像 ---
docker image inspect obu-ocr-service:${IMAGE_VERSION} >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo -e "\033[0;31m错误: 目标镜像 'obu-ocr-service:${IMAGE_VERSION}' 未找到。\033[0m"
  exit 1
fi
echo "镜像检查通过。"

# --- 步骤 3: 动态计算CPU和内存分配 ---
TOTAL_CORES=$(nproc)
TOTAL_MEM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
echo "检测到服务器总核心数: $TOTAL_CORES, 总内存: $TOTAL_MEM_MB MB"

# [CPU分配策略]
# 如果核心数大于1，则预留最后一个核心给操作系统，其余全部分配给应用。
# 这能保证即使应用CPU满载，服务器依然能响应SSH等管理操作。
if [ "$TOTAL_CORES" -gt 1 ]; then
  export APP_CORES_CPUSET="0-$(($TOTAL_CORES - 2))"
else
  export APP_CORES_CPUSET="0"
fi

# [内存分配策略]
# 将总内存减去1GB(1024MB)作为系统保留，其余都给应用作为硬上限。
# 将硬上限的80%作为软性保证内存。
export APP_MEM_LIMIT="$(($TOTAL_MEM_MB - 1024))m"
export APP_MEM_RESERVATION="$(($(($TOTAL_MEM_MB - 1024)) * 8 / 10))m"

echo "为App分配CPU核心: $APP_CORES_CPUSET"
echo "为App分配保障/上限内存: $APP_MEM_RESERVATION / $APP_MEM_LIMIT"

# --- 步骤 4: 启动服务 ---
echo "准备启动服务... 在 byobu/tmux 中运行，按 Ctrl+C 可安全停止。"
docker compose -f docker-compose.prod.yml up --remove-orphans