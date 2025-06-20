#!/bin/bash

# =================================================================
# OBU-OCR 服务镜像与部署包一体化打包脚本 (v29.0 - 最终交付版)
# 设计师: Mr.Wang
# 职责: 创建一个包含所有生产部署所需文件的、单一的交付压缩包。
# =================================================================

echo "--- OBU-OCR All-in-One Packaging Script ---"

# --- 步骤 1: 从 version.txt 读取版本号 ---
if [ ! -s "version.txt" ]; then
    echo "错误: version.txt 文件未找到或为空。"
    exit 1
fi
export IMAGE_VERSION=$(head -n 1 version.txt | tr -d '[:space:]')

# --- 步骤 2: 定义所有文件名 ---
IMAGE_NAME="obu-ocr-service"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_VERSION}"
# 这是一个临时的、中间状态的镜像压缩包名
IMAGE_TARBALL_NAME_TEMP="${IMAGE_NAME}-${IMAGE_VERSION}-temp.tar"
# 这是我们最终的、唯一的交付物
FINAL_BUNDLE_NAME="OBU-OCR-DEPLOY-${IMAGE_VERSION}.tar.gz"

echo "目标镜像: ${FULL_IMAGE_NAME}"
echo "最终交付包名: ${FINAL_BUNDLE_NAME}"

# --- 步骤 3: 检查目标镜像是否存在 ---
echo "正在检查本地是否存在镜像 ${FULL_IMAGE_NAME}..."
docker image inspect ${FULL_IMAGE_NAME} >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo -e "\033[0;31m错误: 目标镜像 '${FULL_IMAGE_NAME}' 未找到。\033[0m"
  echo "请先运行 './start_test.sh' 成功构建并测试该版本的镜像。"
  exit 1
fi
echo "镜像检查通过。"

# --- 步骤 4: 打包Docker镜像为临时文件 ---
echo "正在将镜像保存到临时文件: ${IMAGE_TARBALL_NAME_TEMP}..."
docker save -o ${IMAGE_TARBALL_NAME_TEMP} ${FULL_IMAGE_NAME}
if [ $? -ne 0 ]; then
  echo -e "\033[0;31m错误: Docker镜像打包失败！\033[0m"
  exit 1
fi
echo "Docker镜像打包成功。"

# --- 步骤 5: 创建最终的一体化交付包 ---
echo "正在创建一体化交付包: ${FINAL_BUNDLE_NAME}..."
# 将巨大的镜像tar包和所有轻量级的部署文件，一起打包进最终的压缩包
tar -czvf ${FINAL_BUNDLE_NAME} \
    ${IMAGE_TARBALL_NAME_TEMP} \
    config.py \
    docker-compose.prod.yml \
    start_prod.sh \
    load_img.py \
    version.txt \
    model \
    .env
if [ $? -ne 0 ]; then
  echo -e "\033[0;31m错误: 最终交付包创建失败！\033[0m"
  # 如果最终包创建失败，也删除已创建的临时镜像tar包
  rm -f ${IMAGE_TARBALL_NAME_TEMP}
  exit 1
fi
echo "最终交付包创建成功。"

# --- 步骤 6: 清理临时的镜像tar包 ---
# 既然所有的东西都已经在最终的交付包里了，这个临时的中间文件就可以删除了。
echo "正在清理临时的镜像tar包: ${IMAGE_TARBALL_NAME_TEMP}..."
rm -f ${IMAGE_TARBALL_NAME_TEMP}
echo "清理完成。"

# --- 成功收尾 ---
echo -e "\033[0;32m\n所有打包任务已成功完成！\033[0m"
echo "--------------------------------------------------"
echo "现在只需要交付一个核心产物:"
echo "最终交付包 (大文件): ${FINAL_BUNDLE_NAME}"
echo ""
echo "下一步操作建议:"
echo "1. 将这一个文件上传到生产服务器。"
echo "2. 在生产服务器上，先解压 'tar -xzvf ${FINAL_BUNDLE_NAME}'。"
echo "3. 然后执行 'docker load -i ${IMAGE_TARBALL_NAME_TEMP}'。"
echo "4. 最后运行 './start_prod.sh'。"
echo "--------------------------------------------------"
