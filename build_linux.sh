#!/bin/bash
# 在 Mac 上执行此脚本，生成可在 Linux 上运行的 .difypkg 包
# 依赖：Docker Desktop（确保已启动）

set -e

PLUGIN_DIR=$(pwd)
IMAGE="python:3.12-slim"
OUTPUT_NAME="skill_agent_linux.difypkg"

echo "==> 使用 Docker 在 Linux 环境中打包插件..."

docker run --rm \
  -v "$PLUGIN_DIR":/plugin \
  -w /plugin \
  "$IMAGE" \
  bash -c "
    set -e
    pip install dify-plugin --quiet
    dify plugin package . -o /plugin/$OUTPUT_NAME
  "

echo "==> 打包完成：$OUTPUT_NAME"
