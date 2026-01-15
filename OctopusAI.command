#!/bin/bash
# OctopusAI 启动脚本 for Mac OS
# 双击此文件即可启动应用

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查Java是否安装
if ! command -v java &> /dev/null; then
    echo "错误: 未找到Java运行环境"
    echo "请安装Java 11或更高版本"
    echo "下载地址: https://www.oracle.com/java/technologies/downloads/"
    read -p "按任意键退出..."
    exit 1
fi

# 检查VideoAgent.jar是否存在
if [ ! -f "VideoAgent.jar" ]; then
    echo "错误: 未找到VideoAgent.jar文件"
    echo "请确保VideoAgent.jar与此脚本在同一目录下"
    read -p "按任意键退出..."
    exit 1
fi

# 显示启动信息
echo "================================"
echo "  OctopusAI - AI应用平台"
echo "================================"
echo "正在启动应用..."
echo ""

# 启动应用
java -jar VideoAgent.jar

# 如果应用异常退出，暂停以查看错误信息
if [ $? -ne 0 ]; then
    echo ""
    echo "应用异常退出"
    read -p "按任意键退出..."
fi
