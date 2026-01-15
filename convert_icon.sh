#!/bin/bash
# 图标转换脚本 - 将 icon.ico 转换为 Mac OS .icns 格式
# 需要: ImageMagick 或 iconutil (Mac OS 自带)

ICON_SOURCE="icon.ico"
ICON_OUTPUT="OctopusAI.icns"
ICON_TEMP_DIR="icon_temp"

echo "正在转换图标..."

# 检查源文件是否存在
if [ ! -f "$ICON_SOURCE" ]; then
    echo "警告: 未找到 $ICON_SOURCE"
    echo "跳过图标转换"
    exit 0
fi

# 方法1: 使用 iconutil (Mac OS 自带)
if command -v iconutil &> /dev/null; then
    echo "使用 iconutil 转换图标..."

    # 创建临时目录
    mkdir -p "$ICON_TEMP_DIR/iconset"

    # 使用 sips 转换不同尺寸 (Mac OS 自带)
    sizes=(16 32 128 256 512 1024)
    for size in "${sizes[@]}"; do
        sips -z $size $size "$ICON_SOURCE" --out "$ICON_TEMP_DIR/iconset/icon_${size}x${size}.png" 2>/dev/null
        sips -z $size $size "$ICON_SOURCE" --out "$ICON_TEMP_DIR/iconset/icon_${size}x${size}@2x.png" 2>/dev/null
    done

    # 生成 .icns 文件
    iconutil -c icns "$ICON_TEMP_DIR/iconset" -o "$ICON_OUTPUT"

    # 清理临时文件
    rm -rf "$ICON_TEMP_DIR"

    if [ -f "$ICON_OUTPUT" ]; then
        echo "✅ 图标转换成功: $ICON_OUTPUT"
        exit 0
    fi
fi

# 方法2: 使用 ImageMagick
if command -v convert &> /dev/null; then
    echo "使用 ImageMagick 转换图标..."

    # 创建临时目录
    mkdir -p "$ICON_TEMP_DIR"

    # 转换不同尺寸
    convert "$ICON_SOURCE" -resize 16x16 "$ICON_TEMP_DIR/icon_16x16.png"
    convert "$ICON_SOURCE" -resize 32x32 "$ICON_TEMP_DIR/icon_16x16@2x.png"
    convert "$ICON_SOURCE" -resize 32x32 "$ICON_TEMP_DIR/icon_32x32.png"
    convert "$ICON_SOURCE" -resize 64x64 "$ICON_TEMP_DIR/icon_32x32@2x.png"
    convert "$ICON_SOURCE" -resize 128x128 "$ICON_TEMP_DIR/icon_128x128.png"
    convert "$ICON_SOURCE" -resize 256x256 "$ICON_TEMP_DIR/icon_128x128@2x.png"
    convert "$ICON_SOURCE" -resize 256x256 "$ICON_TEMP_DIR/icon_256x256.png"
    convert "$ICON_SOURCE" -resize 512x512 "$ICON_TEMP_DIR/icon_256x256@2x.png"
    convert "$ICON_SOURCE" -resize 512x512 "$ICON_TEMP_DIR/icon_512x512.png"
    convert "$ICON_SOURCE" -resize 1024x1024 "$ICON_TEMP_DIR/icon_512x512@2x.png"

    # 创建 iconset
    mkdir -p "$ICON_TEMP_DIR/iconset"
    mv "$ICON_TEMP_DIR"/*.png "$ICON_TEMP_DIR/iconset/" 2>/dev/null

    # 使用 iconutil 生成 .icns
    if command -v iconutil &> /dev/null; then
        iconutil -c icns "$ICON_TEMP_DIR/iconset" -o "$ICON_OUTPUT"
    fi

    # 清理临时文件
    rm -rf "$ICON_TEMP_DIR"

    if [ -f "$ICON_OUTPUT" ]; then
        echo "✅ 图标转换成功: $ICON_OUTPUT"
        exit 0
    fi
fi

# 方法3: 简单复制（如果以上方法都失败）
echo "警告: 无法转换图标"
echo "提示: 可以使用在线工具转换图标"
echo "推荐: https://cloudconvert.com/ico-to-icns"
echo ""
echo "或者手动操作:"
echo "1. 使用在线工具将 icon.ico 转换为 OctopusAI.icns"
echo "2. 将 OctopusAI.icns 放在当前目录"
echo "3. 重新运行创建脚本"

exit 1
