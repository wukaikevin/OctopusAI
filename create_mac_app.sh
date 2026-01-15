#!/bin/bash
# 创建 Mac OS .app 应用程序包
# 使用方法: ./create_mac_app.sh

APP_NAME="OctopusAI"
APP_FILE="${APP_NAME}.app"
CONTENTS_DIR="${APP_FILE}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

echo "正在创建 ${APP_FILE}..."

# 创建目录结构
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# 创建 Info.plist
cat > "${CONTENTS_DIR}/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleIdentifier</key>
    <string>com.octopusai.app</string>
    <key>CFBundleName</key>
    <string>OctopusAI</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>OctopusAI</string>
    <key>JVMRuntime</key>
    <string>jdk-11.0.17.jdk</string>
    <key>JVMMainClassName</key>
    <string>-jar</string>
    <key>JVMOptions</key>
    <array>
        <string>-Xmx4096M</string>
        <string>-Dfile.encoding=UTF-8</string>
    </array>
    <key>JVMArguments</key>
    <array>
        <string>VideoAgent.jar</string>
    </array>
</dict>
</plist>
EOF

# 创建启动脚本
cat > "${MACOS_DIR}/launch" << 'EOF'
#!/bin/bash
# OctopusAI 应用启动脚本

# 获取应用程序所在目录
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="${APP_DIR%/*}"  # 去掉 /MacOS
APP_ROOT="${APP_DIR%/*}"  # 去掉 /Contents

# 设置工作目录为应用程序根目录
cd "$APP_ROOT"

# 检查Java是否安装
if ! command -v java &> /dev/null; then
    osascript << 'APPLESCRIPT'
tell application "System Events"
    display dialog "未找到Java运行环境。\n\n请安装Java 11或更高版本以运行OctopusAI。\n\n下载地址:\nhttps://www.oracle.com/java/technologies/downloads/" buttons {"确定"} default button "OK" with icon caution with title "OctopusAI - 错误"
end tell
APPLESCRIPT
    exit 1
fi

# 启动Java应用
exec java -Xmx4096M -Dfile.encoding=UTF-8 -jar VideoAgent.jar
EOF

# 设置执行权限
chmod +x "${MACOS_DIR}/launch"

# 处理图标
echo "正在处理图标..."
if [ -f "OctopusAI.icns" ]; then
    echo "找到图标文件: OctopusAI.icns"
    cp "OctopusAI.icns" "${RESOURCES_DIR}/OctopusAI.icns"
elif [ -f "icon.ico" ]; then
    echo "找到图标文件: icon.ico"
    echo "正在转换图标格式..."

    # 尝试转换图标
    if [ -f "convert_icon.sh" ]; then
        bash convert_icon.sh
        if [ -f "OctopusAI.icns" ]; then
            cp "OctopusAI.icns" "${RESOURCES_DIR}/OctopusAI.icns"
            echo "✅ 图标已添加到应用"
        else
            echo "⚠️  图标转换失败，应用将使用默认图标"
            echo "提示: 可以使用在线工具转换: https://cloudconvert.com/ico-to-icns"
        fi
    else
        echo "⚠️  未找到图标转换脚本"
        echo "提示: Mac使用.icns格式图标"
        echo "可以使用在线工具转换: https://cloudconvert.com/ico-to-icns"
    fi
else
    echo "⚠️  未找到图标文件，应用将使用默认图标"
fi

echo ""
echo "✅ 创建成功!"
echo ""
echo "应用程序包: ${APP_FILE}"
echo "使用方法:"
echo "  1. 双击 ${APP_FILE} 即可启动应用"
echo "  2. 或将 ${APP_FILE} 拖到应用程序文件夹"
echo ""
echo "注意: 首次运行可能需要在系统偏好设置中允许运行"
