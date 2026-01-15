# OctopusAI - Mac OS 使用说明

## 快速启动

### 方法1：使用.command脚本（推荐，最简单）

1. 下载并解压所有文件到同一目录
2. 双击 `OctopusAI.command` 文件
3. 如果提示"无法打开"，右键点击 → 选择"打开" → 点击"打开"

### 方法2：创建.app应用程序（推荐，最专业）

#### 自动创建（推荐）

1. 打开"终端"应用
2. 进入项目目录：
   ```bash
   cd /path/to/VideoAgent_jar
   ```
3. 运行创建脚本：
   ```bash
   bash create_mac_app.sh
   ```
4. 脚本会自动创建 `OctopusAI.app` 应用程序包
5. 双击 `OctopusAI.app` 即可启动应用

#### 手动创建（了解.app结构）

如果想手动创建.app应用程序包：

```bash
# 创建目录结构
mkdir -p OctopusAI.app/Contents/MacOS
mkdir -p OctopusAI.app/Contents/Resources

# 创建 Info.plist 文件
cat > OctopusAI.app/Contents/Info.plist << 'EOF'
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
</dict>
</plist>
EOF

# 创建启动脚本
cat > OctopusAI.app/Contents/MacOS/launch << 'EOF'
#!/bin/bash
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_DIR="${APP_DIR%/*}"
APP_ROOT="${APP_DIR%/*}"
cd "$APP_ROOT"
exec java -Xmx4096M -Dfile.encoding=UTF-8 -jar VideoAgent.jar
EOF

chmod +x OctopusAI.app/Contents/MacOS/launch
```

### 方法3：从终端启动

```bash
# 进入项目目录
cd /path/to/VideoAgent_jar

# 运行应用
java -jar VideoAgent.jar
```

## 系统要求

- **Mac OS X**: 10.10 Yosemite 或更高版本
- **Java**: Java 11 或更高版本（推荐使用 JDK 11）
- **内存**: 至少 4GB RAM
- **磁盘空间**: 至少 500MB 可用空间

## 安装 Java

### 检查是否已安装 Java

打开终端，运行：
```bash
java -version
```

### 如果未安装 Java

1. 访问 Oracle 官网下载：https://www.oracle.com/java/technologies/downloads/
2. 下载 Java JDK 11（LTS 版本）
3. 安装 JDK
4. 重新运行 `java -version` 验证安装

或使用 Homebrew 安装：
```bash
brew install openjdk@11
```

## 首次运行

### 安全设置

Mac OS 可能会阻止首次运行未签名的应用，如果出现以下提示：

**"无法验证开发者"**
1. 点击"取消"
2. 打开"系统偏好设置" → "安全性与隐私"
3. 找到"已阻止使用"部分
4. 点击"仍要打开"

**"应用已损坏"**
1. 打开终端
2. 运行以下命令：
   ```bash
   xattr -cr /path/to/OctopusAI.app
   ```
3. 重新尝试打开应用

## 添加到应用程序文件夹

### 使用.command脚本
无法直接添加到应用程序文件夹，建议使用.app方式

### 使用.app应用程序包

1. 将 `OctopusAI.app` 拖到 `/Applications` 文件夹
2. 可以从启动台（Launchpad）启动应用

## 创建桌面快捷方式

1. 打开 Finder
2. 找到 `OctopusAI.app`
3. 右键点击 → 选择"制作别名"
4. 将别名拖到桌面

## 配置文件位置

应用配置文件位于：
- **应用配置**: `./task_configs/`
- **历史记录**: `./history/`
- **结果文件**: `./results/`
- **日志文件**: `./error.log`

## 常见问题

### Q1: 双击.command文件时显示"无法打开"

**解决方案**:
1. 右键点击 `OctopusAI.command`
2. 按住键盘上的 `Option` 键
3. 菜单中会显示"打开"选项
4. 点击"打开"
5. 以后就可以直接双击打开了

### Q2: 提示"找不到Java"

**解决方案**:
1. 确认已安装 Java 11 或更高版本
2. 在终端运行：`/usr/libexec/java_home`
3. 如果没有输出，需要重新安装 Java

### Q3: 应用启动闪退

**解决方案**:
1. 打开"终端"应用
2. 运行应用查看错误信息：
   ```bash
   cd /path/to/VideoAgent_jar
   java -jar VideoAgent.jar
   ```
3. 根据错误信息排查问题

### Q4: 如何查看日志

日志文件保存在：`./error.log`

查看日志：
```bash
tail -f error.log
```

## 高级配置

### 调整内存设置

编辑启动脚本中的内存参数：
```bash
# 将 -Xmx4096M 改为需要的值，例如 8192M (8GB)
java -Xmx8192M -jar VideoAgent.jar
```

### 添加JVM参数

可以在启动命令中添加额外的JVM参数：
```bash
java -Xmx4096M -Dfile.encoding=UTF-8 -Duser.timezone=Asia/Shanghai -jar VideoAgent.jar
```

### 后台运行

如果想在后台运行应用：
```bash
nohup java -jar VideoAgent.jar > output.log 2>&1 &
```

## 自动启动

### 登录时自动启动

1. 打开"系统偏好设置" → "用户与群组"
2. 选择当前用户
3. 点击"登录项"标签
4. 点击"+"按钮
5. 添加 `OctopusAI.app`

## 卸载

1. 如果已安装到 `/Applications`：
   ```bash
   rm -rf /Applications/OctopusAI.app
   ```

2. 删除应用数据：
   ```bash
   rm -rf ~/Library/Application\ Support/OctopusAI
   ```

## 技术支持

如有问题，请访问：
- **GitHub**: https://github.com/wukaikevin/OctopusAI
- **提交Issue**: https://github.com/wukaikevin/OctopusAI/issues

## 更新日志

### v1.0.0 (2025-01-14)
- 首次发布 Mac OS 版本
- 支持.command脚本启动
- 支持.app应用程序包
- 完整的使用说明文档
