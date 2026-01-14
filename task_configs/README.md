# AI应用和工作流配置文件开发指南

> 本文档面向AI编程智能体，详细描述VideoAgent系统的AI应用组件字段参数规则、多组件协作工作流的输入输出机制和约束条件。

---

## 目录

- [系统架构概述](#系统架构概述)
- [AI应用组件配置规范](#ai应用组件配置规范)
- [输入字段参数规则](#输入字段参数规则)
- [安装配置规范](#安装配置规范)
- [多组件协作工作流](#多组件协作工作流)
- [输入输出机制](#输入输出机制)
- [工作流执行流程](#工作流执行流程)
- [配置验证规则](#配置验证规则)
- [完整配置示例](#完整配置示例)

---

## 系统架构概述

### 核心组件

VideoAgent系统由三个核心JavaFX程序组成：

```
VideoAgent/
└── src/com/xienlive/ai/video/agent/
    ├── AIModelGenerator.java      # AI应用组件生成器
    ├── AIWorkflowGenerator.java   # 工作流生成器
    ├── WorkflowDesigner.java      # 工作流可视化设计器
    ├── TaskConfig.java            # 任务配置数据模型
    ├── DynamicFormGenerator.java  # 动态表单生成器
    ├── BatchCommand.java          # 批量命令定义
    ├── SimpleSSHClient.java       # SSH客户端
    └── MainV2.java                # 主应用程序
```

### 两种应用类型

#### 1. AI应用组件（Component）
- 独立的AI应用模块
- 执行特定的AI任务（如图像生成、语音识别等）
- 有明确的输入和输出接口
- 配置文件标记：`is_cooperation: false` 或省略此字段

#### 2. 工作流应用（Workflow）
- 由多个AI应用组件组合而成
- 通过 `endpoint://` 协议连接各个组件
- 实现复杂的端到端AI处理流程
- 配置文件标记：`is_cooperation: true`

### 数据流转模型

```
┌──────────────┐
│  用户输入    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│         工作流应用                     │
│  ┌────────────────────────────────┐  │
│  │ 组件A ──output──► 组件B         │  │
│  │   │                   │         │  │
│  │  input               output     │  │
│  │   │                   │         │  │
│  │   └───────────────────┼───────► │  │
│  │                       │          │  │
│  │                      组件C       │  │
│  │                       │         │  │
│  │                      output     │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  最终输出    │
└──────────────┘
```

---

## AI应用组件配置规范

### 基础结构

AI应用组件是工作流的基本构建单元，必须遵循以下结构：

```json
{
  "app_id": "组件唯一标识",
  "name": "组件显示名称",
  "description": "详细描述",
  "author": "作者",
  "homepage": "项目主页",
  "email": "联系邮箱",

  "inputs": [
    {
      "label": "参数标签",
      "field_name": "参数字段名",
      "type": "参数类型",
      "default_value": "默认值",
      "editable": true,
      "required": false
    }
  ],

  "variables": {
    "work_dir": "工作目录",
    "model_path": "模型路径"
  },

  "server_output_file": "输出文件路径",
  "server_output_dir": "输出目录路径",

  "installs": [
    {
      "env_name": "环境名称",
      "python_version": "Python版本",
      "cuda_version": "CUDA版本",
      "commands": ["安装命令序列"]
    }
  ],

  "batch": [
    {
      "name": "执行步骤名称",
      "env_name": "使用的环境名",
      "commands": ["命令序列"]
    }
  ],

  "endpoint": "获取输出结果的命令"
}
```

### app_id 命名规范

**格式要求：**
- 只能包含小写字母、数字、连字符（-）和下划线（_）
- 必须在所有配置文件中唯一
- 建议格式：`{功能}-{项目名}[-{版本}]`

**命名示例：**
```
✅ 正确示例：
cosyvoice-tts-app
facefusion
video-high-quality
wan2.2-t2v-a14b
llm-api-chat

❌ 错误示例：
My App              # 包含空格和大写
应用程序            # 使用中文
app-123            # 无意义的数字
stable-diffusion-2-1  # 过长且难以理解
```

### name 字段规范

**格式要求：**
- 格式：`[中文功能描述]-[项目英文名]`
- 前半部分必须是中文功能描述
- 清晰表达组件的核心功能

**命名示例：**
```
✅ 正确示例：
图像生成-Stable-Diffusion
语音合成-CosyVoice
视频增强-Wan2.2
人脸替换-FaceFusion

❌ 错误示例：
Stable Diffusion-stable-diffusion  # 前半部分包含英文
Image Generation-tool               # 前半部分使用英文
图片生成                            # 缺少项目名
```

### description 字段规范

描述必须包含以下要素：

1. **技术基础** - 使用的模型或技术
2. **核心功能** - 能够完成什么任务
3. **输入要求** - 需要用户提供什么
4. **输出说明** - 生成什么样的结果
5. **使用场景** - 适用于什么场景
6. **特殊约束** - 算力要求、时长限制等

**标准描述模板：**

```json
{
  "description": "基于{技术名称}的{功能}应用，{核心功能描述}。用户可以{用户操作}，系统将{输出结果}。支持{支持的模式/特性}。{约束条件和要求}。算力要求：{硬件要求}。"
}
```

**实际示例：**

```json
{
  "description": "基于CosyVoice 3.0的语音合成应用，支持零样本语音克隆和指令控制语音合成。用户可以上传10秒以内的参考音频提取音色，通过文本生成语音。支持两种模式：1) 语音克隆-使用参考音频克隆音色并合成语音；2) 指令控制-用自然语言指令控制语音特征（如方言、风格、语速）。算力要求：单卡GPU（显存6GB以上）。"
}
```

---

## 输入字段参数规则

### 极其重要的规则

**⚠️ 核心原则：inputs字段仅包含推理（Inference）相关的参数！**

#### ✅ 应该包含的参数

**1. 用户输入类参数**
- 提示词/文本输入（prompt, text, input, query等）
- 创作指令（instruction, command等）

**2. 文件输入类参数**
- 图像文件路径（image, input_image, reference_image等）
- 音频文件路径（audio, voice, sample_audio等）
- 视频文件路径（video, input_video等）
- 文档文件路径（document, file等）

**3. 推理参数类**
- 模型控制参数（temperature, top_k, top_p等）
- 生成参数（steps, num_inference_steps, cfg_scale等）
- 输出控制参数（width, height, resolution, duration等）
- 随机种子（seed）

**4. 输出配置类**
- 输出文件名（output_file, filename等）
- 输出格式（format, file_format等）

#### ❌ 不应该包含的参数

**1. 训练相关参数**
- learning_rate（学习率）
- batch_size（批次大小）
- epochs（训练轮数）
- optimizer（优化器）
- loss_function（损失函数）

**2. 数据集相关参数**
- dataset_path（数据集路径）
- train_data_dir（训练数据目录）
- annotation_file（标注文件）
- split_ratio（数据集划分比例）

**3. 系统配置参数**
- config_file（配置文件路径）
- log_dir（日志目录）
- checkpoint_dir（检查点目录）
- tensorboard_port（TensorBoard端口）

**4. 开发调试参数**
- debug_mode（调试模式）
- verbose（详细输出）
- save_intermediate（保存中间结果）
- profile（性能分析）

### 输入类型详细规范

#### 1. text - 单行文本输入

```json
{
  "label": "API密钥",
  "field_name": "api_key",
  "type": "text",
  "default_value": "",
  "editable": true,
  "required": false
}
```

**使用场景：** 短文本输入，如API密钥、文件名、简短参数等

#### 2. textarea - 多行文本输入

```json
{
  "label": "提示词",
  "field_name": "prompt",
  "type": "textarea",
  "default_value": "描述你想要生成的内容，如：一只在森林里奔跑的狐狸，阳光穿过树叶，电影级画质",
  "editable": true,
  "required": true
}
```

**使用场景：** 长文本输入，如提示词、文章内容、代码片段等

#### 3. number - 数字输入

```json
{
  "label": "生成数量",
  "field_name": "num_images",
  "type": "number",
  "default_value": 4,
  "editable": true,
  "required": true,
  "min_value": 1,
  "max_value": 10,
  "step_value": 1
}
```

**参数说明：**
- `min_value`: 最小值（可选）
- `max_value`: 最大值（可选）
- `step_value`: 步长（可选）

**使用场景：** 数量、次数、比例、等级等数值参数

#### 4. select - 下拉选择

```json
{
  "label": "视频分辨率",
  "field_name": "resolution",
  "type": "select",
  "default_value": "720*1280",
  "editable": true,
  "required": true,
  "options": [
    {
      "label": "竖屏-480P (480×832)",
      "value": "480*832"
    },
    {
      "label": "竖屏-720P (720×1280)",
      "value": "720*1280"
    },
    {
      "label": "竖屏-1080P (1080×1920)",
      "value": "1080*1920"
    }
  ]
}
```

**使用场景：** 预定义的选项，如分辨率、格式、模式等

#### 5. file - 单文件上传

```json
{
  "label": "参考图像",
  "field_name": "reference_image",
  "type": "file",
  "file_extensions": ["jpg", "jpeg", "png", "webp"],
  "default_value": "",
  "editable": true,
  "required": true
}
```

**file_extensions 说明：**
- 必须指定支持的文件扩展名
- 常见格式：图片（jpg, png, webp）、音频（mp3, wav, m4a）、视频（mp4, avi, mov）

**文件处理机制：**
1. 用户选择文件后，系统将文件上传到服务器 `/tmp/uploads/` 目录
2. 文件重命名为 `{UUID}.{扩展名}`
3. 变量替换时，`{field_name}` 会被替换为服务器上的文件路径

#### 6. files - 多文件上传

```json
{
  "label": "参考图像集",
  "field_name": "reference_images",
  "type": "files",
  "file_extensions": ["jpg", "png"],
  "default_value": "",
  "editable": true,
  "required": false
}
```

**多文件处理：**
- `field_name` 会被替换为JSON格式的文件路径列表
- 格式：`'["/tmp/uploads/uuid1.jpg", "/tmp/uploads/uuid2.png"]'`

#### 7. switch - 开关按钮

```json
{
  "label": "启用高清处理",
  "field_name": "enable_hd",
  "type": "switch",
  "default_value": false,
  "editable": true,
  "required": false
}
```

**使用场景：** 二元选项，如启用/禁用、是/否等

#### 8. radio - 单选按钮组

```json
{
  "label": "处理模式",
  "field_name": "mode",
  "type": "radio",
  "default_value": "fast",
  "editable": true,
  "required": true,
  "options": [
    {"label": "快速模式", "value": "fast"},
    {"label": "质量模式", "value": "quality"},
    {"label": "平衡模式", "value": "balanced"}
  ]
}
```

**使用场景：** 互斥的选择项，通常2-4个选项

#### 9. multiselect - 多选列表

```json
{
  "label": "附加功能",
  "field_name": "features",
  "type": "multiselect",
  "default_value": [],
  "editable": true,
  "required": false,
  "options": [
    {"label": "去噪", "value": "denoise"},
    {"label": "增强", "value": "enhance"},
    {"label": "色彩校正", "value": "color_correction"}
  ]
}
```

**使用场景：** 可多选的选项列表

#### 10. prompt - 提示词输入

```json
{
  "label": "创作指令",
  "field_name": "prompt",
  "type": "prompt",
  "default_value": "描述你想要AI创作的内容",
  "editable": true,
  "required": true
}
```

**使用场景：** AI提示词输入，通常用于生成式AI任务

### 字段命名规范

**通用命名原则：**

1. **使用小写字母和下划线**
   ```
   ✅ input_text
   ✅ reference_image
   ✅ num_inference_steps

   ❌ inputText
   ❌ ReferenceImage
   ❌ num-inference-steps
   ```

2. **使用清晰、描述性的名称**
   ```
   ✅ prompt / input_text / query
   ✅ reference_image / source_image
   ✅ num_images / generation_count

   ❌ param1 / field_a / temp
   ```

3. **遵循领域惯例**
   ```
   提示词: prompt, text, query, instruction
   图像: image, input_image, reference_image
   音频: audio, voice, sample_audio
   视频: video, input_video, source_video
   数量: num_{noun}, {noun}_count
   路径: {noun}_path, {noun}_dir
   ```

---

## 安装配置规范

### ⚠️ 极其重要的约束

**禁止在installs命令中使用conda环境创建和激活命令！**

```bash
# ❌ 绝对禁止的命令：
conda create -n my-env python=3.10 -y
conda activate my-env
source activate my-env
conda init bash

# ✅ 允许的命令：
pip install torch torchvision
pip install -r requirements.txt
mkdir -p {work_dir}
git clone https://github.com/xxx/xxx.git
cd {work_dir}
```

### 标准安装模板

```json
{
  "installs": [
    {
      "env_name": "my-app-env",
      "python_version": "3.10",
      "cuda_version": "12.1",
      "commands": [
        "mkdir -p {model_path}",
        "mkdir -p {work_dir}",
        "mkdir -p {server_output_dir}",

        "git clone https://github.com/project/repo.git {work_dir}",
        "cd {work_dir}",

        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install -r requirements.txt",

        "pip install transformers",
        "pip install diffusers",
        "pip install accelerate"
      ]
    }
  ]
}
```

### 目录创建规则

#### 1. model_path（模型存储目录）

```bash
# 在installs阶段创建
mkdir -p {model_path}

# 下载模型到该目录
huggingface-cli download model-name --local-dir {model_path}
```

#### 2. work_dir（工作目录）

```bash
# 在installs阶段删除后创建
rm -rf {work_dir}
mkdir -p {work_dir}

# 克隆代码到工作目录
git clone https://github.com/xxx/xxx.git {work_dir}
```

#### 3. server_output（输出路径）

**情况A：多文件输出（作为目录处理）**
```bash
# 在batch阶段处理
rm -rf {server_output}
mkdir -p {server_output}
```

**情况B：单文件输出（作为文件路径）**
```bash
# 在batch阶段处理
mkdir -p $(dirname {server_output})
```

### 安装命令顺序建议

```json
{
  "commands": [
    // 1. 创建必要的目录
    "mkdir -p {model_path}",
    "mkdir -p {work_dir}",
    "mkdir -p {server_output_dir}",

    // 2. 克隆代码仓库
    "git clone https://github.com/xxx/repo.git {work_dir}",
    "cd {work_dir}",

    // 3. 安装系统依赖（如果有）
    "apt-get update && apt-get install -y ffmpeg libsndfile1",

    // 4. 安装Python依赖
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "pip install -r requirements.txt",

    // 5. 下载模型文件
    "huggingface-cli download model-name --local-dir {model_path}"
  ]
}
```

---

## 多组件协作工作流

### 工作流核心铁律

**⚠️ 必须牢记的五条铁律：**

1. **工作流 = 多个组件协同工作**
   - 工作流不是独立的节点列表
   - 必须通过 `endpoint://` 协议连接各个组件
   - 后面的组件必须使用前面组件的输出

2. **连接的唯一方式：endpoint://协议**
   - 这是连接组件的唯一正确方式
   - 绝不能重复要求用户输入已经生成的数据

3. **必须包含endpoint://引用**
   - 生成的配置中必须包含 `endpoint://` 引用
   - 如果配置中没有 `endpoint://`，则是错误的

4. **工作流可以有任意数量的节点**
   - 不局限于3个节点
   - 可以是2个、3个、5个、10个甚至更多
   - 根据实际需求添加必要的所有组件

5. **每个后续组件都使用前一个组件的输出**
   - 形成链式处理流程：A → B → C → D → E → ...
   - 数据在组件间自动流转

### 工作流配置结构

```json
{
  "app_id": "workflow-id",
  "name": "工作流名称",
  "description": "工作流描述",
  "is_cooperation": true,
  "server_output": "/tmp/workflow-output/final.mp4",

  "inputs": [
    {
      "label": "用户输入参数",
      "field_name": "workflow_param",
      "type": "text",
      "required": true
    }
  ],

  "batch": [
    {
      "name": "步骤1：第一个组件",
      "env_name": "workflow-env",
      "cooperation": {
        "app_id": "component-a",
        "inputs": [
          {
            "field_name": "input_field",
            "value": "{workflow_param}"
          }
        ]
      }
    },
    {
      "name": "步骤2：第二个组件",
      "env_name": "workflow-env",
      "cooperation": {
        "app_id": "component-b",
        "inputs": [
          {
            "field_name": "input_from_a",
            "value": "endpoint://component-a"
          }
        ]
      }
    },
    {
      "name": "步骤3：第三个组件",
      "env_name": "workflow-env",
      "cooperation": {
        "app_id": "component-c",
        "inputs": [
          {
            "field_name": "input_from_b",
            "value": "endpoint://component-b"
          }
        ]
      }
    }
  ],

  "endpoint": "echo \"{server_output}\""
}
```

### 正确示例 vs 错误示例

#### ✅ 正确的做法：使用endpoint://连接

```json
{
  "batch": [
    {
      "name": "语音识别",
      "cooperation": {
        "app_id": "fun-asr-app",
        "inputs": [
          {
            "field_name": "sample_audio",
            "value": "{user_audio_file}"
          }
        ]
      }
    },
    {
      "name": "语音合成",
      "cooperation": {
        "app_id": "cosyvoice-tts-app",
        "inputs": [
          {
            "field_name": "input_text",
            "value": "endpoint://fun-asr-app"
          },
          {
            "field_name": "sample_audio",
            "value": "{user_audio_file}"
          }
        ]
      }
    },
    {
      "name": "唇形同步",
      "cooperation": {
        "app_id": "latentsync",
        "inputs": [
          {
            "field_name": "video",
            "value": "{user_video}"
          },
          {
            "field_name": "audio",
            "value": "endpoint://cosyvoice-tts-app"
          }
        ]
      }
    }
  ]
}
```

#### ❌ 错误的做法：重复要求用户输入

```json
{
  "batch": [
    {
      "name": "语音识别",
      "cooperation": {
        "app_id": "fun-asr-app",
        "inputs": [
          {
            "field_name": "sample_audio",
            "value": "{user_audio_file}"
          }
        ]
      }
    },
    {
      "name": "语音合成",
      "cooperation": {
        "app_id": "cosyvoice-tts-app",
        "inputs": [
          {
            "field_name": "input_text",
            "value": "{user_text}"
          }
        ]
      }
    }
  ]
}
```

**问题分析：**
- 第二个组件要求用户提供 `user_text`
- 但这个文本本应由第一个组件（语音识别）生成
- 应该使用 `endpoint://fun-asr-app` 引用识别结果

---

## 输入输出机制

### 三种输入值类型

#### 1. USER_INPUT（用户输入）

**定义：** 由工作流的最终用户提供的数据

**引用格式：**
```json
{
  "field_name": "input_field",
  "value": "{workflow_param}"
}
```

**适用场景：**
- 原始输入数据（文件、文本、图像等）
- 用户指定的参数（分辨率、数量等）

**示例：**
```json
{
  "cooperation": {
    "app_id": "facefusion",
    "inputs": [
      {
        "field_name": "face_image",
        "value": "{user_face_image}"
      },
      {
        "field_name": "input_video",
        "value": "{user_video}"
      }
    ]
  }
}
```

#### 2. NODE_OUTPUT（节点引用）⚠️ 核心机制

**定义：** 引用其他组件的输出结果

**引用格式：**
```json
{
  "field_name": "input_field",
  "value": "endpoint://source_app_id"
}
```

**适用场景：**
- 使用前一个组件的输出作为当前组件的输入
- 形成组件间的数据流转链

**工作原理：**
1. 系统执行 `source_app_id` 组件
2. 执行该组件的 `endpoint` 命令获取输出
3. 将输出作为当前组件的输入值

**关键约束：**
- 被引用的组件必须在当前组件之前执行
- 被引用的组件必须正确定义 `endpoint` 字段
- 不能形成循环引用（A引用B，B引用A）

**示例：**
```json
{
  "batch": [
    {
      "name": "组件A：生成文本",
      "cooperation": {
        "app_id": "text-generator",
        "inputs": [
          {
            "field_name": "prompt",
            "value": "{user_prompt}"
          }
        ]
      }
    },
    {
      "name": "组件B：文本转语音",
      "cooperation": {
        "app_id": "tts-engine",
        "inputs": [
          {
            "field_name": "text",
            "value": "endpoint://text-generator"
          }
        ]
      }
    }
  ]
}
```

**处理流程：**
```
1. 执行 text-generator
   - 输入：用户提供的 prompt
   - 输出：生成的文本

2. 执行 tts-engine
   - 输入：endpoint://text-generator（会被替换为组件A的输出）
   - 输出：合成的音频
```

#### 3. DEFAULT_VALUE（默认值）

**定义：** 固定的配置值或参数

**引用格式：**
```json
{
  "field_name": "mode",
  "value": "zero_shot"
}
```

**适用场景：**
- 模式选择（如处理模式、质量级别）
- 固定参数（如采样率、编码格式）
- 系统配置（如设备选择、并行度）

**示例：**
```json
{
  "cooperation": {
    "app_id": "cosyvoice-tts-app",
    "inputs": [
      {
        "field_name": "mode",
        "value": "zero_shot"
      },
      {
        "field_name": "instruction",
        "value": ""
      }
    ]
  }
}
```

### endpoint协议详解

#### endpoint协议的作用

`endpoint://` 是连接组件的核心机制，实现了：

1. **自动数据流转**：前一个组件的输出自动成为下一个组件的输入
2. **延迟执行**：只有当需要使用输出时才执行endpoint命令
3. **类型转换**：将命令行输出转换为字符串值

#### endpoint命令定义

每个AI应用组件必须定义 `endpoint` 字段：

```json
{
  "app_id": "my-app",
  "server_output_file": "/tmp/myapp/results/output.txt",
  "endpoint": "echo \"{server_output_file}\""
}
```

**endpoint命令类型：**

**1. 返回文件路径（最常见）**
```bash
# 输出单个文件路径
echo "{server_output_file}"

# 输出多个文件路径（第一个匹配）
find {server_output_dir} -name "*.mp4" | head -n 1

# 输出最新文件
ls -t {server_output_dir}/*.png | head -n 1
```

**2. 返回文件内容**
```bash
# 输出文本内容
cat "{server_output_file}"

# 输出特定行
head -n 10 "{server_output_file}"
```

**3. 复杂路径查找**
```bash
# 查找并输出
python -c "import glob; print(glob.glob('{server_output_dir}/*.mp4')[0])"
```

#### endpoint在协作中的处理

**处理逻辑：**

```java
// 伪代码：endpoint处理流程
for (BatchCommand step : workflowSteps) {
    String appId = step.getCooperationAppId();

    // 1. 执行组件
    executeComponent(appId, step.getInputs());

    // 2. 获取组件的endpoint命令
    String endpointCmd = appConfigs.get(appId).getEndpoint();

    // 3. 执行endpoint命令获取输出
    String output = executeCommand(endpointCmd);

    // 4. 存储输出供后续组件使用
    componentOutputs.put(appId, output);
}

// 后续组件引用
String value = componentOutputs.get("source_app_id");
// 如果配置是 "endpoint://source_app_id"
// 则会被替换为实际输出值
```

**实际示例：**

```json
// 配置1：文本生成组件
{
  "app_id": "text-gen",
  "server_output_file": "/tmp/text-gen/result.txt",
  "endpoint": "echo \"{server_output_file}\"",
  "batch": [
    {
      "commands": [
        "echo '生成的文本' > /tmp/text-gen/result.txt"
      ]
    }
  ]
}

// 配置2：使用文本生成组件的工作流
{
  "app_id": "workflow",
  "batch": [
    {
      "cooperation": {
        "app_id": "text-gen",
        "inputs": [...]
      }
    },
    {
      "cooperation": {
        "app_id": "tts",
        "inputs": [
          {
            "field_name": "text",
            "value": "endpoint://text-gen"
            // 执行时会被替换为：/tmp/text-gen/result.txt
          }
        ]
      }
    }
  ]
}
```

### required_inputs 约束

#### 定义

`required_inputs` 是可选字段，用于指定执行某个batch步骤所必需的输入字段。

#### 使用场景

**1. 条件执行**
```json
{
  "name": "音频融合",
  "env_name": "my-env",
  "required_inputs": ["audio"],
  "commands": [
    "ffmpeg -i {video} -i {audio} -c:v copy -c:a aac output.mp4"
  ]
}
```

**工作原理：**
- 只有当用户提供了 `audio` 字段且不为空时，才执行此步骤
- 如果 `audio` 为空或未提供，跳过此步骤

**2. 多种模式处理**

```json
{
  "batch": [
    {
      "name": "模式A：处理",
      "env_name": "my-env",
      "required_inputs": ["mode_a_param"],
      "commands": [
        "echo '执行模式A'"
      ]
    },
    {
      "name": "模式B：处理",
      "env_name": "my-env",
      "required_inputs": ["mode_b_param"],
      "commands": [
        "echo '执行模式B'"
      ]
    }
  ]
}
```

#### 验证逻辑

```java
// 伪代码：required_inputs验证
for (BatchCommand step : batchSteps) {
    List<String> requiredInputs = step.getRequiredInputs();

    if (requiredInputs != null && !requiredInputs.isEmpty()) {
        // 检查每个必需字段
        for (String fieldName : requiredInputs) {
            Object value = getInputValue(fieldName);

            if (value == null || isEmpty(value)) {
                // 跳过此步骤
                skipStep(step);
                continue nextStep;
            }
        }
    }

    // 所有必需字段都存在，执行步骤
    executeStep(step);
}
```

---

## 工作流执行流程

### 完整执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                  1. 配置加载与验证                          │
│  - 加载工作流配置JSON                                       │
│  - 验证is_cooperation标记                                  │
│  - 检查server_output字段                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  2. 加载组件配置                            │
│  - 根据app_id加载每个组件的配置                            │
│  - 验证组件的endpoint定义                                  │
│  - 构建组件依赖图                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. 收集工作流输入                          │
│  - 分析所有组件的inputs配置                                │
│  - 识别USER_INPUT类型的输入                                │
│  - 生成工作流的inputs字段                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  4. 用户填写表单                            │
│  - 显示工作流输入表单                                       │
│  - 用户填写参数值                                          │
│  - 上传文件                                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  5. 建立SSH连接                             │
│  - 连接到远程服务器                                        │
│  - 验证环境状态                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  6. 执行组件（循环）                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 对于每个组件（按batch顺序）:                       │    │
│  │                                                     │    │
│  │ a) 准备输入值                                       │    │
│  │    - 解析USER_INPUT：{workflow_param}             │    │
│  │    - 解析NODE_OUTPUT：endpoint://app_id           │    │
│  │    - 解析DEFAULT_VALUE：固定值                     │    │
│  │                                                     │    │
│  │ b) 执行组件的batch命令                             │    │
│  │    - 在指定环境中执行                               │    │
│  │    - 实时显示输出                                   │    │
│  │                                                     │    │
│  │ c) 获取组件输出                                     │    │
│  │    - 执行endpoint命令                               │    │
│  │    - 解析输出结果                                   │    │
│  │    - 存储到组件输出映射                             │    │
│  │                                                     │    │
│  │ d) 检查错误                                         │    │
│  │    - 如果失败，停止工作流                           │    │
│  │    - 显示错误信息                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  7. 获取最终输出                            │
│  - 执行工作流的endpoint命令                                │
│  - 下载结果文件（可选）                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  8. 完成并清理                              │
│  - 关闭SSH连接                                             │
│  - 显示执行摘要                                            │
└─────────────────────────────────────────────────────────────┘
```

### 组件执行详细流程

#### 输入值解析

```java
// 伪代码：输入值解析
private Object parseInputValue(JSONObject coopInput,
                                Map<String, Object> workflowInputs,
                                Map<String, String> componentOutputs) {
    String value = coopInput.getString("value");

    // 1. 检查是否是endpoint引用
    if (value.startsWith("endpoint://")) {
        String sourceAppId = value.substring(11); // 移除 "endpoint://"
        String output = componentOutputs.get(sourceAppId);
        if (output == null) {
            throw new Exception("组件输出不存在: " + sourceAppId);
        }
        return output;
    }

    // 2. 检查是否是工作流输入引用
    if (value.matches("\\{[a-zA-Z0-9_\\.]+\\}")) {
        String fieldName = value.substring(1, value.length() - 1);
        Object inputValue = workflowInputs.get(fieldName);
        if (inputValue == null) {
            throw new Exception("工作流输入不存在: " + fieldName);
        }
        return inputValue;
    }

    // 3. 默认值
    return value;
}
```

#### 组件输出获取

```java
// 伪代码：获取组件输出
private String getComponentOutput(String appId,
                                   Map<String, JSONObject> appConfigs) {
    // 1. 获取组件配置
    JSONObject config = appConfigs.get(appId);
    if (config == null) {
        throw new Exception("组件配置不存在: " + appId);
    }

    // 2. 获取endpoint命令
    String endpointCmd = config.optString("endpoint", "");
    if (endpointCmd.isEmpty()) {
        throw new Exception("组件未定义endpoint: " + appId);
    }

    // 3. 执行endpoint命令
    String output = executeRemoteCommand(endpointCmd);

    // 4. 解析输出（通常是路径）
    output = output.trim();

    return output;
}
```

#### 依赖关系验证

```java
// 伪代码：验证依赖关系
private void validateDependencies(JSONArray batch,
                                   Map<String, JSONObject> appConfigs) {
    Set<String> executedApps = new HashSet<>();

    for (int i = 0; i < batch.length(); i++) {
        JSONObject step = batch.getJSONObject(i);
        JSONObject cooperation = step.optJSONObject("cooperation");

        if (cooperation != null) {
            JSONArray inputs = cooperation.optJSONArray("inputs");
            if (inputs != null) {
                for (int j = 0; j < inputs.length(); j++) {
                    JSONObject input = inputs.getJSONObject(j);
                    String value = input.optString("value", "");

                    // 检查endpoint引用
                    if (value.startsWith("endpoint://")) {
                        String sourceAppId = value.substring(11);

                        // 验证被引用的组件
                        if (!appConfigs.containsKey(sourceAppId)) {
                            throw new Exception("引用的组件不存在: " + sourceAppId);
                        }

                        // 验证执行顺序
                        if (!executedApps.contains(sourceAppId)) {
                            throw new Exception("组件顺序错误: " + sourceAppId +
                                              " 必须在当前组件之前执行");
                        }
                    }
                }
            }

            // 记录已执行的组件
            String appId = cooperation.getString("app_id");
            executedApps.add(appId);
        }
    }
}
```

### 错误处理机制

#### 错误类型

**1. 配置错误**
- 组件的app_id不存在
- endpoint未定义
- 必需字段缺失

**2. 依赖错误**
- 循环依赖
- 依赖顺序错误
- 引用不存在的组件

**3. 执行错误**
- 命令执行失败
- 组件输出为空
- 文件不存在

**4. 数据类型错误**
- 期望文件但得到文本
- 期望数字但得到字符串

#### 错误处理流程

```java
try {
    // 执行工作流
    executeWorkflow(workflowConfig);

} catch (ComponentNotFoundException e) {
    // 组件不存在
    showError("组件配置缺失: " + e.getMessage());
    logError("请检查所有引用的组件是否存在");

} catch (DependencyCycleException e) {
    // 循环依赖
    showError("工作流存在循环依赖: " + e.getMessage());
    logError("请检查组件间的引用关系");

} catch (ComponentExecutionException e) {
    // 组件执行失败
    showError("组件执行失败: " + e.getComponentId());
    logError("错误详情: " + e.getMessage());
    logError("请检查组件的日志输出");

} catch (EndpointNotFoundException e) {
    // endpoint未定义
    showError("组件未定义endpoint: " + e.getAppId());
    logError("请为该组件添加endpoint字段");

} catch (Exception e) {
    // 其他错误
    showError("工作流执行失败: " + e.getMessage());
    logError("请查看详细日志");
}
```

---

## 配置验证规则

### AIModelGenerator验证规则

#### 1. app_id唯一性验证

```java
// 验证app_id是否已存在
String appId = config.optString("app_id", "");
if (existingAppConfigs.containsKey(appId)) {
    throw new RuntimeException("应用ID已存在: " + appId);
}
```

#### 2. URL格式验证

```java
// GitHub URL格式
Pattern GITHUB_PATTERN = Pattern.compile(
    "https://github\\.com/([^/]+)/([^/]+)/?"
);

// ModelScope URL格式
Pattern MODEL_SCOPE_PATTERN = Pattern.compile(
    "https://(?:www\\.)?modelscope\\.cn/(?:models/)?([^/]+)/([^/?#]+)/?"
);

// 验证URL
if (!GITHUB_PATTERN.matcher(url).matches() &&
    !MODEL_SCOPE_PATTERN.matcher(url).matches()) {
    throw new RuntimeException("不支持的网址格式");
}
```

#### 3. 应用名称格式验证

```java
// 验证name格式：[中文功能描述]-[项目英文名]
Pattern NAME_PATTERN = Pattern.compile(
    "^[\\u4e00-\\u9fa5]+[\\-\\w]+$"
);

String name = config.optString("name", "");
if (!NAME_PATTERN.matcher(name).matches()) {
    throw new RuntimeException("应用名称格式错误");
}
```

### AIWorkflowGenerator验证规则

#### 1. endpoint://引用验证

```java
// 检查是否包含endpoint://引用
String configStr = config.toString();
if (!configStr.contains("endpoint://")) {
    logError("⚠️ 警告：配置中未检测到endpoint://引用");
    logError("工作流应该包含组件间的连接");
}

// 统计引用次数
int count = configStr.split("endpoint://", -1).length - 1;
logInfo("✅ endpoint://引用数量: " + count);
```

#### 2. 组件顺序验证

```java
// 验证组件执行顺序
Set<String> executedApps = new HashSet<>();
for (int i = 0; i < batch.length(); i++) {
    JSONObject step = batch.getJSONObject(i);
    String appId = step.getJSONObject("cooperation").getString("app_id");

    // 检查此步骤引用的所有组件
    JSONArray inputs = step.getJSONObject("cooperation")
                          .getJSONArray("inputs");

    for (int j = 0; j < inputs.length(); j++) {
        String value = inputs.getJSONObject(j).getString("value");
        if (value.startsWith("endpoint://")) {
            String sourceAppId = value.substring(11);
            if (!executedApps.contains(sourceAppId)) {
                throw new RuntimeException(
                    "组件顺序错误: " + sourceAppId +
                    " 必须在 " + appId + " 之前执行"
                );
            }
        }
    }

    executedApps.add(appId);
}
```

#### 3. 循环依赖检测

```java
// 使用深度优先搜索检测循环依赖
private boolean hasCycle(String currentApp, String targetApp,
                        Set<String> visited) {
    if (currentApp.equals(targetApp)) {
        return true; // 找到循环
    }

    if (visited.contains(currentApp)) {
        return false; // 已访问过，跳过
    }

    visited.add(currentApp);

    // 获取当前组件依赖的所有组件
    List<String> dependencies = getDependencies(currentApp);

    for (String dep : dependencies) {
        if (hasCycle(dep, targetApp, visited)) {
            return true;
        }
    }

    return false;
}
```

### WorkflowDesigner验证规则

#### 1. 输入值类型验证

```java
// 验证输入值配置
private void validateInputValueConfig(AppNode node, String fieldName) {
    InputValueConfig config = node.getInputValueConfig(fieldName);

    if (config == null) {
        throw new ValidationException(
            "输入字段未配置: " + node.getAppId() + "." + fieldName
        );
    }

    // 验证配置类型
    InputValueType type = config.getType();
    switch (type) {
        case NODE_OUTPUT:
            AppNode sourceNode = config.getSourceNode();
            if (sourceNode == null) {
                throw new ValidationException(
                    "NODE_OUTPUT类型未指定源节点: " + fieldName
                );
            }
            break;

        case USER_INPUT:
            // 验证是否在工作流输入中
            String inputKey = node.getAppId() + "." + fieldName;
            if (!isWorkflowInput(inputKey)) {
                throw new ValidationException(
                    "USER_INPUT未定义为工作流输入: " + inputKey
                );
            }
            break;

        case DEFAULT_VALUE:
            if (config.getUserValue() == null) {
                throw new ValidationException(
                    "DEFAULT_VALUE类型必须指定值: " + fieldName
                );
            }
            break;
    }
}
```

#### 2. 连接完整性验证

```java
// 验证所有连接都有效
private void validateConnections() {
    for (Connection conn : connections) {
        AppNode source = conn.getSourceNode();
        AppNode target = conn.getTargetNode();

        // 1. 验证源节点存在
        if (!appNodes.containsKey(source.getNodeId())) {
            throw new ValidationException(
                "源节点不存在: " + source.getAppId()
            );
        }

        // 2. 验证目标节点存在
        if (!appNodes.containsKey(target.getNodeId())) {
            throw new ValidationException(
                "目标节点不存在: " + target.getAppId()
            );
        }

        // 3. 验证源节点定义了endpoint
        JSONObject sourceConfig = appConfigs.get(source.getAppId());
        if (sourceConfig == null ||
            !sourceConfig.has("endpoint") ||
            sourceConfig.getString("endpoint").isEmpty()) {
            throw new ValidationException(
                "源节点未定义endpoint: " + source.getAppId()
            );
        }

        // 4. 验证顺序：源在目标之前
        int sourceOrder = nodeOrder.get(source);
        int targetOrder = nodeOrder.get(target);
        if (sourceOrder >= targetOrder) {
            throw new ValidationException(
                "节点顺序错误: " + source.getAppId() +
                " 应该在 " + target.getAppId() + " 之前"
            );
        }
    }
}
```

---

## 完整配置示例

### 示例1：简单AI应用组件

**图像生成组件**

```json
{
  "app_id": "image-gen-pro",
  "name": "AI图像生成-Stable-Diffusion",
  "description": "基于Stable Diffusion的AI图像生成工具，支持文本生成图像。用户输入提示词描述想要的图像，系统生成对应的高质量图像。支持自定义图像尺寸、生成数量、推理步数等参数。输出为PNG格式图像文件。算力要求：GPU显存8GB以上，推荐使用V100或A100。",
  "author": "AI Lab",
  "homepage": "https://github.com/AILab/image-gen-pro",
  "email": "contact@ai-lab.com",

  "inputs": [
    {
      "label": "提示词",
      "field_name": "prompt",
      "type": "textarea",
      "default_value": "一只可爱的猫咪在阳光下睡觉，油画风格，细节丰富",
      "editable": true,
      "required": true
    },
    {
      "label": "负面提示词",
      "field_name": "negative_prompt",
      "type": "textarea",
      "default_value": "低质量, 模糊, 扭曲, 变形",
      "editable": true,
      "required": false
    },
    {
      "label": "图像尺寸",
      "field_name": "size",
      "type": "select",
      "default_value": "512*512",
      "editable": true,
      "required": true,
      "options": [
        {"label": "方形 (512×512)", "value": "512*512"},
        {"label": "横向 (768×512)", "value": "768*512"},
        {"label": "竖向 (512×768)", "value": "512*768"},
        {"label": "高清 (1024×1024)", "value": "1024*1024"}
      ]
    },
    {
      "label": "生成数量",
      "field_name": "num_images",
      "type": "number",
      "default_value": 4,
      "editable": true,
      "required": true,
      "min_value": 1,
      "max_value": 10,
      "step_value": 1
    },
    {
      "label": "推理步数",
      "field_name": "steps",
      "type": "number",
      "default_value": 50,
      "editable": true,
      "required": false,
      "min_value": 10,
      "max_value": 150,
      "step_value": 10
    },
    {
      "label": "随机种子",
      "field_name": "seed",
      "type": "number",
      "default_value": 42,
      "editable": true,
      "required": false
    }
  ],

  "variables": {
    "work_dir": "$HOME/image-gen-pro",
    "model_path": "$HOME/.cache/stable-diffusion"
  },

  "server_output_dir": "/tmp/image-gen-pro-results",
  "model_path": "$HOME/.cache/image-gen-pro-models",

  "installs": [
    {
      "env_name": "image-gen-env",
      "python_version": "3.10",
      "cuda_version": "12.1",
      "commands": [
        "mkdir -p {work_dir}",
        "mkdir -p {server_output_dir}",
        "mkdir -p {model_path}",

        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install diffusers transformers accelerate",
        "pip install safetensors",
        "pip install pillow",
        "pip install huggingface_hub[cli]",

        "huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir {model_path}"
      ]
    }
  ],

  "uploads": [
    {
      "source_path": "image-gen-pro/generate.py",
      "server_path": "/tmp/image-gen-pro/generate.py"
    }
  ],

  "batch": [
    {
      "name": "准备模型",
      "env_name": "image-gen-env",
      "commands": [
        "huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir {model_path}"
      ]
    },
    {
      "name": "生成图像",
      "env_name": "image-gen-env",
      "commands": [
        "rm -rf {server_output_dir}/*",
        "cd {work_dir}",
        "python /tmp/image-gen-pro/generate.py \\",
        "  --model_dir {model_path} \\",
        "  --prompt \"{prompt}\" \\",
        "  --negative_prompt \"{negative_prompt}\" \\",
        "  --size {size} \\",
        "  --num_images {num_images} \\",
        "  --steps {steps} \\",
        "  --seed {seed} \\",
        "  --output_dir {server_output_dir}"
      ]
    }
  ],

  "endpoint": "find {server_output_dir} -name \"*.png\" | head -n 1"
}
```

### 示例2：多组件协作工作流

**数字人视频生成工作流**

```json
{
  "app_id": "digital-human-workflow",
  "name": "数字人视频生成工作流",
  "description": "完整的数字人视频生成流程，整合语音识别、人脸替换、语音合成、唇形同步、背景替换和视频放大等多个AI组件。用户只需提供参考音频、人脸图像、背景素材，系统自动生成高质量的数字人视频。适用于虚拟主播、数字人直播、视频内容创作等场景。",
  "is_cooperation": true,
  "server_output": "/tmp/digital-human-results/final_video.mp4",

  "inputs": [
    {
      "label": "参考音频（10秒以内，用于提取音色）",
      "field_name": "sample_audio",
      "type": "file",
      "file_extensions": ["mp3", "wav", "m4a"],
      "required": true
    },
    {
      "label": "人脸图像",
      "field_name": "face_image",
      "type": "file",
      "file_extensions": ["jpg", "jpeg", "png"],
      "required": true
    },
    {
      "label": "替换视频",
      "field_name": "input_video",
      "type": "file",
      "file_extensions": ["mp4", "avi", "mov"],
      "required": true
    },
    {
      "label": "替换背景（图像或视频）",
      "field_name": "background",
      "type": "file",
      "file_extensions": ["jpg", "jpeg", "png", "mp4"],
      "required": true
    },
    {
      "label": "放大倍数",
      "field_name": "zoom_factor",
      "type": "number",
      "default_value": 2,
      "required": true,
      "min_value": 2,
      "max_value": 4
    }
  ],

  "batch": [
    {
      "name": "步骤1：音频文本提取",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "fun-asr-app",
        "inputs": [
          {
            "field_name": "sample_audio",
            "value": "{sample_audio}"
          }
        ]
      }
    },
    {
      "name": "步骤2：人脸替换",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "facefusion",
        "inputs": [
          {
            "field_name": "face_image",
            "value": "{face_image}"
          },
          {
            "field_name": "input_video",
            "value": "{input_video}"
          }
        ]
      }
    },
    {
      "name": "步骤3：语音合成",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "cosyvoice-tts-app",
        "inputs": [
          {
            "field_name": "input_text",
            "value": "endpoint://fun-asr-app"
          },
          {
            "field_name": "sample_audio",
            "value": "{sample_audio}"
          },
          {
            "field_name": "mode",
            "value": "zero_shot"
          }
        ]
      }
    },
    {
      "name": "步骤4：唇形同步",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "latentsync",
        "inputs": [
          {
            "field_name": "video",
            "value": "endpoint://facefusion"
          },
          {
            "field_name": "audio",
            "value": "endpoint://cosyvoice-tts-app"
          }
        ]
      }
    },
    {
      "name": "步骤5：背景替换",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "robust-videomatting",
        "inputs": [
          {
            "field_name": "input_source",
            "value": "endpoint://latentsync"
          },
          {
            "field_name": "background",
            "value": "{background}"
          }
        ]
      }
    },
    {
      "name": "步骤6：视频放大",
      "env_name": "digital-human-env",
      "cooperation": {
        "app_id": "video-high-quality",
        "inputs": [
          {
            "field_name": "video_file",
            "value": "endpoint://robust-videomatting"
          },
          {
            "field_name": "zoom_factor",
            "value": "{zoom_factor}"
          }
        ]
      }
    }
  ],

  "endpoint": "echo \"{server_output}\""
}
```

**工作流说明：**

```
数据流转流程：
用户音频 ──────► [语音识别] ──► 识别文本
                     │
                     ▼
用户音频 + 人脸图像 ──────► [语音合成] ──► 合成音频
                     │
                     ▼
用户视频 + 人脸图像 ──────► [人脸替换] ──► 替换后视频
                     │
                     ▼
替换视频 + 合成音频 ──────► [唇形同步] ──► 同步后视频
                     │
                     ▼
同步视频 + 背景图 ──────► [背景替换] ──► 替换背景视频
                     │
                     ▼
背景视频 + 放大倍数 ──────► [视频放大] ──► 最终视频
```

### 示例3：条件执行工作流

**图像处理工作流**

```json
{
  "app_id": "image-processing-workflow",
  "name": "智能图像处理工作流",
  "description": "根据用户需求执行不同的图像处理流程。支持图像增强、去噪、色彩校正、超分辨率等功能。用户可以选择启用特定的处理步骤。",
  "is_cooperation": true,
  "server_output": "/tmp/image-processing-results/output.png",

  "inputs": [
    {
      "label": "输入图像",
      "field_name": "input_image",
      "type": "file",
      "file_extensions": ["jpg", "jpeg", "png"],
      "required": true
    },
    {
      "label": "启用图像增强",
      "field_name": "enable_enhance",
      "type": "switch",
      "default_value": false,
      "required": false
    },
    {
      "label": "启用去噪",
      "field_name": "enable_denoise",
      "type": "switch",
      "default_value": false,
      "required": false
    },
    {
      "label": "启用超分辨率",
      "field_name": "enable_superres",
      "type": "switch",
      "default_value": false,
      "required": false
    },
    {
      "label": "放大倍数",
      "field_name": "scale_factor",
      "type": "select",
      "default_value": "2",
      "required": false,
      "options": [
        {"label": "2倍", "value": "2"},
        {"label": "4倍", "value": "4"}
      ]
    }
  ],

  "batch": [
    {
      "name": "步骤1：图像增强（可选）",
      "env_name": "image-processing-env",
      "required_inputs": ["enable_enhance"],
      "cooperation": {
        "app_id": "image-enhance",
        "inputs": [
          {
            "field_name": "input_image",
            "value": "{input_image}"
          }
        ]
      }
    },
    {
      "name": "步骤2：图像去噪（可选）",
      "env_name": "image-processing-env",
      "required_inputs": ["enable_denoise"],
      "cooperation": {
        "app_id": "image-denoise",
        "inputs": [
          {
            "field_name": "input_image",
            "value": "endpoint://image-enhance"
          }
        ]
      }
    },
    {
      "name": "步骤3：超分辨率（可选）",
      "env_name": "image-processing-env",
      "required_inputs": ["enable_superres"],
      "cooperation": {
        "app_id": "image-super-resolution",
        "inputs": [
          {
            "field_name": "input_image",
            "value": "endpoint://image-denoise"
          },
          {
            "field_name": "scale_factor",
            "value": "{scale_factor}"
          }
        ]
      }
    }
  ],

  "endpoint": "endpoint://image-super-resolution"
}
```

**条件执行说明：**

```
可能的执行路径：

1. 仅启用增强：
   输入图像 → [图像增强] → 输出

2. 启用增强 + 去噪：
   输入图像 → [图像增强] → [图像去噪] → 输出

3. 启用增强 + 去噪 + 超分辨率：
   输入图像 → [图像增强] → [图像去噪] → [超分辨率] → 输出

4. 仅启用超分辨率：
   输入图像 → [超分辨率] → 输出
```

---

## 关键约束总结

### AI应用组件约束

1. **app_id唯一性**：所有配置文件中必须唯一
2. **name格式**：必须使用 `[中文功能描述]-[项目英文名]` 格式
3. **inputs约束**：仅包含推理相关参数，不包含训练、数据集等参数
4. **安装命令约束**：禁止使用conda创建和激活命令
5. **endpoint必需**：必须定义endpoint字段用于输出结果

### 工作流约束

1. **is_cooperation标记**：工作流必须设置 `is_cooperation: true`
2. **endpoint://引用**：必须使用 `endpoint://` 协议连接组件
3. **执行顺序**：被引用的组件必须在当前组件之前执行
4. **无循环依赖**：不能形成组件间的循环引用
5. **完整inputs**：必须明确指定工作流需要的所有用户输入

### 输入值类型约束

1. **USER_INPUT**：必须是工作流inputs中定义的字段
2. **NODE_OUTPUT**：必须引用已定义的组件app_id
3. **DEFAULT_VALUE**：必须有明确的值

### 验证规则

1. **配置完整性**：所有必需字段必须存在
2. **格式正确性**：字段值必须符合预期格式
3. **依赖正确性**：组件依赖关系必须合理
4. **连接有效性**：所有连接必须指向有效组件

---

**文档维护者**: AI开发团队
**最后更新**: 2025-01-11
**文档状态**: 正式版 v1.0
