# 任务：修改AI应用模块配置

## 当前配置
```
{
  "variables": {"work_dir": "$HOME/Qwen-Image-Edit-2511"},
  "server_output_dir": "/tmp/Qwen-Image-Edit-2511_results",
  "author": "通义千问",
  "inputs": [
    {
      "editable": true,
      "default_value": "请将这张图片的背景替换为大海",
      "label": "提示词(描述编辑要求)",
      "type": "textarea",
      "required": true,
      "field_name": "prompt"
    },
    {
      "editable": true,
      "default_value": " ",
      "label": "负面提示词(避免出现的内容)",
      "type": "text",
      "required": false,
      "field_name": "negative_prompt"
    },
    {
      "file_extensions": [
        "jpg",
        "jpeg",
        "png"
      ],
      "editable": true,
      "default_value": "",
      "label": "参考图像(待编辑的原始图片以及参考图片，可多图混用)",
      "type": "files",
      "required": true,
      "field_name": "input_image"
    },
    {
      "editable": true,
      "default_value": "40",
      "label": "推理步骤(迭代次数，值越大质量越高但速度越慢，建议20-50，过低质量差，过高耗时增加)",
      "type": "number",
      "required": true,
      "field_name": "num_inference_steps"
    },
    {
      "editable": true,
      "default_value": "0",
      "label": "随机值(生成种子，相同种子可复现结果，0为随机)",
      "type": "number",
      "required": true,
      "field_name": "seed"
    },
    {
      "editable": true,
      "default_value": "4",
      "label": "提示词匹配度(控制编辑强度，值越大越遵循提示词但可能失真，建议3-7，过低编辑效果弱，过高可能破坏原图)",
      "type": "number",
      "required": true,
      "field_name": "true_cfg_scale"
    }
  ],
  "batch": [{
    "env_name": "Qwen-Image-Edit-2511",
    "name": "图像编辑",
    "commands": [
      "rm -rf {server_output_dir} && mkdir -p {server_output_dir}",
      "cd {work_dir}",
      "cp /tmp/Qwen-Image-Edit-2511_run.py ./run.py",
      "echo '待处理图像：{input_image}'",
      "python run.py \"{prompt}\" '{input_image}' {server_output_file} {num_inference_steps} 1.0 {seed} {true_cfg_scale} \"{negative_prompt}\" 1"
    ]
  }],
  "description": "基于通义千问Qwen-Image-Edit-2511的智能图像编辑模型，采用320亿参数的修正流变换器架构。支持根据自然语言指令进行图像生成、编辑和多图组合。提供推理步数、随机种子、提示词匹配度等多种参数控制，支持负面提示词避免不希望的内容。可精确控制编辑强度，实现高质量图像生成与编辑，适用于创意设计、图像修整、风格转换等场景。",
  "model_path": "$HOME/.cache/Qwen-Image-Edit-2511",
  "installs": [{
    "env_name": "Qwen-Image-Edit-2511",
    "cuda_version": "12.9",
    "python_version": "3.12",
    "commands": [
      "rm -rf {work_dir}",
      "mkdir -p {model_path}",
      "mkdir {work_dir}",
      "conda install pytorch=2.9.* torchvision -c conda-forge -c pytorch -y",
      "pip install transformers",
      "pip install accelerate",
      "pip install git+https://github.com/huggingface/diffusers",
      "pip install packaging"
    ]
  }],
  "uploads": [{
    "server_path": "/tmp/Qwen-Image-Edit-2511_run.py",
    "source_path": "Qwen-Image-Edit-2511/run.py"
  }],
  "endpoint": "echo \"{server_output_file}\"",
  "name": "智能图像编辑 - Qwen-Image-Edit-2511",
  "app_id": "Qwen-Image-Edit-2511",
  "email": "wuk@qq.com",
  "server_output_file": "/tmp/Qwen-Image-Edit-2511_results/result.png",
  "homepage": "https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511"
}
```

## 用户修改要求
【AI调试模式】

当前应用执行时出现以下错误：

```
8755-c6b658490f48.png
开始上传文件: C:\Users\ThinkPad\Downloads\yaya-model.png -> /tmp/uploads/7a1afe31-6740-4a3d-8a85-d32a02f8ffc6.png (大小: 21.7 KB)
上传进度: 21.7 KB / 21.7 KB (100.0%)
等待通道关闭... 已等待 0 秒
文件上传成功: C:\Users\ThinkPad\Downloads\yaya-model.png -> /tmp/uploads/7a1afe31-6740-4a3d-8a85-d32a02f8ffc6.png
智能图像编辑 - Qwen-Image-Edit-2511 > 图像编辑
待处理图像：["/tmp/uploads/8bf0c24d-ef4d-456f-8755-c6b658490f48.png","/tmp/uploads/7a1afe31-6740-4a3d-8a85-d32a02f8ffc6.png"]

Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]
Loading pipeline components...:  17%|█▋        | 1/6 [00:00<00:00,  7.04it/s]
Loading checkpoint shards:   0%|          | 0/5 [00:00<?, ?it/s][A
Loading checkpoint shards:  40%|████      | 2/5 [00:00<00:00, 18.11it/s][A
Loading checkpoint shards: 100%|██████████| 5/5 [00:00<00:00, 34.62it/s]

Loading pipeline components...:  33%|███▎      | 2/6 [00:00<00:00,  5.25it/s]
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][A
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 137.32it/s]

Loading pipeline components...:  83%|████████▎ | 5/6 [00:00<00:00,  5.39it/s]
Loading pipeline components...: 100%|██████████| 6/6 [00:00<00:00,  6.44it/s]
pipeline loaded
/root/miniconda3/envs/Qwen-Image-Edit-2511/lib/python3.12/site-packages/PIL/Image.py:3577: UserWarning: image file could not be identified because AVIF support not installed
  warnings.warn(message)
Traceback (most recent call last):
  File "/root/Qwen-Image-Edit-2511/run.py", line 25, in <module>
    img_data = Image.open(image_path)
               ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/Qwen-Image-Edit-2511/lib/python3.12/site-packages/PIL/Image.py", line 3579, in open
    raise UnidentifiedImageError(msg)
PIL.UnidentifiedImageError: cannot identify image file '/tmp/uploads/7a1afe31-6740-4a3d-8a85-d32a02f8ffc6.png'
ERROR conda.cli.main_run:execute(127): `conda run bash conda_task_script.sh` failed. (See above for error)
服务器报错:1
SSH连接已断开
SSH连接已断开
错误: 服务器报错:1

```

请分析上述错误信息，并修改配置文件以解决这些问题。
可能的问题包括：
- 安装命令错误（缺少依赖、安装顺序错误等）
- 推理命令错误（命令格式、参数引用等）
- 目录创建规则不符合要求
- inputs字段定义不完整或不正确
- 其他配置问题

请生成修正后的完整配置文件。

## 配置文件规范

### JSON结构说明

```json
{
  "app_id": "应用唯一标识",
  "name": "应用名称",
  "description": "应用描述",
  "author": "作者",
  "homepage": "项目主页URL",
  "email": "联系邮箱（可选）",

  "inputs": [
    {
      "label": "参数标签",
      "field_name": "字段名",
      "type": "类型（text/textarea/number/select/file）",
      "default_value": "默认值",
      "editable": true,
      "required": true,
      "options": [  // 仅select类型需要
        {"label": "选项标签", "value": "选项值"}
      ]
    }
  ],

  "variables": {
    "work_dir": "$HOME/工作目录路径"
  },

  "server_output": "输出路径",
  "model_path": "模型路径（可选）",

  "installs": [
    {
      "env_name": "conda环境名",
      "python_version": "Python版本（如3.10）",
      "cuda_version": "CUDA版本（如12.1）",
      "commands": [
        "安装命令1",
        "安装命令2",
        "..."
      ]
    }
  ],

  "batch": [
    {
      "name": "任务名称",
      "env_name": "使用的conda环境名",
      "commands": [
        "cd {work_dir}",
        "执行命令1",
        "执行命令2",
        "..."
      ]
    }
  ],

  "uploads": [
    {
      "source_path": "本地源文件路径",
      "server_path": "服务器目标路径"
    }
  ],

  "endpoint": "echo \"{server_output}\""
}
```

### 字段详细说明

**inputs类型说明**:
- `text`: 单行文本输入
- `textarea`: 多行文本输入
- `number`: 数字输入
- `select`: 下拉选择（需要提供options）
- `file`: 文件选择

**变量替换**:
- `{field_name}`: inputs中定义的字段值
- `{work_dir}`: variables中定义的work_dir
- `{server_output}`: server_output的值

**installs.commands常用命令**:
⚠️ **重要**：运行环境已经自动处理了conda环境的创建和激活，**不要包含**以下命令：
- ❌ 不要使用: `conda create -n {env_name} python=...`
- ❌ 不要使用: `conda activate {env_name}`
- ❌ 不要使用: `conda create` 或 `conda activate` 的任何变体

**应该使用的命令**：
- ✅ `pip install xxx` - 安装Python包
- ✅ `pip install -r requirements.txt` - 批量安装依赖
- ✅ `mkdir -p {model_path}` - 创建模型缓存目录
- ✅ `rm -rf {work_dir}` - 删除旧工作目录
- ✅ `mkdir -p {work_dir}` - 创建新工作目录
- ✅ `git clone https://xxx {work_dir}` - 克隆项目代码
- ✅ `cd {work_dir}` - 切换到项目目录

**常见AI依赖库参考列表**（请根据README和项目类型智能判断需要安装哪些）:

**深度学习框架**（通常必装）:
- `torch` (PyTorch) - 大多数AI项目的基础框架
- `torchvision` - 计算机视觉项目
- `torchaudio` - 音频处理项目
- `tensorflow` - 部分项目使用
- `flux` 或 `flax` - JAX生态框架

**模型管理和下载**:
- `transformers` - Hugging Face模型库（NLP/CV/多模态项目常用）
- `huggingface_hub[cli]` - Hugging Face CLI工具（模型下载必需）
- `diffusers` - 扩散模型库（图像生成、视频生成项目常用，强烈推荐安装）
- `modelscope[framework]` - ModelScope框架（包含所有必需依赖，ModelScope项目必装）
- `modelscope[multi-modal]` - ModelScope多模态框架（包含所有必需依赖，以及图像、视频推理所需的管道依赖，ModelScope多模态项目必装）
**图像和视频处理**:
- `diffusers` - 扩散模型项目（图像生成、视频生成）
- `accelerate` - 分布式训练和推理加速
- `opencv-python` - 图像处理基础库
- `pillow` - 图像处理基础库
- `imageio` 或 `imageio-ffmpeg` - 视频处理
- `moviepy` - 视频编辑

**音频处理**:
- `librosa` - 音频分析
- `soundfile` - 音频文件读写
- `whisper` - OpenAI语音识别

**加速和优化**:
- `flash-attn` - Flash Attention（transformer加速，CUDA编译可能需要时间）
- `xformers` - Transformer优化库
- `bitsandbytes` - 量化支持
- `deepspeed` - 大模型分布式训练

**NLP相关**:
- `tokenizers` - 分词器
- `datasets` - Hugging Face数据集
- `sentencepiece` - 分词工具

**Web和API**:
- `gradio` - Web界面（很多项目提供Gradio demo）
- `fastapi` - API服务
- `uvicorn` - ASGI服务器

**安装策略和顺序**:
1. **分析README**: 仔细阅读README的安装说明，确定项目类型
2. **基础优先**: 先安装 torch、transformers 等基础框架
3. **平台特定**: ModelScope项目需要 `modelscope[framework]或者modelscope[multi-modal]`，Hugging Face项目需要 `transformers` 和 `huggingface_hub[cli]`
4. **⚠️ 切记：不要包含conda环境创建和激活命令**，环境已由运行环境自动处理
5. **安装命令示例**:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate diffusers opencv-python
   pip install modelscope[framework]  # 如果是ModelScope项目（必须安装完整框架）
   pip install modelscope[multi-modal] # ModelScope多模态框架（包含所有必需依赖，以及图像、视频推理所需的管道依赖，ModelScope多模态项目必装）
   mkdir -p {work_dir}
   git clone https://github.com/user/repo {work_dir}
   cd {work_dir}
   pip install -r requirements.txt
   ```

**依赖判断规则**:
- **扩散模型项目** ( Stable Diffusion、FLUX等): torch, diffusers, transformers, accelerate, opencv-python
- **LLM/文本生成**: torch, transformers, accelerate, tokenizers
- **视频生成**: torch, diffusers, transformers, opencv-python, imageio-ffmpeg
- **音频处理**: torch, transformers, librosa, soundfile
- **计算机视觉**: torch, torchvision, opencv-python, pillow

**目录创建要求**:
- ⚠️ **极其重要：配置文件中定义的所有目录路径，必须按照规则提前创建或清理！**

**1. model_path（模型存储目录）- 在installs阶段创建**
  - 处理方式：直接创建目录（不删除）
  - 命令：`mkdir -p {model_path}`
  - 示例：`mkdir -p $HOME/.cache/stable-diffusion`
  - 原因：模型文件可以缓存在本地，避免重复下载

**2. work_dir（工作目录）- 在installs阶段删除后创建**
  - 处理方式：先删除旧目录，确保全新安装
  - 命令序列：
    ```
    rm -rf {work_dir}
    mkdir -p {work_dir}
    ```
  - 原因：清除旧文件，避免冲突，确保每次都是干净的安装环境

**3. server_output（输出路径）- 在batch阶段处理**
  - **情况A：server_output作为目录使用**（多个输出文件）
    - 处理方式：先删除旧目录，再创建新目录
    - 命令序列：
      ```
      rm -rf {server_output}
      mkdir -p {server_output}
      ```
    - 示例：`rm -rf /tmp/output && mkdir -p /tmp/output`

  - **情况B：server_output作为文件使用**（单个输出文件）
    - 处理方式：只创建父目录
    - 命令：`mkdir -p $(dirname {server_output})`
    - 示例：`mkdir -p $(dirname /tmp/result.png)`  # 创建/tmp目录
    - 说明：不删除文件，直接覆盖输出文件即可

**4. 判断server_output是目录还是文件**:
  - 根据README和推理逻辑判断：
    - 如果推理输出多个文件（视频帧+音频、分割结果等）→ 作为目录
    - 如果推理输出单个文件（生成一张图片、一个文本等）→ 作为文件
  - 当不确定时，默认作为目录处理（更安全）

**batch数组字段说明**:
- ⚠️ **极其重要：batch数组中每个元素的name字段必须是中文！**
- **name字段**: 描述该步骤的功能，必须使用中文（如："模型推理"、"图像生成"、"视频处理"）
- **env_name字段**: 使用的conda环境名称
- **commands字段**: 执行的命令列表

**batch.commands注意事项**:
- 第一个命令应该是处理server_output目录/文件
- 第二个命令应该是 `cd {work_dir}`
- 使用 `{field_name}` 引用inputs参数
- 确保命令能实际执行模型推理
- **server_output处理示例**：
  ```
  # 作为目录使用（多文件输出）
  rm -rf {server_output}
  mkdir -p {server_output}
  
  # 作为文件使用（单文件输出）
  mkdir -p $(dirname {server_output})
  ```
**batch.commands推理程序构建规范**:
- ⚠️ **极其重要：推理程序必须在batch阶段执行！**
- ❌ **禁止在installs阶段构建推理程序**
- ✅ **必须在batch.commands中直接执行推理命令**

**推荐执行方式**（根据项目类型选择）：

**方式1：使用python -c（推荐，适用于Python推理）**
```bash
# batch.commands 示例 - ModelScope项目
[
  \"mkdir -p $(dirname {server_output})\",
  \"python -c \\"from modelscope.pipelines import pipeline; pipe = pipeline('image-captioning', 'model-id'); result = pipe('{image_path}'); print(result)\""
]
```

**方式2：直接使用shell命令（适用于工具类推理）**
```bash
# batch.commands 示例 - 视频处理
[
  \"ffmpeg -i {input_video} -vf scale=1920:1080 {server_output}/output.mp4\"
]
```

**方式3：复杂逻辑才创建脚本文件（仅特殊情况）**
```bash
# 当需要多步骤复杂逻辑时
[
  \"cat > process.sh << 'EOF'\",
  \"#!/bin/bash\",
  \"step1_output=$(cmd1)\",
  \"cmd2 $step1_output\",
  \"EOF\",
  \"chmod +x process.sh && ./process.sh\"
]
```

**代码质量要求**:
- ✅ **简洁明了**: 只包含必要的逻辑
- ✅ **清晰可读**: 代码格式规范，易于理解
- ✅ **直接输出**: 使用 > 或 >> 将结果写入{server_output}
- ✅ **错误处理**: 添加基本的try-except（Python）或set -e（Shell）

**参数传递规则**（规避bash转义错误）:
- ✅ **优先使用python -c**: 代码简洁，无需创建文件
- ✅ **直接引用变量**: 在python -c中直接使用{variable}
- ✅ **正确转义**: 转义引号和特殊字符
- ❌ **避免heredoc嵌套变量**: 容易产生转义错误
- ❌ **避免复杂嵌套引号**: 保持代码扁平化

**必须避免的模式**:
- ❌ 不要在installs阶段创建inference.py
- ❌ 不要在heredoc内使用{variable}变量
- ❌ 不要使用多层嵌套的引号转义
- ✅ 使用python -c直接执行
- ✅ 使用简单的单引号或双引号
- ✅ 保持命令扁平化

**endpoint字段输出要求**: 根据输出类型选择合适的命令
- **作用**: 告诉前端软件生成结果的文件路径、目录路径或文本内容
- **使用规则**（根据输出类型选择）:

  **情况A：输出文本文件（推荐使用cat直接输出内容）**
  - 适用场景：推理结果保存为.txt、.json、.md等文本文件
  - 使用方式: `cat {文本文件路径}`
  - 示例:
    ```json
    {
      "server_output": "/tmp/output",
      "endpoint": "cat \"{server_output}/result.txt\""
    }
    ```
  - 优点：前端直接获取文本内容，无需再次读取文件

  **情况B：输出二进制文件或目录（使用echo输出路径）**
  - 适用场景：图片、视频、音频、模型文件等二进制文件，或多文件目录
  - 使用方式: `echo {文件/目录路径}`
  - 示例:
    ```json
    {
      "server_output": "/tmp/output",
      "endpoint": "echo \"{server_output}/image.png\""
    }
    ```

- **判断标准**:
  1. 文本文件（.txt, .json, .md, .csv, .xml等）→ 使用cat
  2. 二进制文件（.png, .jpg, .mp4, .mp3, .pth等）→ 使用echo
  3. 目录（多文件输出）→ 使用echo
  4. 不确定时默认使用echo（更通用）

**文本变量占位符换行问题**:
- ⚠️ **极其重要：在Python脚本中处理包含换行的文本变量时需要特别小心！**
- **问题**: 当inputs中的文本字段（textarea类型）包含换行符时，直接嵌入命令会导致Shell解析错误
- **错误示例**:
  ```
  python inference.py --prompt "{prompt}"  # 如果prompt包含换行会报错
  ```
- **正确处理方式**：
  1. 使用环境变量传递多行文本:
     ```
     export PROMPT="{prompt}"
     python inference.py --prompt "$PROMPT"
     ```
  2. 使用临时文件传递多行文本:
     ```
     echo "{prompt}" > /tmp/prompt.txt
     python inference.py --prompt-file /tmp/prompt.txt
     ```
  3. 使用printf代替echo（更安全）:
     ```
     printf '%s' "{prompt}" > /tmp/prompt.txt
     ```
- **判断标准**: 如果inputs中有textarea类型（可能包含换行），必须使用上述方法处理


## 重要提示

1. **保持app_id不变**: app_id: 'Qwen-Image-Edit-2511'
2. **只修改相关部分**: 根据用户要求只修改必要的字段
3. **保持结构完整**: 返回完整的配置对象，不要省略任何字段
4. **不要添加is_cooperation字段**: 这不是工作流配置
5. **返回纯JSON**: 不要包含markdown代码块标记，直接返回JSON对象

请现在生成修改后的完整配置文件（纯JSON格式）：
使用中文对话，构建完毕的应用配置保存为[app_id].json