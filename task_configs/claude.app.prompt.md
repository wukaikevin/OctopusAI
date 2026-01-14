# 任务：生成AI应用模块配置文件

## 项目信息
- 项目网址: https://www.modelscope.cn/models/Lightricks/LTX-2
- 组织: Lightricks
- 项目名: LTX-2
- 建议的app_id: Lightricks-LTX-2

## 项目README.md内容

### 📦 项目来源：ModelScope

**注意**：当前项目来自ModelScope（模型托管平台）
- 如果README内容不够详细，可能是因为该项目主要用于模型托管，而非源码管理
- 建议查看该项目是否有对应的GitHub源码仓库

### 📄 项目README.md
```
---language:- en- de- es- fr- ja- ko- zh- it- ptlibrary_name: diffuserslicense: otherlicense_name: ltx-2-community-license-agreementlicense_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSEpipeline_tag: image-to-videoarxiv: 2601.03233tags:- image-to-video- text-to-video- video-to-video- image-text-to-video- audio-to-video- text-to-audio- video-to-audio- audio-to-audio- text-to-audio-video- image-to-audio-video- image-text-to-audio-video- ltx-2- ltx-video- ltxv- lightrickspinned: truedemo: https://app.ltx.studio/ltx-2-playground/i2v---# LTX-2 Model CardThis model card focuses on the LTX-2 model, as presented in the paper \[LTX-2: Efficient Joint Audio-Visual Foundation Model\](https://huggingface.co/papers/2601.03233). The codebase is available \[here\](https://github.com/Lightricks/LTX-2).LTX-2 is a DiT-based audio-video foundation model designed to generate synchronized video and audio within a single model. It brings together the core building blocks of modern video generation, with open weights and a focus on practical, local execution. \[!\[LTX-2 Open Source\](https://img.youtube.com/vi/8fWAJXZJbRA/maxresdefault.jpg)\](https://www.youtube.com/watch?v=8fWAJXZJbRA)# Model Checkpoints\| Name \| Notes \|\|--------------------------------\|----------------------------------------------------------------------------------------------------------------\|\| ltx-2-19b-dev \| The full model, flexible and trainable in bf16 \|\| ltx-2-19b-dev-fp8 \| The full model in fp8 quantization \|\| ltx-2-19b-dev-fp4 \| The full model in nvfp4 quantization \| \| ltx-2-19b-distilled \| The distilled version of the full model, 8 steps, CFG=1 \|\| ltx-2-19b-distilled-lora-384 \| A LoRA version of the distilled model applicable to the full model \|\| ltx-2-spatial-upscaler-x2-1.0 \| An x2 spatial upscaler for the ltx-2 latents, used in multi stage (multiscale) pipelines for higher resolution \|\| ltx-2-temporal-upscaler-x2-1.0 \| An x2 temporal upscaler for the ltx-2 latents, used in multi stage (multiscale) pipelines for higher FPS \|## Model Details- \*\*Developed by:\*\* Lightricks- \*\*Model type:\*\* Diffusion-based audio-video foundation model- \*\*Language(s):\*\* English# Online demoLTX-2 is accessible right away via the following links:- \[LTX-Studio text-to-video\](https://app.ltx.studio/ltx-2-playground/t2v)- \[LTX-Studio image-to-video\](https://app.ltx.studio/ltx-2-playground/i2v)# Run locally## Direct use licenseYou can use the models - full, distilled, upscalers and any derivatives of the models - for purposes under the \[license\](./LICENSE).## ComfyUIWe recommend you use the built-in LTXVideo nodes that can be found in the ComfyUI Manager. For manual installation information, please refer to our \[documentation site\](https://docs.ltx.video/open-source-model/integration-tools/comfy-ui).## PyTorch codebaseThe \[LTX-2 codebase\](https://github.com/Lightricks/LTX-2) is a monorepo with several packages. From model definition in 'ltx-core' to pipelines in 'ltx-pipelines' and training capabilities in 'ltx-trainer'.The codebase was tested with Python \>=3.12, CUDA version \>12.7, and supports PyTorch \~= 2.7.### Installation\`\`\`bashgit clone https://github.com/Lightricks/LTX-2.gitcd LTX-2# From the repository rootuv syncsource .venv/bin/activate\`\`\`### InferenceTo use our model, please follow the instructions in our \[ltx-pipelines\](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md) package.## Diffusers 🧨LTX-2 is supported in the \[Diffusers Python library\](https://huggingface.co/docs/diffusers/main/en/index) for image-to-video generation.## General tips:\* Width \& height settings must be divisible by 32. Frame count must be divisible by 8 + 1. \* In case the resolution or number of frames are not divisible by 32 or 8 + 1, the input should be padded with -1 and then cropped to the desired resolution and number of frames.\* For tips on writing effective prompts, please visit our \[Prompting guide\](https://ltx.video/blog/how-to-prompt-for-ltx-2) ### Limitations- This model is not intended or able to provide factual information.- As a statistical model this checkpoint might amplify existing societal biases.- The model may fail to generate videos that matches the prompts perfectly.- Prompt following is heavily influenced by the prompting-style.- The model may generate content that is inappropriate or offensive.- When generating audio without speech, the audio may be of lower quality.# Train the modelThe base (dev) model is fully trainable.It's extremely easy to reproduce the LoRAs and IC-LoRAs we publish with the model by following the instructions on the \[LTX-2 Trainer Readme\](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/README.md).Training for motion, style or likeness (sound+appearance) can take less than an hour in many settings.## Citation\`\`\`bibtex@article{hacohen2025ltx2, title={LTX-2: Efficient Joint Audio-Visual Foundation Model}, author={HaCohen, Yoav and Brazowski, Benny and Chiprut, Nisan and Bitterman, Yaki and Kvochko, Andrew and Berkowitz, Avishai and Shalem, Daniel and Lifschitz, Daphna and Moshe, Dudu and Porat, Eitan and Richardson, Eitan and Guy Shiran and Itay Chachy and Jonathan Chetboun and Michael Finkelson and Michael Kupchick and Nir Zabari and Nitzan Guetta and Noa Kotler and Ofir Bibi and Ori Gordon and Poriya Panet and Roi Benita and Shahar Armon and Victor Kulikov and Yaron Inger and Yonatan Shiftan and Zeev Melumian and Zeev Farbman}, journal={arXiv preprint arXiv:2601.03233}, year={2025}}\`\`\`

```

## 用户额外要求
推理示例：https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md

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
    "work_dir": "/工作目录路径"
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
3. **平台特定**: ModelScope项目需要 `modelscope[framework]`，Hugging Face项目需要 `transformers` 和 `huggingface_hub[cli]`
4. **⚠️ 切记：不要包含conda环境创建和激活命令**，环境已由运行环境自动处理
5. **安装命令示例**:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate diffusers opencv-python
   pip install modelscope[framework]  # 如果是ModelScope项目（必须安装完整框架）
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

1. **app_id必须唯一**: 使用 'Lightricks-LTX-2' 作为app_id，确保不与现有应用冲突
2. **应用名称规范**: name字段必须严格遵循格式：`项目功能名称[中文]-项目英文名称`
   - ⚠️⚠️⚠️ **极其重要**：前半部分必须是**纯中文**的功能描述，不能包含英文！
   - 格式：`[中文功能描述]-[项目英文名]`
   - 示例1: `图像生成-stable-diffusion`（✅ 正确：前半是中文功能）
   - 示例2: `文本生成-llama-3`（✅ 正确：前半是中文功能）
   - 示例3: `语音合成-modelscope-tts`（✅ 正确：前半是中文功能）
   - ❌ 错误示例：`Stable Diffusion-stable-diffusion`（前半部分含英文）
   - ❌ 错误示例：`通义万相-Tongyi-Wanxiang`（前半部分是品牌名，不是功能描述）
   - **生成步骤**：
     1. 分析项目功能：这个AI模型是做什么的？（如图像生成、文本生成、语音识别等）
     2. 提取中文功能名：用简洁的中文描述功能（如"图像生成"、"文本生成"）
     3. 使用项目英文名：后半部分直接使用项目URL中的英文名称
3. **准确性优先**: 基于README.md中的实际信息生成配置，不要臆造
4. **语言要求**: 配置文件中的描述和标签必须使用**中文**
   - ✅ `description` 字段：使用中文描述应用功能
   - ✅ `inputs[].label` 字段：使用中文标签（如 "提示词" 而不是 "Prompt"）
   - ✅ `inputs[].options[].label` 字段：使用中文选项标签
   - ❌ 不要使用英文描述（除非是专业术语无法翻译）
   - 示例: {"label": "提示词", "field_name": "prompt", ...}

5. **inputs字段**: 仔细分析README，提取模型的主要输入参数
   **⚠️⚠️⚠️ 极其重要：仅包含推理（Inference）相关的参数！⚠️⚠️⚠️
   - ✅ 应该包含：推理时需要用户提供的参数
     - 提示词/文本输入（prompt, text, input等）
     - 图像/音频/视频输入文件路径
     - 推理参数（如temperature, top_k, steps, cfg_scale等）
     - 输出路径或文件名参数
   - ❌ 不应该包含：与推理业务无关的参数
     - 训练相关参数（learning_rate, batch_size, epochs等）
     - 数据集路径或配置文件路径
     - 模型保存/导出路径
     - 调试/日志相关参数
     - 硬件配置参数（device, num_workers等）
   - **判断标准**：问自己'用户在推理时需要修改这个参数吗？'如果答案是否定的，就不应该放在inputs中
   - **示例对比**：
     - ✅ 正确：prompt（提示词）, image（输入图像）, steps（推理步数）
     - ❌ 错误：train_batch_size（训练参数）, dataset_path（数据集路径）, output_dir（模型保存目录）

6. **installs字段**: 根据README的安装说明生成依赖安装配置
   **⚠️⚠️⚠️ 极其重要：禁止使用conda环境创建和激活命令！⚠️⚠️⚠️
   - ❌ 绝对禁止: `conda create -n xxx python=...`
   - ❌ 绝对禁止: `conda activate xxx`
   - ✅ 只使用: `pip install xxx`, `pip install -r requirements.txt`, `mkdir -p`, `git clone`, `cd` 等
   - 原因: 运行环境已经自动处理了conda环境的创建和激活

   **常见依赖库安装**:
   - 仔细分析项目类型，智能判断并安装必要的常见AI依赖库
   - 即使README中只有 `pip install -r requirements.txt`，你也应该在之前预先安装常见依赖
   - ⚠️⚠️⚠️ **ModelScope项目安装规则**（必须严格遵守）：
     **步骤1: 安装深度学习框架**（先于modelscope安装）
     - PyTorch项目: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     - TensorFlow项目: pip install tensorflow==2.13.0

     **步骤2: 根据模型领域安装ModelScope**（必须使用领域安装）
     - NLP领域: pip install "modelscope[nlp]"
     - CV领域: pip install "modelscope[cv]"
     - Audio领域: pip install "modelscope[audio]"
     - 多模态: pip install "modelscope[multi-modal]"
     - 通用/不确定: pip install "modelscope[all]"

     ⚠️ **禁止使用**: pip install modelscope（缺少领域依赖，会导致运行时错误）
     ✅ **必须使用**: pip install "modelscope[xxx]"（xxx为领域名）

7. **batch字段**: 生成执行模型推理的命令
   ⚠️ **极其重要1：batch数组中每个元素的name字段必须是中文！**
   - **name字段**: 使用中文描述该步骤的功能（如："模型推理"、"图像生成"、"视频处理"）
   - ❌ 不要使用英文（如："Inference", "Generate", "Process"）
   ⚠️ **极其重要3：推理程序构建方式** 
   - ❌ **禁止在installs阶段构建推理程序**：installs.commands只负责环境配置和依赖安装
   - ✅ **必须在batch.commands中直接执行推理命令**
   - **原因**：
     1. installs阶段只执行一次，batch阶段每次任务都会执行
     2. 确保推理使用最新的参数值
     3. 避免硬编码导致的参数引用错误
   - **推理执行方式**（根据模型选择）：
     **方式1：直接使用shell命令（推荐，适用于简单推理）**
     ```
     # ModelScope pipeline方式
     python -c "     from modelscope.pipelines import pipeline;     pipe = pipeline('task-name', 'model-id');     result = pipe('{image_path}');     print(result)     " > {server_output}/result.txt     ```     **方式2：使用shell脚本工具（如ffmpeg、imagemagick等）**
     ```     # 视频处理示例
     ffmpeg -i {input_video} -vf scale=1920:1080 {server_output}/output.mp4
     ```
     **方式3：复杂逻辑才创建脚本文件（特殊情况）**
     ```
     # 仅当逻辑复杂时才创建脚本
     cat > run.sh << 'EOF'
     #!/bin/bash
     step1_output=$(step1_command)
     step2_command $step1_output
     EOF
     chmod +x run.sh && ./run.sh
     ```
   - **参数传递规则**（规避bash转义错误）：
     - ✅ **优先使用python -c**：代码简洁，无需创建文件
     - ✅ **直接引用变量**：在-c参数中直接使用{variable}
     - ✅ **使用双引号包裹**：防止参数中的空格破坏命令结构
     - ❌ **避免在heredoc内使用{variable}**：容易转义错误
   - **python -c代码质量要求**：
     - ✅ 简洁明了：只包含必要的导入和核心逻辑
     - ✅ 清晰可读：使用有意义的变量名
     - ✅ 输出到文件：使用 > 或 >> 将结果写入{server_output}
     - ✅ 基本错误处理：添加try-except处理关键操作
     mkdir -p {work_dir}    # 创建新工作目录
     ```
   - **batch.commands第一个命令**：处理server_output
     ```
     # 作为目录（多文件输出）
     rm -rf {server_output}
     mkdir -p {server_output}
     
     # 作为文件（单文件输出）
     mkdir -p $(dirname {server_output})
     ```
   - **判断标准**：
     - 多文件输出→作为目录处理
     - 单文件输出→作为文件处理
     - 不确定时默认作为目录处理

8. **endpoint字段输出要求**: 根据输出类型选择合适的命令
   - **作用**: 告诉前端软件生成结果的文件路径、目录路径或文本内容
   - **使用规则**（根据输出类型选择）:

     **情况A：输出文本文件（推荐使用cat直接输出内容）**
     - 适用场景：推理结果保存为.txt、.json、.md等文本文件
     - 使用方式: `cat {文本文件路径}`
     - 示例:
       ```
       # 输出单个文本文件内容
       cat \"{server_output}/result.txt\"
       
       # 输出JSON文件内容
       cat \"{server_output}/output.json\"
       ```
     - 优点：前端直接获取文本内容，无需再次读取文件

     **情况B：输出二进制文件或目录（使用echo输出路径）**
     - 适用场景：图片、视频、音频、模型文件等二进制文件，或多文件目录
     - 使用方式: `echo {文件/目录路径}`
     - 示例:
       ```
       # 输出图片路径
       echo \"{server_output}/image.png\"
       
       # 输出视频路径
       echo \"{server_output}/video.mp4\"
       
       # 输出目录路径
       echo \"{server_output}\"
       
       # 输出子目录路径
       echo \"{server_output}/frames\"
       ```
     - 说明：前端根据路径下载或预览文件

   - **判断标准**:
     1. 文本文件（.txt, .json, .md, .csv, .xml等）→ 使用cat
     2. 二进制文件（.png, .jpg, .mp4, .mp3, .pth等）→ 使用echo
     3. 目录（多文件输出）→ 使用echo
     4. 不确定时默认使用echo（更通用）

9. **文本变量换行处理**: 如果inputs中有textarea类型字段，必须使用特殊方法处理
   - ⚠️ **极其重要：多行文本直接嵌入命令会导致Shell解析错误！**
   - **推荐方案1**：使用环境变量传递
     ```
     export PROMPT="{prompt}"
     python inference.py --prompt "$PROMPT"
     ```
   - **推荐方案2**：使用临时文件传递
     ```
     echo "{prompt}" > /tmp/prompt.txt
     python inference.py --prompt-file /tmp/prompt.txt
     ```
   - **判断标准**: 检查inputs中是否有type为textarea的字段，如果有则必须使用上述方法

10. **不要包含is_cooperation字段**: 这不是工作流配置
11. **返回纯JSON**: 不要包含markdown代码块标记，直接返回JSON对象

请现在生成完整的配置文件（纯JSON格式）：
使用中文对话，构建完毕的应用配置保存为[app_id].json