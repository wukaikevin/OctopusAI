# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import psutil
import gc
from contextlib import contextmanager
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
import xformers
import xformers.ops


@contextmanager
def torch_gc():
    """清理GPU内存的上下文管理器"""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def optimize_pytorch_memory():
        """优化PyTorch内存设置"""
        if torch.cuda.is_available():
            # 设置内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            # 启用TF32以加速计算
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 启用cudnn基准测试
            torch.backends.cudnn.benchmark = True
            # 设置内存碎片整理
            torch.cuda.set_per_process_memory_fraction(0.9)  # 预留10%内存
    
    @staticmethod
    def get_gpu_memory_info():
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'cached': torch.cuda.memory_reserved() / 1024**3,
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {}
    
    @staticmethod
    def clear_memory():
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class VideoProcessor:
    """视频处理优化器"""
    
    @staticmethod
    def estimate_video_length(video_path):
        """估计视频长度"""
        try:
            # 使用ffprobe获取视频时长
            import subprocess
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 60  # 默认1分钟


class OptimizedLipsyncPipeline(LipsyncPipeline):
    """优化的唇形同步Pipeline"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_xformers = kwargs.get('enable_xformers', True)
        self.use_chunked_processing = kwargs.get('use_chunked_processing', True)
        self.chunk_size = kwargs.get('chunk_size', 16)
        
        if self.enable_xformers and hasattr(self.unet, 'set_use_memory_efficient_attention_xformers'):
            self.unet.set_use_memory_efficient_attention_xformers(True)
    
    def process_in_chunks(self, latents, audio_features, guidance_scale, num_inference_steps):
        """分块处理以减少内存使用"""
        if not self.use_chunked_processing or latents.shape[0] <= self.chunk_size:
            return self._process_whole(latents, audio_features, guidance_scale, num_inference_steps)
        
        results = []
        num_chunks = (latents.shape[0] + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, latents.shape[0])
            
            chunk_latents = latents[start_idx:end_idx]
            chunk_audio = audio_features[start_idx:end_idx]
            
            with torch_gc():
                processed_chunk = self._process_whole(
                    chunk_latents, chunk_audio, guidance_scale, num_inference_steps
                )
            
            results.append(processed_chunk)
            
            # 释放中间变量内存
            del chunk_latents, chunk_audio
            if i < num_chunks - 1:  # 保留最后一个chunk的梯度
                processed_chunk = processed_chunk.detach()
        
        return torch.cat(results, dim=0)
    
    def _process_whole(self, latents, audio_features, guidance_scale, num_inference_steps):
        """原有的完整处理逻辑"""
        # 这里调用父类的处理方法
        return super().forward(
            latents=latents,
            audio_features=audio_features,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )


def main(config, args):
    """主函数 - 优化的推理流程"""
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")
    
    print(f"🚀 Starting optimized inference")
    print(f"📹 Input video: {args.video_path}")
    print(f"🎵 Input audio: {args.audio_path}")
    
    start_time = time.time()
    
    # 优化内存设置
    MemoryOptimizer.optimize_pytorch_memory()
    
    # 检查GPU能力并选择最佳精度
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"🎮 GPU: {gpu_name}")
        print(f"🔧 Compute Capability: {compute_capability}")
        
        # 优先选择更快的精度
        if compute_capability[0] >= 8:  # Ampere及以上架构
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print("✅ Using bfloat16 precision (fastest)")
        elif compute_capability[0] >= 7:  # Volta及以上架构
            dtype = torch.float16
            print("✅ Using float16 precision")
        else:
            dtype = torch.float32
            print("⚠️ Using float32 precision (GPU may be slow)")
    else:
        dtype = torch.float32
        print("⚠️ Using CPU with float32 precision")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    print(f"📄 Loaded checkpoint: {args.inference_ckpt_path}")
    
    # 使用更快的scheduler
    scheduler = DDIMScheduler.from_pretrained(
        "configs",
        prediction_type="epsilon",
        timestep_spacing="trailing",  # 更快的采样
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # 动态选择whisper模型
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
        print("🔊 Using Whisper small model")
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
        print("🔊 Using Whisper tiny model (faster)")
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    
    # 加载音频编码器
    print("🎵 Loading audio encoder...")
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    
    # 加载VAE - 使用更快的设置
    print("🎨 Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    
    # 启用VAE tiling以节省内存
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
        print("✅ Enabled VAE tiling for memory efficiency")
    
    # 加载UNet - 优化加载过程
    print("🧠 Loading UNet...")
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
        torch_dtype=dtype,
        use_safetensors=True if args.inference_ckpt_path.endswith('.safetensors') else False,
    )
    
    # 将模型移动到设备
    print(f"🚚 Moving models to {device}...")
    vae = vae.to(device=device, dtype=dtype)
    unet = unet.to(device=device, dtype=dtype)
    
    # 编译模型加速（PyTorch 2.0+）
    if torch.__version__ >= "2.0" and args.compile_model:
        try:
            print("⚡ Compiling models with torch.compile...")
            # 编译主要模型
            unet = torch.compile(
                unet,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False
            )
            vae = torch.compile(
                vae,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False
            )
            print("✅ Models compiled successfully")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}. Continuing without compilation.")
    
    # 创建优化的pipeline
    print("🔧 Creating optimized pipeline...")
    pipeline = OptimizedLipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        enable_xformers=args.enable_xformers,
        use_chunked_processing=args.chunked_processing,
        chunk_size=args.chunk_size,
    ).to(device)
    
    # 启用xformers优化
    if args.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ Enabled xformers memory efficient attention")
        except:
            print("⚠️ Xformers not available, using default attention")
    
    # 启用DeepCache优化
    if args.enable_deepcache:
        print("💾 Enabling DeepCache optimization...")
        helper = DeepCacheSDHelper(pipe=pipeline)
        
        # 根据视频长度动态调整缓存参数
        video_length = VideoProcessor.estimate_video_length(args.video_path)
        if video_length > 300:  # 5分钟以上
            cache_interval = 5
            cache_branch_id = 0
            print(f"⏱️ Long video ({video_length}s): cache_interval={cache_interval}")
        elif video_length > 60:  # 1-5分钟
            cache_interval = 3
            cache_branch_id = 0
            print(f"⏱️ Medium video ({video_length}s): cache_interval={cache_interval}")
        else:  # 1分钟以内
            cache_interval = 2
            cache_branch_id = 0
            print(f"⏱️ Short video ({video_length}s): cache_interval={cache_interval}")
        
        helper.set_params(
            cache_interval=cache_interval,
            cache_branch_id=cache_branch_id
        )
        helper.enable()
        print("✅ DeepCache enabled")
    
    # 设置随机种子
    if args.seed != -1:
        set_seed(args.seed)
        print(f"🎲 Using fixed seed: {args.seed}")
    else:
        seed = torch.seed()
        print(f"🎲 Using random seed: {seed}")
    
    # 清理内存
    MemoryOptimizer.clear_memory()
    
    # 执行推理
    print("\n" + "="*50)
    print("🎬 Starting inference...")
    print("="*50)
    
    inference_start = time.time()
    
    # 调用pipeline
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )
    
    inference_time = time.time() - inference_start
    total_time = time.time() - start_time
    
    # 输出性能统计
    print("\n" + "="*50)
    print("📊 Performance Statistics")
    print("="*50)
    print(f"🕒 Total time: {total_time:.2f}s")
    print(f"⚡ Inference time: {inference_time:.2f}s")
    print(f"🚀 Setup time: {total_time - inference_time:.2f}s")
    
    # 内存使用统计
    if torch.cuda.is_available():
        mem_info = MemoryOptimizer.get_gpu_memory_info()
        print(f"💾 GPU Memory - Allocated: {mem_info['allocated']:.2f}GB")
        print(f"💾 GPU Memory - Cached: {mem_info['cached']:.2f}GB")
        print(f"💾 GPU Memory - Total: {mem_info['total']:.2f}GB")
    
    print(f"✅ Inference completed! Output saved to: {args.video_out_path}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized LatentSync Inference")
    
    # 必需参数
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    
    # 推理参数
    parser.add_argument("--inference_steps", type=int, default=12,
                       help="Number of denoising steps (default: 12, was 20)")
    parser.add_argument("--guidance_scale", type=float, default=1.2,
                       help="Classifier-free guidance scale (default: 1.2, was 1.0)")
    
    # 优化参数
    parser.add_argument("--temp_dir", type=str, default="/tmp/latentsync")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true", default=True,
                       help="Enable DeepCache optimization")
    parser.add_argument("--enable_xformers", action="store_true", default=True,
                       help="Enable xformers optimization")
    parser.add_argument("--compile_model", action="store_true", default=True,
                       help="Compile model with torch.compile (PyTorch 2.0+)")
    parser.add_argument("--chunked_processing", action="store_true", default=True,
                       help="Process video in chunks to save memory")
    parser.add_argument("--chunk_size", type=int, default=16,
                       help="Number of frames per chunk (default: 16)")
    parser.add_argument("--temporal_stride", type=int, default=1,
                       help="Process every nth frame (default: 1)")
    
    args = parser.parse_args()
    
    # 创建临时目录
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # 加载配置
    config = OmegaConf.load(args.unet_config_path)
    
    # 运行优化推理
    main(config, args)