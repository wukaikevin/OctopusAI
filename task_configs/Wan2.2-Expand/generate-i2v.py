# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import tempfile
import shutil
import subprocess
import time

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import numpy as np

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


def _validate_args(args):
    """验证参数 - 基于官方代码结构"""
    # Basic check
    if not args.ckpt_dir:
        raise ValueError("Please specify the checkpoint directory.")
    
    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")
    
    # 检查图像输入
    if not args.image:
        raise ValueError("Please provide input image for i2v task")
    
    # 检查是否是多提示词模式
    if args.multi_prompt:
        if not args.prompt:
            raise ValueError("Please provide prompts for multi-prompt generation")
        
        # 按换行分割提示词
        args.prompt_list = [p.strip() for p in args.prompt.split('\n') if p.strip()]
        if not args.prompt_list:
            raise ValueError("No valid prompts found")
        
        logging.info(f"Parsed {len(args.prompt_list)} prompts for multi-prompt generation")
    else:
        args.prompt_list = [args.prompt] if args.prompt else []
    
    cfg = WAN_CONFIGS[args.task]
    
    # 设置默认值 - 严格按照官方代码
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    # 检查图像文件是否存在（只在rank 0检查）
    rank = int(os.getenv("RANK", 0))
    if rank == 0 and not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # 检查size是否支持
    if args.task == 'i2v-A14B':
        if args.size not in SIZE_CONFIGS:
            if rank == 0:
                logging.warning(f"Size {args.size} may not be optimized for i2v task")
        # 设置max_area
        if args.size in MAX_AREA_CONFIGS:
            args.max_area = MAX_AREA_CONFIGS[args.size]
            if rank == 0:
                logging.info(f"Using max_area: {args.max_area} for size {args.size}")
    else:
        if rank == 0:
            # 对于其他任务，严格检查size
            if args.size not in SUPPORTED_SIZES.get(args.task, []):
                supported = SUPPORTED_SIZES.get(args.task, [])
                raise ValueError(f"Unsupported size {args.size} for task {args.task}, supported sizes are: {', '.join(supported)}")


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Generate video from image and text prompts using Wan - Support multi-prompt i2v"
    )
    
    # 必需参数
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text (use newlines for multi-prompt)")
    parser.add_argument("--save_file", type=str, required=True, help="Output video file path")
    
    # 任务参数
    parser.add_argument("--task", type=str, default="i2v-A14B", choices=["i2v-A14B"], help="Task to run")
    
    # 视频尺寸参数
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    
    # 多提示词选项
    parser.add_argument("--multi_prompt", action="store_true", help="Enable multi-prompt generation")
    
    # 生成参数 - 与官方代码完全一致
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames per segment")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale")
    parser.add_argument("--base_seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU after each forward")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="Sampling solver")
    
    # 分布式参数 - 与官方代码完全一致
    parser.add_argument("--ulysses_size", type=int, default=1, help="Size of ulysses parallelism in DiT")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 model on CPU")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Convert model parameters dtype")
    
    # 其他参数
    parser.add_argument("--keep_segments", action="store_true", default=False, help="Keep segment files")
    
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    """初始化日志"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def ensure_directory_exists(path):
    """确保目录存在"""
    if not path:
        return "."
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory if directory else "."


def save_segment_video(video_tensor, segment_path, fps=8):
    """保存视频片段并验证"""
    try:
        ensure_directory_exists(segment_path)
        
        save_video(
            tensor=video_tensor[None],
            save_file=segment_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        if os.path.exists(segment_path):
            file_size = os.path.getsize(segment_path)
            if file_size > 0:
                file_size_mb = file_size / 1024 / 1024
                logging.info(f"✓ Segment saved: {segment_path} ({file_size_mb:.2f} MB)")
                return True
            else:
                raise ValueError(f"Segment file is empty: {segment_path}")
        else:
            raise FileNotFoundError(f"Segment file not created: {segment_path}")
            
    except Exception as e:
        logging.error(f"Error saving segment video: {e}")
        raise


def extract_last_frame(video_tensor):
    """从视频张量中提取最后一帧"""
    # video_tensor shape: [C, T, H, W]
    if video_tensor.dim() == 4:
        last_frame = video_tensor[:, -1, :, :]  # 提取最后一帧
        return last_frame
    else:
        raise ValueError(f"Unexpected video tensor shape: {video_tensor.shape}")


def broadcast_pil_image(image, rank):
    """在分布式环境中广播PIL图像 - 简化版本"""
    if not dist.is_initialized():
        return image
    
    if rank == 0:
        # 保存图像到内存
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        img_size = len(img_data)
        
        # 广播大小
        size_tensor = torch.tensor([img_size], dtype=torch.int64, device=rank)
        dist.broadcast(size_tensor, src=0)
        
        # 广播数据
        data_tensor = torch.ByteTensor(list(img_data)).to(rank)
        dist.broadcast(data_tensor, src=0)
        
        return image
    else:
        # 接收大小
        size_tensor = torch.tensor([0], dtype=torch.int64, device=rank)
        dist.broadcast(size_tensor, src=0)
        img_size = size_tensor.item()
        
        # 接收数据
        data_tensor = torch.ByteTensor(img_size).to(rank)
        dist.broadcast(data_tensor, src=0)
        
        # 转换为图像
        import io
        img_data = bytes(data_tensor.cpu().numpy())
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        return image


def tensor_to_image(tensor, rank):
    """将张量转换为PIL Image"""
    try:
        # 确保张量在CPU上
        tensor = tensor.cpu().float()
        
        # 归一化到[0, 1]范围
        tensor = (tensor + 1.0) / 2.0  # 从[-1, 1]转换到[0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为numpy数组并调整维度
        if tensor.dim() == 3:  # [C, H, W]
            img_array = tensor.numpy()
            img_array = np.transpose(img_array, (1, 2, 0))  # 转换为[H, W, C]
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # 转换为PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # 确保图像是RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 在分布式环境中广播图像
        if dist.is_initialized():
            img = broadcast_pil_image(img, rank)
            
        return img
        
    except Exception as e:
        logging.error(f"Error converting tensor to image: {e}")
        raise


def simple_merge_videos(segment_files, output_path):
    """合并视频片段"""
    if not segment_files:
        raise ValueError("No segment files to merge")
    
    if len(segment_files) == 1:
        try:
            shutil.copy2(segment_files[0], output_path)
            logging.info(f"Only one segment, copied to: {output_path}")
            return True
        except Exception as e:
            raise RuntimeError(f"Error copying single segment: {e}")
    
    logging.info(f"Merging {len(segment_files)} segments...")
    
    # 验证所有片段文件都存在
    for seg_file in segment_files:
        if not os.path.exists(seg_file):
            raise FileNotFoundError(f"Segment file not found: {seg_file}")
        if os.path.getsize(seg_file) == 0:
            raise ValueError(f"Segment file is empty: {seg_file}")
    
    # 使用ffmpeg合并
    list_file = None
    try:
        # 创建文件列表
        list_file = output_path + ".txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for seg_file in segment_files:
                abs_path = os.path.abspath(seg_file)
                f.write(f"file '{abs_path}'\n")
        
        # 使用ffmpeg
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"ffmpeg failed with return code {result.returncode}:\n"
            error_msg += f"stderr: {result.stderr[:500]}\n"
            error_msg += f"stdout: {result.stdout[:500]}"
            raise RuntimeError(error_msg)
        
        # 验证输出文件
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file was not created: {output_path}")
        
        if os.path.getsize(output_path) == 0:
            raise ValueError(f"Output file is empty: {output_path}")
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        logging.info(f"✓ Merge successful: {output_path} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        # 清理可能已损坏的输出文件
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise RuntimeError(f"Failed to merge videos: {e}")
    
    finally:
        # 清理临时文件
        if list_file and os.path.exists(list_file):
            try:
                os.unlink(list_file)
            except:
                pass


def generate_i2v_single_segment(wan_i2v, prompt, image, args, seed, segment_index=None, rank=0):
    """生成单个视频片段 - 严格按照官方参数顺序"""
    # 获取size配置
    max_area = getattr(args, 'max_area', None)
    
    if rank == 0 and segment_index is not None:
        logging.info(f"  Generating segment {segment_index}")
        logging.info(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        logging.info(f"  Size: {args.size}, Max area: {max_area}")
        logging.info(f"  Frame num: {args.frame_num}, Steps: {args.sample_steps}")
        logging.info(f"  Seed: {seed}, Offload model: {args.offload_model}")
    
    # 调用官方generate方法 - 严格按照官方参数顺序
    video = wan_i2v.generate(
        prompt,           # 提示词
        image,            # 图像
        max_area=max_area,  # 最大面积
        frame_num=args.frame_num,  # 帧数
        shift=args.sample_shift,  # shift参数
        sample_solver=args.sample_solver,  # 采样器
        sampling_steps=args.sample_steps,  # 采样步数
        guide_scale=args.sample_guide_scale,  # 引导尺度
        seed=seed,  # 种子
        offload_model=args.offload_model  # 是否offload模型
    )
    
    return video


def generate_i2v_multi_prompt(args, wan_i2v, cfg, rank, initial_image):
    """生成多提示词视频并合并"""
    # 确保输出目录存在
    if rank == 0:
        output_dir = ensure_directory_exists(args.save_file)
    
    segment_files = []
    current_image = initial_image
    
    for i, prompt in enumerate(args.prompt_list):
        try:
            if rank == 0:
                logging.info(f"\n{'='*60}")
                logging.info(f"Segment {i+1}/{len(args.prompt_list)}")
            
            # 生成视频片段
            video = generate_i2v_single_segment(
                wan_i2v=wan_i2v,
                prompt=prompt,
                image=current_image,
                args=args,
                seed=args.base_seed + i,
                segment_index=i+1,
                rank=rank
            )
            
            if rank == 0:
                # 创建片段文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_clean = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
                segment_name = f"segment_{i+1:03d}_{args.size.replace('*', 'x')}_{prompt_clean}_{timestamp}.mp4"
                segment_path = os.path.join(output_dir, segment_name) if output_dir != "." else segment_name
                
                # 保存片段
                save_segment_video(video, segment_path, cfg.sample_fps)
                segment_files.append(segment_path)
                
                # 提取最后一帧作为下一段的参考图像
                if i < len(args.prompt_list) - 1:
                    last_frame = extract_last_frame(video)
                    current_image = tensor_to_image(last_frame, rank)
                    if current_image:
                        logging.info(f"  Extracted last frame as reference for next segment")
            
            # 清理内存
            del video
            
        except Exception as e:
            logging.error(f"Error generating segment {i+1}: {e}")
            raise
    
    return segment_files if rank == 0 else []


def generate(args):
    """主生成函数 - 严格按照官方代码结构"""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    # 初始化日志
    _init_logging(rank)
    
    # 设置 offload_model 默认值 - 与官方完全一致
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        if rank == 0:
            logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    
    # 分布式初始化 - 与官方完全一致
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."
    
    # Ulysses序列并行初始化
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()
    
    # 广播随机种子
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
        if rank == 0:
            logging.info(f"Broadcast seed: {args.base_seed}")
    
    if rank == 0:
        logging.info(f"Generation job args: {args}")
        logging.info(f"World size: {world_size}, Rank: {rank}")
    
    # 获取配置
    cfg = WAN_CONFIGS[args.task]
    if rank == 0:
        logging.info(f"Generation model config: {cfg}")
        logging.info(f"Selected size: {args.size}")
        if args.size in SIZE_CONFIGS:
            logging.info(f"Size config: {SIZE_CONFIGS[args.size]}")
        if args.size in MAX_AREA_CONFIGS:
            logging.info(f"Max area: {MAX_AREA_CONFIGS[args.size]}")
    
    # 对于Ulysses并行，检查头数是否能被整除
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
    
    # 加载第一张图像
    current_image = None
    if rank == 0:
        current_image = Image.open(args.image).convert("RGB")
        logging.info(f"Loaded input image: {args.image}")
        logging.info(f"Image size: {current_image.size}")
        
        # 设置max_area参数
        if args.size in MAX_AREA_CONFIGS:
            args.max_area = MAX_AREA_CONFIGS[args.size]
            logging.info(f"Using max_area: {args.max_area} for size {args.size}")
    
    # 在分布式环境中广播图像
    if dist.is_initialized():
        current_image = broadcast_pil_image(current_image, rank)
    
    # 广播提示词列表
    if args.multi_prompt:
        if dist.is_initialized():
            if rank == 0:
                prompt_data = [args.prompt_list]
            else:
                prompt_data = [None]
            dist.broadcast_object_list(prompt_data, src=0)
            args.prompt_list = prompt_data[0]
    
    if rank == 0:
        logging.info(f"Input prompts: {len(args.prompt_list)} prompts")
        for i, prompt in enumerate(args.prompt_list):
            logging.info(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    # 创建模型管道 - 与官方完全一致
    logging.info("Creating WanI2V pipeline.")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    
    try:
        # 多提示词生成
        if args.multi_prompt and len(args.prompt_list) > 1:
            if rank == 0:
                logging.info(f"Multi-prompt mode with {len(args.prompt_list)} prompts")
            
            segment_files = generate_i2v_multi_prompt(args, wan_i2v, cfg, rank, current_image)
            
            if rank == 0:
                logging.info(f"\n{'='*60}")
                
                if not segment_files:
                    raise RuntimeError("No segments were generated successfully")
                
                logging.info(f"Successfully generated {len(segment_files)} segments")
                
                # 确保最终输出文件名
                if not args.save_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.dirname(args.save_file) if os.path.dirname(args.save_file) else "."
                    args.save_file = os.path.join(output_dir, f"merged_i2v_{args.size.replace('*', 'x')}_{timestamp}.mp4") if output_dir != "." else f"merged_i2v_{args.size.replace('*', 'x')}_{timestamp}.mp4"
                
                # 尝试合并视频
                try:
                    success = simple_merge_videos(segment_files, args.save_file)
                    
                    if success and os.path.exists(args.save_file):
                        file_size = os.path.getsize(args.save_file) / 1024 / 1024
                        logging.info(f"\n✅ Final video created: {args.save_file} ({file_size:.2f} MB)")
                        
                        # 是否保留片段文件
                        if not args.keep_segments:
                            for seg_file in segment_files:
                                try:
                                    os.unlink(seg_file)
                                    logging.info(f"Cleaned up segment: {seg_file}")
                                except Exception as e:
                                    logging.warning(f"Failed to cleanup segment {seg_file}: {e}")
                        else:
                            logging.info(f"Segment files kept in: {output_dir}")
                    else:
                        raise RuntimeError("Merge completed but output file validation failed")
                        
                except Exception as e:
                    logging.error(f"\n❌ Failed to merge video segments:")
                    logging.error(f"Error: {e}")
                    output_dir = os.path.dirname(args.save_file) if os.path.dirname(args.save_file) else "."
                    if output_dir:
                        logging.error(f"\nSegment files are still available in: {output_dir}")
                    raise
                
                logging.info(f"✅ Generation and merging completed successfully!")
                logging.info(f"Output file: {args.save_file}")
        
        else:
            # 单提示词模式
            if rank == 0:
                logging.info(f"Single prompt mode")
                logging.info(f"Generating video from image and prompt...")
            
            prompt = args.prompt_list[0] if args.prompt_list else args.prompt
            
            # 获取max_area
            max_area = getattr(args, 'max_area', MAX_AREA_CONFIGS.get(args.size))
            
            # 使用与官方完全一致的参数顺序和命名
            video = wan_i2v.generate(
                prompt,
                current_image,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)
            
            if rank == 0:
                # 确保目录存在
                ensure_directory_exists(args.save_file)
                
                if args.save_file is None:
                    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
                    suffix = '.mp4'
                    args.save_file = f"{args.task}_i2v_{args.size.replace('*', 'x')}_{formatted_prompt}_{formatted_time}" + suffix
                
                logging.info(f"Saving generated video to {args.save_file}")
                save_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                
                if os.path.exists(args.save_file):
                    file_size = os.path.getsize(args.save_file) / 1024 / 1024
                    logging.info(f"✅ Video saved: {args.save_file} ({file_size:.2f} MB)")
                else:
                    raise RuntimeError(f"Video file not created: {args.save_file}")
            
            del video
    
    finally:
        # 清理和同步
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    
    if rank == 0:
        logging.info("Finished.")


if __name__ == "__main__":
    try:
        args = _parse_args()
        generate(args)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)