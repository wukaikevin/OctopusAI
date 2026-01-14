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
    
    # 检查是否是多提示词模式
    if args.multi_prompt and "t2v" in args.task:
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
    
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    # 简化大小检查
    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size: {args.size}")


def _parse_args():
    """解析命令行参数 - 保持原有参数，增加分布式相关参数"""
    parser = argparse.ArgumentParser(
        description="Generate video from text prompt using Wan - Support multi-prompt"
    )
    
    # 必需参数
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text (use newlines for multi-prompt)")
    
    # 基本参数
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=["t2v-A14B"], help="Task to run")
    parser.add_argument("--size", type=str, default="720*1280", help="Video size (width*height)")
    parser.add_argument("--save_file", type=str, required=True, help="Output video file path")
    
    # 多提示词选项
    parser.add_argument("--multi_prompt", action="store_true", help="Enable multi-prompt generation")
    
    # 生成参数
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames to generate")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale")
    parser.add_argument("--base_seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="Sampling solver")
    
    # 分布式参数（与官方代码一致）
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    
    # 其他参数
    parser.add_argument(
        "--keep_segments",
        action="store_true",
        default=False)
    
    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    """初始化日志 - 基于官方代码"""
    if rank == 0:
        # set format
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
        logging.info(f"Created directory: {directory}")
    return directory if directory else "."


def save_segment_video(video_tensor, segment_path, fps=8):
    """保存视频片段并验证"""
    try:
        # 确保目录存在
        ensure_directory_exists(segment_path)
        
        logging.info(f"Saving segment to: {segment_path}")
        
        # 保存视频
        save_video(
            tensor=video_tensor[None],
            save_file=segment_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        
        # 检查文件是否成功创建
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
    
    # 尝试使用ffmpeg合并
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
        
        logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
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
                logging.info(f"Cleaned up failed output file: {output_path}")
            except:
                pass
        
        # 重新抛出异常
        raise RuntimeError(f"Failed to merge videos: {e}")
    
    finally:
        # 清理临时文件
        if list_file and os.path.exists(list_file):
            try:
                os.unlink(list_file)
            except:
                pass


def generate_t2v_multi_prompt(args, wan_t2v, cfg, rank):
    """生成多提示词视频并合并"""
    # 确保输出目录存在（只在rank 0上创建）
    if rank == 0:
        output_dir = ensure_directory_exists(args.save_file)
    
    segment_files = []
    
    for i, prompt in enumerate(args.prompt_list):
        try:
            if rank == 0:
                logging.info(f"\n{'='*60}")
                logging.info(f"Generating segment {i+1}/{len(args.prompt_list)}")
                logging.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            # 生成视频
            video = wan_t2v.generate(
                prompt,  # 提示词作为第一个参数
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed + i,
                offload_model=args.offload_model)
            
            if rank == 0:
                # 创建片段文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prompt_clean = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
                segment_name = f"segment_{i+1:03d}_{prompt_clean}_{timestamp}.mp4"
                segment_path = os.path.join(output_dir, segment_name) if output_dir != "." else segment_name
                
                # 保存片段
                save_segment_video(video, segment_path, cfg.sample_fps)
                segment_files.append(segment_path)
                logging.info(f"✓ Segment {i+1} saved successfully")
            
            # 清理内存
            del video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error generating segment {i+1}: {e}")
            raise
    
    return segment_files if rank == 0 else []


def generate(args):
    """主生成函数 - 基于官方代码结构"""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    # 初始化日志
    _init_logging(rank)
    
    # 设置 offload_model 默认值（与官方代码一致）
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        if rank == 0:
            logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    
    # 分布式初始化（与官方代码一致）
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
    
    # Ulysses序列并行初始化（与官方代码一致）
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()
    
    # 广播随机种子（与官方代码一致）
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
    
    # 对于Ulysses并行，检查头数是否能被整除
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
    
    # 广播提示词列表（在多GPU环境下）
    if args.multi_prompt and "t2v" in args.task:
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
    
    # 创建模型管道（与官方代码一致）
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
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
    
    # 多提示词生成
    if args.multi_prompt and len(args.prompt_list) > 1:
        if rank == 0:
            logging.info(f"Multi-prompt mode with {len(args.prompt_list)} prompts")
        
        try:
            segment_files = generate_t2v_multi_prompt(args, wan_t2v, cfg, rank)
            
            if rank == 0:
                logging.info(f"\n{'='*60}")
                
                if not segment_files:
                    raise RuntimeError("No segments were generated successfully")
                
                logging.info(f"Successfully generated {len(segment_files)} segments")
                
                # 确保最终输出文件名
                if not args.save_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.dirname(args.save_file) if os.path.dirname(args.save_file) else "."
                    args.save_file = os.path.join(output_dir, f"merged_{timestamp}.mp4") if output_dir != "." else f"merged_{timestamp}.mp4"
                
                # 检查ffmpeg是否可用
                try:
                    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError("ffmpeg command failed. Make sure ffmpeg is installed and in PATH.")
                    if rank == 0:
                        logging.info(f"ffmpeg version: {result.stdout.split('\n')[0] if result.stdout else 'Unknown'}")
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg not found. Please install ffmpeg: apt-get install ffmpeg or conda install ffmpeg")
                
                # 尝试合并视频
                try:
                    success = simple_merge_videos(segment_files, args.save_file)
                    
                    if success and os.path.exists(args.save_file):
                        file_size = os.path.getsize(args.save_file) / 1024 / 1024
                        logging.info(f"\n✅ Final video created: {args.save_file} ({file_size:.2f} MB)")
                        
                        # 不自动删除片段文件
                        output_dir = os.path.dirname(args.save_file) if os.path.dirname(args.save_file) else "."
                        logging.info(f"Segment files kept in: {output_dir}")
                    else:
                        raise RuntimeError("Merge completed but output file validation failed")
                        
                except Exception as e:
                    logging.error(f"\n❌ Failed to merge video segments:")
                    logging.error(f"Error: {e}")
                    if output_dir:
                        logging.error(f"\nSegment files are still available in: {output_dir}")
                    raise
                
                logging.info(f"✅ Generation and merging completed successfully!")
                logging.info(f"Output file: {args.save_file}")
            
        except Exception as e:
            if rank == 0:
                logging.error(f"\n{'='*60}")
                logging.error(f"❌ Generation failed with error:")
                logging.error(f"{e}")
            if world_size > 1:
                dist.destroy_process_group()
            sys.exit(1)
    else:
        # 单提示词模式
        if rank == 0:
            logging.info(f"Single prompt mode")
        
        prompt = args.prompt_list[0] if args.prompt_list else args.prompt
        
        if rank == 0:
            logging.info(f"Generating video ...")
        
        video = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
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
                args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            
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
    
    # 清理和同步（与官方代码一致）
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