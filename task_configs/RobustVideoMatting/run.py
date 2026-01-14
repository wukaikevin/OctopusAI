#!/usr/bin/env python3
"""
RobustVideoMatting 通用动态适配版 - 支持图像和视频背景
修复批处理错误 + 保持原分辨率 + 自动GPU适配 + 多类型背景支持
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import warnings
import subprocess
import json
import logging
import time
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class BackgroundType(Enum):
    """背景类型枚举"""
    IMAGE = "image"
    VIDEO = "video"

@dataclass
class AdaptiveConfig:
    """动态适配配置"""
    # 根据显存自动计算的批处理大小
    batch_size: int = 1
    # 视频处理
    keep_original_resolution: bool = True
    downsample_ratio: float = 0.25
    # 模型选择
    use_resnet50: bool = False
    # 音频
    keep_audio: bool = True
    # 设备
    device: str = "cuda"
    # 背景模式
    background_mode: str = "static"  # static, loop, sync
    # 背景缩放模式
    bg_resize_mode: str = "fit"  # fit, crop, stretch

class BackgroundProcessor:
    """背景处理器"""
    
    def __init__(self, bg_path: str, target_resolution: Tuple[int, int], 
                 bg_mode: str = "static", resize_mode: str = "fit"):
        """
        初始化背景处理器
        
        Args:
            bg_path: 背景文件路径
            target_resolution: 目标分辨率 (width, height)
            bg_mode: 背景模式 (static, loop, sync)
            resize_mode: 缩放模式 (fit, crop, stretch)
        """
        self.bg_path = bg_path
        self.target_resolution = target_resolution
        self.bg_mode = bg_mode
        self.resize_mode = resize_mode
        
        # 检测背景类型
        self.bg_type = self._detect_background_type()
        logger.info(f"检测到背景类型: {self.bg_type.value}")
        
        # 初始化背景
        if self.bg_type == BackgroundType.IMAGE:
            self._init_image_background()
        elif self.bg_type == BackgroundType.VIDEO:
            self._init_video_background()
    
    def _detect_background_type(self) -> BackgroundType:
        """检测背景文件类型"""
        path = Path(self.bg_path)
        
        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"背景文件不存在: {self.bg_path}")
        
        # 根据扩展名判断
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        ext = path.suffix.lower()
        if ext in image_exts:
            return BackgroundType.IMAGE
        elif ext in video_exts:
            return BackgroundType.VIDEO
        else:
            # 尝试通过文件内容判断
            try:
                # 尝试作为图像打开
                img = cv2.imread(self.bg_path)
                if img is not None:
                    return BackgroundType.IMAGE
            except:
                pass
            
            # 尝试作为视频打开
            try:
                cap = cv2.VideoCapture(self.bg_path)
                if cap.isOpened():
                    cap.release()
                    return BackgroundType.VIDEO
            except:
                pass
            
            raise ValueError(f"无法识别的背景文件类型: {self.bg_path}")
    
    def _init_image_background(self):
        """初始化图像背景"""
        # 加载图像
        self.bg_image = cv2.imread(self.bg_path)
        if self.bg_image is None:
            raise ValueError(f"无法加载背景图片: {self.bg_path}")
        
        # 转换颜色空间
        self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        self.bg_image = self._resize_background(self.bg_image)
        
        # 转换为tensor
        self.bg_tensor = torch.from_numpy(self.bg_image).permute(2, 0, 1).float() / 255.0
        
        logger.info(f"图像背景已加载: {self.bg_image.shape[1]}x{self.bg_image.shape[0]}")
    
    def _init_video_background(self):
        """初始化视频背景"""
        # 打开视频
        self.bg_video_cap = cv2.VideoCapture(self.bg_path)
        if not self.bg_video_cap.isOpened():
            raise ValueError(f"无法打开背景视频: {self.bg_path}")
        
        # 获取视频信息
        self.bg_video_fps = self.bg_video_cap.get(cv2.CAP_PROP_FPS)
        self.bg_video_width = int(self.bg_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.bg_video_height = int(self.bg_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bg_video_total_frames = int(self.bg_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"视频背景已加载: {self.bg_video_width}x{self.bg_video_height}, "
                   f"{self.bg_video_fps:.1f} FPS, {self.bg_video_total_frames} 帧")
        
        # 预加载第一帧
        ret, first_frame = self.bg_video_cap.read()
        if not ret:
            raise ValueError("无法读取背景视频的第一帧")
        
        self.bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
        
        # 将第一帧作为默认背景
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_resized = self._resize_background(first_frame_rgb)
        self.bg_tensor = torch.from_numpy(first_frame_resized).permute(2, 0, 1).float() / 255.0
    
    def _resize_background(self, bg_frame: np.ndarray) -> np.ndarray:
        """调整背景大小到目标分辨率"""
        target_w, target_h = self.target_resolution
        bg_h, bg_w = bg_frame.shape[:2]
        
        if self.resize_mode == "stretch":
            # 直接拉伸
            return cv2.resize(bg_frame, (target_w, target_h))
        
        elif self.resize_mode == "fit":
            # 保持宽高比，填充黑边
            scale = min(target_w / bg_w, target_h / bg_h)
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            
            resized = cv2.resize(bg_frame, (new_w, new_h))
            
            # 创建黑色画布
            canvas = np.zeros((target_h, target_w, 3), dtype=bg_frame.dtype)
            
            # 计算居中位置
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # 将调整后的图像放到画布上
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        
        elif self.resize_mode == "crop":
            # 保持宽高比，裁剪
            scale = max(target_w / bg_w, target_h / bg_h)
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            
            resized = cv2.resize(bg_frame, (new_w, new_h))
            
            # 计算裁剪区域
            x_start = (new_w - target_w) // 2
            y_start = (new_h - target_h) // 2
            
            cropped = resized[y_start:y_start+target_h, x_start:x_start+target_w]
            
            return cropped
        
        else:
            # 默认使用fit模式
            return cv2.resize(bg_frame, (target_w, target_h))
    
    def get_background_tensor(self, device: str = "cuda") -> torch.Tensor:
        """获取当前背景tensor"""
        if device == "cuda" and torch.cuda.is_available():
            return self.bg_tensor.cuda()
        return self.bg_tensor
    
    def get_next_video_frame(self) -> Optional[np.ndarray]:
        """获取视频背景的下一帧"""
        if self.bg_type != BackgroundType.VIDEO:
            return None
        
        ret, frame = self.bg_video_cap.read()
        if not ret:
            if self.bg_mode == "loop":
                # 循环播放
                self.bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.bg_video_cap.read()
                if not ret:
                    return None
            else:
                # 播放完毕，返回最后一帧
                return None
        
        # 转换为RGB并调整大小
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self._resize_background(frame_rgb)
        
        # 更新tensor
        self.bg_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        
        return frame_resized
    
    def reset_video(self):
        """重置视频到开头"""
        if self.bg_type == BackgroundType.VIDEO:
            self.bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'bg_video_cap') and self.bg_video_cap is not None:
            self.bg_video_cap.release()

class AdaptiveProcessor:
    """自适应处理器"""
    
    def __init__(self):
        self.model = None
        self.rec_states = None  # 用于存储递归状态
        self.background_processor = None
        
    def calculate_batch_size(self, gpu_memory_gb: float, model_variant: str, 
                           resolution: Tuple[int, int]) -> int:
        """根据显存动态计算批处理大小"""
        # 基础显存需求估算（1080p，单帧）
        base_memory_per_frame = 0.5  # GB
        if model_variant == "resnet50":
            base_memory_per_frame *= 1.5
        
        # 根据分辨率调整
        pixels = resolution[0] * resolution[1]
        scale_factor = pixels / (1920 * 1080)  # 相对于1080p
        
        # 可用显存（保留2GB给系统和递归状态）
        available_memory = max(1.0, gpu_memory_gb - 2.0)
        
        # 计算最大批处理大小
        max_batch = int(available_memory / (base_memory_per_frame * scale_factor))
        
        # 限制范围
        batch_size = max(1, min(max_batch, 32))
        
        logger.info(f"显存: {gpu_memory_gb:.1f}GB, 分辨率: {resolution}, 计算批处理大小: {batch_size}")
        return batch_size
    
    def load_model(self, variant: str = "mobilenetv3", device: str = "cuda"):
        """加载模型"""
        try:
            # 使用TorchHub加载
            model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                variant,
                pretrained=True,
                trust_repo=True
            )
            logger.info(f"✅ 加载 {variant} 模型成功")
        except Exception as e:
            logger.warning(f"TorchHub加载失败: {e}")
            try:
                # 本地加载
                from model import MattingNetwork
                model = MattingNetwork(variant=variant)
                checkpoint = f'rvm_{variant}.pth'
                model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
                logger.info(f"✅ 本地加载 {variant} 成功")
            except Exception as e2:
                logger.error(f"所有加载方式都失败: {e2}")
                raise
        
        # 设置为评估模式
        model = model.eval()
        
        # 移动到设备
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            logger.info("模型已移动到CUDA")
        else:
            model = model.cpu()
            device = "cpu"
            logger.info("模型在CPU上运行")
        
        self.model = model
        return model, device
    
    def initialize_rec_states(self, batch_size: int, device: str):
        """初始化递归状态 - 关键修复：确保状态与批次大小匹配"""
        # 初始状态为None，让模型在第一次推理时自动创建正确形状的状态
        self.rec_states = [None] * 4
        logger.info(f"初始化递归状态，批次大小: {batch_size}")
    
    def process_single_frame(self, frame: np.ndarray, bg_tensor: torch.Tensor, 
                           downsample_ratio: float = 0.25) -> np.ndarray:
        """处理单帧 - 避免批处理的递归状态问题"""
        # 转换为tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        # 移动到设备
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
            bg_tensor = bg_tensor.cuda()
        
        # 推理
        with torch.no_grad():
            fgr, pha, *self.rec_states = self.model(
                frame_tensor, *self.rec_states, downsample_ratio
            )
        
        # 合成背景
        composite = fgr * pha + bg_tensor * (1 - pha)
        
        # 转换回numpy
        result = composite[0].cpu().numpy() * 255
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = np.transpose(result, (1, 2, 0))
        
        return result
    
    def process_video_frames(self, frames: List[np.ndarray], bg_processor: BackgroundProcessor,
                           downsample_ratio: float = 0.25) -> List[np.ndarray]:
        """处理帧列表 - 支持动态背景"""
        results = []
        
        for i, frame in enumerate(frames):
            # 获取当前背景tensor
            bg_tensor = bg_processor.get_background_tensor()
            
            # 处理当前帧
            result = self.process_single_frame(frame, bg_tensor, downsample_ratio)
            results.append(result)
            
            # 如果是视频背景，获取下一帧
            if bg_processor.bg_type == BackgroundType.VIDEO:
                bg_processor.get_next_video_frame()
            
            if (i + 1) % 50 == 0:
                logger.info(f"  处理帧: {i + 1}/{len(frames)}")
        
        return results
    
    def process_video_adaptive(self, source_video: str, background: str, 
                             output_video: str, config: AdaptiveConfig) -> bool:
        """自适应处理视频 - 支持图像和视频背景"""
        logger.info("开始自适应视频处理...")
        
        # 检查文件
        if not Path(source_video).exists():
            logger.error(f"源视频不存在: {source_video}")
            return False
        
        if not Path(background).exists():
            logger.error(f"背景文件不存在: {background}")
            return False
        
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix="rvm_adaptive_"))
        temp_video = temp_dir / "processed_no_audio.mp4"
        
        try:
            # 1. 获取视频信息
            cap = cv2.VideoCapture(source_video)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {source_video}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            logger.info(f"视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")
            logger.info(f"保持原分辨率: {width}x{height}")
            
            # 2. 动态选择模型
            model_variant = "resnet50" if config.use_resnet50 else "mobilenetv3"
            
            # 3. 动态计算批处理大小
            if config.device == "cuda" and torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                batch_size = self.calculate_batch_size(
                    gpu_memory_gb, model_variant, (width, height)
                )
            else:
                batch_size = 1  # CPU模式
            
            # 限制批处理大小避免递归状态错误
            safe_batch_size = min(batch_size, 16)  # 保守一点
            logger.info(f"使用批处理大小: {safe_batch_size}")
            
            # 4. 加载模型
            model, actual_device = self.load_model(model_variant, config.device)
            
            # 5. 初始化背景处理器
            self.background_processor = BackgroundProcessor(
                bg_path=background,
                target_resolution=(width, height),
                bg_mode=config.background_mode,
                resize_mode=config.bg_resize_mode
            )
            
            # 6. 初始化递归状态
            self.initialize_rec_states(safe_batch_size, actual_device)
            
            # 7. 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
            
            if not out.isOpened():
                # 尝试其他编码器
                for codec in ['X264', 'avc1', 'XVID']:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
                    if out.isOpened():
                        logger.info(f"使用编码器: {codec}")
                        break
            
            if not out.isOpened():
                logger.error("无法创建视频写入器")
                return False
            
            # 8. 处理视频（批处理读取，但逐帧处理）
            cap = cv2.VideoCapture(source_video)
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    frames = []
                    
                    # 读取一批帧
                    for _ in range(safe_batch_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 保持原分辨率，只转换颜色空间
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    
                    if not frames:
                        break
                    
                    # 处理这批帧（内部逐帧处理）
                    results = self.process_video_frames(
                        frames, self.background_processor, config.downsample_ratio
                    )
                    
                    # 写入结果
                    for result in results:
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        out.write(result_bgr)
                    
                    frame_count += len(frames)
                    
                    # 显示进度
                    if frame_count % 100 == 0:
                        elapsed = time.time() - start_time
                        fps_rate = frame_count / elapsed if elapsed > 0 else 0
                        progress = (frame_count / total_frames) * 100
                        remaining = (total_frames - frame_count) / fps_rate if fps_rate > 0 else 0
                        
                        logger.info(
                            f"进度: {frame_count}/{total_frames} ({progress:.1f}%) | "
                            f"速度: {fps_rate:.1f} FPS | "
                            f"剩余: {remaining:.0f}s"
                        )
                        
            except Exception as e:
                logger.error(f"处理过程中出错: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            finally:
                cap.release()
                out.release()
                if self.background_processor:
                    self.background_processor.close()
            
            # 9. 性能报告
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            logger.info("=" * 50)
            logger.info("性能报告:")
            logger.info(f"总帧数: {frame_count}")
            logger.info(f"总时间: {total_time:.2f}秒")
            logger.info(f"平均速度: {avg_fps:.2f} FPS")
            logger.info(f"背景类型: {self.background_processor.bg_type.value}")
            
            if config.device == "cuda" and torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"最大显存使用: {memory_used:.2f}GB")
            
            if frame_count == 0:
                logger.error("没有处理任何帧")
                return False
            
            # 10. 音频处理
            if config.keep_audio:
                logger.info("合并音频...")
                self.merge_audio_safely(source_video, str(temp_video), output_video)
            else:
                shutil.copy(str(temp_video), output_video)
            
            # 11. 验证输出
            if Path(output_video).exists():
                file_size = Path(output_video).stat().st_size / (1024 * 1024)
                logger.info(f"✅ 处理完成! 输出文件: {output_video} ({file_size:.1f} MB)")
                return True
            else:
                logger.error("输出文件未创建")
                return False
                
        except Exception as e:
            logger.error(f"处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 清理
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def merge_audio_safely(self, source_video: str, video_no_audio: str, output_video: str):
        """安全合并音频"""
        try:
            # 检查是否有音频
            has_audio = False
            try:
                cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a',
                      '-show_entries', 'stream=codec_type', '-of', 'json', source_video]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    has_audio = 'streams' in data and len(data['streams']) > 0
            except:
                pass
            
            if not has_audio:
                logger.info("源视频没有音频，直接复制视频")
                shutil.copy(video_no_audio, output_video)
                return
            
            logger.info("开始合并音频...")
            
            # 使用最稳定的软件编码器
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_no_audio,
                '-i', source_video,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-movflags', '+faststart',
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ 音频合并成功")
            else:
                logger.warning("音频合并失败，使用无音频版本")
                shutil.copy(video_no_audio, output_video)
                
        except Exception as e:
            logger.warning(f"音频处理异常: {e}")
            shutil.copy(video_no_audio, output_video)

def main():
    parser = argparse.ArgumentParser(
        description='RobustVideoMatting - 自适应GPU版本 (支持图像和视频背景)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用图像背景，自动适配GPU
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4
  
  # 使用视频背景，循环播放
  python rvm_adaptive.py --source input.mp4 --background bg_video.mp4 --output result.mp4
  
  # 使用resnet50模型 (需要更多显存)
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4 --resnet
  
  # CPU模式
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4 --device cpu
  
  # 设置背景模式
  python rvm_adaptive.py --source input.mp4 --background bg_video.mp4 --output result.mp4 --bg-mode loop
  
  # 设置背景缩放模式
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4 --bg-resize crop
  
  # 不保留音频
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4 --no-audio
  
  # 自定义下采样率
  python rvm_adaptive.py --source input.mp4 --background bg.jpg --output result.mp4 --downsample-ratio 0.2
        """
    )
    
    # 基本参数
    parser.add_argument('--source', required=True, help='源视频路径')
    parser.add_argument('--background', required=True, help='背景文件路径 (支持图像或视频)')
    parser.add_argument('--output', default='output.mp4', help='输出视频路径')
    
    # 模型参数
    parser.add_argument('--resnet', action='store_true', help='使用resnet50模型 (更高质量，需要更多显存)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='运行设备: auto(自动), cuda(GPU), cpu')
    parser.add_argument('--downsample-ratio', type=float, default=0.25,
                       help='下采样率 (0.125-1.0，默认0.25)')
    
    # 音频参数
    parser.add_argument('--keep-audio', action='store_true', default=True,
                       help='保留音频 (默认: 是)')
    parser.add_argument('--no-audio', dest='keep_audio', action='store_false',
                       help='不保留音频')
    
    # 背景处理参数
    parser.add_argument('--bg-mode', default='static', choices=['static', 'loop'],
                       help='背景模式: static(静态), loop(循环播放视频背景)')
    parser.add_argument('--bg-resize', default='fit', choices=['fit', 'crop', 'stretch'],
                       help='背景缩放模式: fit(适应并填充), crop(裁剪), stretch(拉伸)')
    
    # 日志参数
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 50)
    logger.info("RobustVideoMatting 自适应GPU版本 - 支持图像/视频背景")
    logger.info("=" * 50)
    
    # 确定设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ 检测到GPU: {gpu_name}, 显存: {memory_gb:.1f}GB")
        else:
            device = 'cpu'
            logger.info("ℹ️  未检测到GPU，使用CPU")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("指定了CUDA但不可用，回退到CPU")
            device = 'cpu'
    
    # 创建配置
    config = AdaptiveConfig(
        batch_size=1,  # 由处理器动态计算
        keep_original_resolution=True,
        downsample_ratio=args.downsample_ratio,
        use_resnet50=args.resnet,
        keep_audio=args.keep_audio,
        device=device,
        background_mode=args.bg_mode,
        bg_resize_mode=args.bg_resize
    )
    
    logger.info("配置信息:")
    logger.info(f"  源视频: {args.source}")
    logger.info(f"  背景文件: {args.background}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"  模型: {'resnet50' if config.use_resnet50 else 'mobilenetv3'}")
    logger.info(f"  设备: {config.device}")
    logger.info(f"  背景模式: {config.background_mode}")
    logger.info(f"  背景缩放: {config.bg_resize_mode}")
    logger.info(f"  下采样率: {config.downsample_ratio}")
    logger.info(f"  保留音频: {config.keep_audio}")
    
    # 创建处理器并运行
    processor = AdaptiveProcessor()
    
    success = processor.process_video_adaptive(
        source_video=args.source,
        background=args.background,
        output_video=args.output,
        config=config
    )
    
    if success:
        logger.info("✅ 处理完成!")
        sys.exit(0)
    else:
        logger.error("❌ 处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()