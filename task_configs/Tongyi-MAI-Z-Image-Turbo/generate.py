import sys
import torch
import argparse
from modelscope import ZImagePipeline
from pathlib import Path
import time

def generate_images(args):
    """生成图片的主函数"""
    
    # 初始化管道
    print("正在初始化模型...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    pipe.transformer.set_attention_backend("flash")
    print("模型初始化完成！")
    
    # 创建输出目录（如果不存在）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = map(int, args.resolution.split('*'))
    # 批量生成图片
    for i in range(args.num):
        print(f"\n正在生成第 {i+1}/{args.num} 张图片...")
        
        # 生成图片
        image = pipe(
            prompt=args.prompt,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed((args.seed * i)-1),  # 每次使用不同的种子
        ).images[0]
        
        # 保存图片
        if args.num == 1:
            # 如果只生成一张图片，使用指定的文件名
            filename = args.output
        else:
            # 如果生成多张图片，使用带编号的文件名
            filename = output_dir / f"{Path(args.output).stem}_{i+1:03d}{Path(args.output).suffix}"
        
        image.save(str(filename))
        print(f"图片已保存: {filename}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo 图片生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="图片描述提示词"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.png",
        help="输出图片文件名（当num=1时）或基础文件名"
    )
    
    
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="1288*1920",
        help="生成图片的尺寸"
    )
    
    parser.add_argument(
        "--num", 
        type=int, 
        default=1,
        help="生成图片的数量"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="当生成多张图片时的输出目录"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子基数"
    )
    
    parser.add_argument(
        "--steps", 
        type=int, 
        default=9,
        help="推理步数"
    )
    # 解析参数
    args = parser.parse_args()
    
    # 参数验证
    if args.num < 1:
        parser.error("--num 必须大于0")
    
    # 显示参数信息
    print("=" * 50)
    print("图片生成参数:")
    print(f"提示词: {args.prompt}")
    print(f"尺寸: {args.resolution}")
    print(f"数量: {args.num}")
    print(f"输出: {args.output if args.num == 1 else args.output_dir}")
    print(f"种子: {args.seed}")
    print(f"推理步数: {args.steps}")
    print("=" * 50)
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成图片
    try:
        generate_images(args)
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"生成完成！共生成 {args.num} 张图片")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均每张: {elapsed_time/args.num:.2f} 秒")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()