import sys
import os

# 在导入任何可能使用BigVGAN的模块之前应用修复
mmaudio_path = os.path.expanduser(os.environ.get('HOME', '/root') + '/MMAudio')
if os.path.exists(mmaudio_path):
    sys.path.insert(0, mmaudio_path)

try:
    from huggingface_hub import PyTorchModelHubMixin
    
    # 保存原始方法
    _original_from_pretrained = PyTorchModelHubMixin.from_pretrained
    
    @classmethod
    def _patched_from_pretrained(cls, *args, proxies=None, resume_download=True, **kwargs):
        """修复后的from_pretrained，确保proxies和resume_download有默认值"""
        if proxies is None:
            proxies = {}
        return _original_from_pretrained(cls, *args, proxies=proxies, resume_download=resume_download, **kwargs)
    
    # 应用修复
    PyTorchModelHubMixin.from_pretrained = _patched_from_pretrained
    print('✅ 已应用BigVGANv2运行时修复补丁')
except Exception as e:
    print(f'⚠️ 修复补丁应用失败（可能已修复）: {e}')

