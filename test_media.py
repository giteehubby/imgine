import torch
import numpy as np
from PIL import Image
from media import media_conv
from i2t import jpg2tensor, tensor2pil

def test_media_filter():
    """测试中值滤波函数"""
    # 创建一个简单的测试图像
    # 创建一个3x3的彩色图像，包含一些噪声
    test_image = np.array([
        [[100, 150, 200], [110, 160, 210], [120, 170, 220]],
        [[130, 180, 230], [140, 190, 240], [150, 200, 250]],
        [[160, 210, 255], [170, 220, 255], [180, 230, 255]]
    ], dtype=np.uint8)
    
    # 转换为tensor格式 (C, H, W)
    test_tensor = torch.from_numpy(test_image.transpose(2, 0, 1))
    
    print("原始图像tensor:")
    print(test_tensor)
    print(f"形状: {test_tensor.shape}")
    print(f"数据类型: {test_tensor.dtype}")
    
    # 应用中值滤波
    filtered_tensor = media_conv(test_tensor)
    
    print("\n滤波后图像tensor:")
    print(filtered_tensor)
    print(f"形状: {filtered_tensor.shape}")
    print(f"数据类型: {filtered_tensor.dtype}")
    
    # 检查数值范围
    print(f"\n数值范围: {filtered_tensor.min().item()} - {filtered_tensor.max().item()}")
    
    # 检查是否有NaN或无穷大值
    print(f"是否有NaN: {torch.isnan(filtered_tensor).any().item()}")
    print(f"是否有无穷大: {torch.isinf(filtered_tensor).any().item()}")
    
    return filtered_tensor

if __name__ == "__main__":
    test_media_filter() 