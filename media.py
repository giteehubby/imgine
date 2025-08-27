import torch
import torch.nn.functional as F
from torch import median

def media_conv(img_tensor):
    """
    中值滤波 - 优化版本
    使用向量化操作和PyTorch内置函数提高效率
    """
    # if img_tensor.dtype != torch.float32:
    #     img_tensor = img_tensor.float()
    
    # 获取图像尺寸
    channels, height, width = img_tensor.shape
    
    # 创建输出tensor
    output = torch.zeros((channels,height-4,width-4),dtype=img_tensor.dtype)
    
    # 使用unfold操作创建滑动窗口
    # 对于5x5的窗口，我们需要在高度和宽度上各减少4
    kernel_size = 5
     
    # 对每个通道进行处理
    for c in range(channels):
        # 提取当前通道
        channel_data = img_tensor[c:c+1]  # 保持维度 [1, H, W]
        
        # 使用unfold创建滑动窗口
        # 在高度方向上unfold
        h_unfolded = channel_data.unfold(1, kernel_size, 1)  # [1, H-4, W, 5]
        # 在宽度方向上unfold
        hw_unfolded = h_unfolded.unfold(2, kernel_size, 1)  # [1, H-4, W-4, 5, 5]
        
        # 重塑为 [H-4, W-4, 25] 以便计算中值
        patches = hw_unfolded.reshape(hw_unfolded.shape[1],hw_unfolded.shape[2], kernel_size * kernel_size)
        
        # 计算中值
        medians = torch.median(patches, dim=-1)[0]  # [H-4, W-4]
        # import pdb;pdb.set_trace()
        # 将结果放回输出tensor
        output[c] = medians
    
    return output

def media_conv_alternative(img_tensor):
    """
    中值滤波 - 替代实现（使用更简单的方法）
    如果上面的方法在某些情况下有问题，可以使用这个版本
    """
    if img_tensor.dtype != torch.float32:
        img_tensor = img_tensor.float()
    
    batch_size, channels, height, width = img_tensor.shape
    output = img_tensor.clone()
    kernel_size = 5
    padding = kernel_size // 2
    
    # 使用更直接的滑动窗口方法
    for c in range(channels):
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                # 提取5x5窗口
                window = img_tensor[c, i-padding:i+padding+1, j-padding:j+padding+1]
                # 计算中值
                median_val = torch.median(window.flatten())
                output[c, i, j] = median_val
    
    return output

# 保持原始函数名以兼容现有代码
def media_conv_original(img_tensor):
    """原始实现 - 保持向后兼容"""
    shape = (img_tensor.shape[1]-4,img_tensor.shape[2]-4)
    for c in range(3):
        for i in range(shape[0]):
            for j in range(shape[1]):
                value = median(img_tensor[c,i:i+5,j:j+5].reshape(-1))
                img_tensor[c,i+2,j+2] = value.item()
    return img_tensor
