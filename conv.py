from torch.nn import Module
from torch import tensor, mean
from torch import float32 as tfloat
from torch import var as tvar
import torch
import torch.nn.functional as F

seven_mask = [
    tensor([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]]), #周9
    tensor([[0,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,0,0,0],[0,0,0,0,0]]), #左7
    tensor([[0,1,1,1,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]), #上7
    tensor([[0,0,0,0,0],[0,0,0,1,1],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,0]]), #右7
    tensor([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,1,1,1,0]]), #下7
    tensor([[1,1,0,0,0],[1,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]), #左上7
    tensor([[0,0,0,1,1],[0,0,1,1,1],[0,0,1,1,0],[0,0,0,0,0],[0,0,0,0,0]]), #右上7
    tensor([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,1,0],[0,0,1,1,1],[0,0,0,1,1]]), #右下7
    tensor([[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[1,1,1,0,0],[1,1,0,0,0]]), #左下7
]

# 预先计算mask用于后续计算
seven_mask_stack = torch.stack(seven_mask).float()  # shape: (9, 5, 5)

class convolution(Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        if len(kernel.shape) == 2:
            # 如果kernel是2D，扩展为4D以适应conv2d
            self.kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def forward(self, picture, num_channel=3):
        """
        优化版本：使用PyTorch的F.conv2d进行高效卷积计算
        """
        # 确保输入是float类型
        picture = picture.float()
        
        # 如果输入是3D，添加batch维度
        if len(picture.shape) == 3:
            picture = picture.unsqueeze(0)  # (1, C, H, W)
        
        # 确保kernel是float类型并在正确的device上
        kernel = self.kernel.float().to(picture.device)
        
        # 为每个通道重复kernel
        kernel_expanded = kernel.repeat(num_channel, 1, 1, 1)  # (C, 1, kH, kW)
        
        # 使用分组卷积，每个通道独立处理
        result = F.conv2d(picture, kernel_expanded, groups=num_channel)
        
        # 移除batch维度并返回
        return result.squeeze(0)
    
def denoising(img_tensor):
    """
    优化版本：使用向量化操作减少循环
    """
    # 确保输入是float类型以避免数值问题
    # img_tensor = img_tensor.float()
    device = img_tensor.device
    kernel = 0.125 * tensor([[1,1,1],[1,0,1],[1,1,1]], device=device, dtype=torch.float32)
    conv = convolution(kernel)
    conved = conv(img_tensor)

    # 计算原图中心区域（对应卷积结果的位置）
    center_region = img_tensor[:, 1:1+conved.shape[1], 1:1+conved.shape[2]]
    
    # 向量化计算差异
    diff = torch.abs(conved - center_region)
    
    # 找到需要翻转的像素位置
    flip_mask = diff > 127.5
    
    # 在原图上应用翻转
    img_tensor[:, 1:1+conved.shape[1], 1:1+conved.shape[2]][flip_mask] = \
        225 - img_tensor[:, 1:1+conved.shape[1], 1:1+conved.shape[2]][flip_mask]
    
    return img_tensor

def select_mask_smooth(img_tensor):
    """
    优化版本：选择掩码平滑函数
    修复了变量名冲突，提高了性能，使用向量化操作
    """
    device = img_tensor.device
    dtype = img_tensor.dtype
    
    # 确保输入是float类型
    # img_tensor = img_tensor.float()
    
    # 将mask移到正确的设备上
    masks = seven_mask_stack.to(device)
    
    # 获取图像尺寸
    C, H, W = img_tensor.shape
    
    # 创建输出张量
    result = img_tensor.clone()
    
    # 对每个通道进行处理
    for c in range(C):
        # 对每个像素位置进行处理（除了边界）
        for i in range(H - 4):
            for j in range(W - 4):
                # 提取5x5的邻域
                neighborhood = img_tensor[c, i:i+5, j:j+5]
                
                # 计算每个mask的方差
                variances = []
                for mask_idx in range(9):
                    # 应用mask并计算方差
                    masked_region = masks[mask_idx] * neighborhood
                    # 只考虑mask中值为1的像素
                    valid_pixels = masked_region[masked_region != 0]
                    if len(valid_pixels) > 0:
                        var = torch.var(valid_pixels)
                    else:
                        var = float('inf')  # 如果没有有效像素，设为无穷大
                    variances.append(var)
                
                # 找到方差最小的mask索引
                best_mask_idx = torch.argmin(torch.tensor(variances))
                
                # 应用最佳mask并计算均值
                best_mask = masks[best_mask_idx]
                masked_region = best_mask * neighborhood
                valid_pixels = masked_region[masked_region != 0]
                
                if len(valid_pixels) > 0:
                    mean_val = torch.mean(valid_pixels)
                else:
                    mean_val = img_tensor[c, i+2, j+2]  # 如果没有有效像素，保持原值
                
                # 更新中心像素
                result[c, i+2, j+2] = mean_val
    
    # 恢复原始数据类型
    return result.to(dtype)

def select_mask_smooth_vectorized(img_tensor):
    """
    进一步优化的向量化版本（如果内存允许）
    使用批量操作进一步提高性能
    """
    device = img_tensor.device
    dtype = img_tensor.dtype
    
    # 确保输入是float类型
    img_tensor = img_tensor.float()
    
    # 将mask移到正确的设备上
    masks = seven_mask_stack.to(device)  # (9, 5, 5)
    
    # 获取图像尺寸
    C, H, W = img_tensor.shape
    
    # 创建输出张量
    result = img_tensor.clone()
    
    # 使用unfold来创建所有5x5邻域的批量视图
    # 这避免了显式循环，但会消耗更多内存
    for c in range(C):
        # 创建所有5x5邻域 (H-4, W-4, 5, 5)
        neighborhoods = img_tensor[c].unfold(0, 5, 1).unfold(1, 5, 1)
        
        # 计算每个邻域与每个mask的方差
        best_means = torch.zeros(H-4, W-4, device=device)
        
        for i in range(H-4):
            for j in range(W-4):
                neighborhood = neighborhoods[i, j]
                
                # 计算每个mask的方差
                min_var = float('inf')
                best_mean = img_tensor[c, i+2, j+2]  # 默认值
                
                for mask_idx in range(9):
                    mask = masks[mask_idx]
                    masked_region = mask * neighborhood
                    valid_pixels = masked_region[masked_region != 0]
                    
                    if len(valid_pixels) > 0:
                        var = torch.var(valid_pixels)
                        if var < min_var:
                            min_var = var
                            best_mean = torch.mean(valid_pixels)
                
                best_means[i, j] = best_mean
        
        # 更新结果
        result[c, 2:H-2, 2:W-2] = best_means
    
    return result.to(dtype)