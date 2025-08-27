from torch import log, uint8, clamp, ones_like

def dyrc(img_tensor, c=5.54):
    """
    动态范围压缩函数
    使用对数变换来压缩图像的动态范围
    
    Args:
        img_tensor: 输入图像张量 (C, H, W)
        c: 压缩系数，默认为5.54（log(255)）
    
    Returns:
        压缩后的图像张量
    """
    # 确保输入是float类型
    img_tensor = img_tensor.float()
    
    # 避免log(0)的问题，将0值替换为1
    img_tensor_safe = img_tensor.clone()
    img_tensor_safe[img_tensor_safe == 0] = 1
    
    # 应用对数变换
    log_result = log(img_tensor_safe)
    
    # 应用压缩系数
    compressed = c * log_result
    
    # 将结果截断到[0, 255]范围
    compressed = clamp(compressed, 0, 255)
    
    # 转换为uint8类型
    return compressed.to(uint8)
