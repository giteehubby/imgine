def ladder(img_tensor, num_step=9):
    """
    阶梯量化函数
    
    Args:
        img_tensor: 输入图像张量 (C, H, W)
        num_step: 量化步长，默认为9
    
    Returns:
        量化后的图像张量
    """
    # 确保输入是float类型以进行精确计算
    img_tensor = img_tensor.float()
    
    # 执行阶梯量化
    quantized = (img_tensor // num_step) * num_step
    
    return quantized