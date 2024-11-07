from PIL import Image  
import torchvision.transforms as transforms 
import numpy as np
from torch import from_numpy,uint8


def jpg2tensor(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    # 如果图像是 RGB 图像，则形状为 (H, W, C)，需要转换为 (C, H, W) 以匹配 PyTorch 的输入要求  
    if image_np.ndim == 3 and image_np.shape[2] == 3:  
        image_np = image_np.transpose((2, 0, 1))  # 将形状从 (H, W, C) 转换为 (C, H, W) 
    return from_numpy(image_np).to(uint8)

def tensor2pil(img_tensor):
    to_pil_image = transforms.ToPILImage()
    pil_image = to_pil_image(img_tensor)
    return pil_image
