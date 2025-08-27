from PIL import Image  
import torchvision.transforms as transforms 
import numpy as np
from torch import from_numpy,uint8


def jpg2tensor(image_path):
    image = Image.open(image_path)
    # 确保图像是RGB格式，避免RGBA或其他格式导致的问题
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)
    # 现在图像一定是RGB格式，形状为 (H, W, 3)，需要转换为 (3, H, W) 以匹配 PyTorch 的输入要求  
    image_np = image_np.transpose((2, 0, 1))  # 将形状从 (H, W, 3) 转换为 (3, H, W) 
    return from_numpy(image_np).to(uint8)

def tensor2pil(img_tensor):
    to_pil_image = transforms.ToPILImage()
    pil_image = to_pil_image(img_tensor)
    return pil_image
