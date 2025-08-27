from torch import uint8

def threshold(img_tensor,thres=127):
    return 255*(img_tensor>thres).to(uint8)

def scared(img_tensor):
    return img_tensor.float()