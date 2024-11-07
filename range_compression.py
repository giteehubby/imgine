from torch import log, uint8

def dyrc(img_tensor,c=30):
    return  c * log(img_tensor).to(uint8)
