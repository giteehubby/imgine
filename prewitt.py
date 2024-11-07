from torch.nn import Module
from torch import abs,uint8, tensor
from conv import convolution
from hiseq import histo_equa

class prewitter(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = convolution(tensor([[1,0,-1],[1,0,-1],[1,0,-1]])/6)
        self.conv2 = convolution(tensor([[1,1,1],[0,0,0],[-1,-1,-1]])/6)


    def forward(self,img_tensor):
        r1 = self.conv1(img_tensor)
        r2 = self.conv2(img_tensor)
        return (abs(r1) + abs(r2)).to(uint8)


class sobeller(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = convolution(tensor([[0,0,0],[0,0,-1],[0,1,0]])/6)
        self.conv2 = convolution(tensor([[0,0,0],[0,-1,0],[0,0,1]])/6)


    def forward(self,img_tensor):
        r1 = self.conv1(img_tensor)
        r2 = self.conv2(img_tensor)
        return (abs(r1) + abs(r2)).to(uint8)
    
class laplacer(Module):
    def __init__(self):
        super().__init__()
        self.conv = convolution(tensor([[0,1,0],[1,-4,1],[0,1,0]])/8)

    def forward(self,img_tensor):
        r1 = self.conv(img_tensor)
        return (127 + r1).to(uint8)
    
def prewitti(img_tensor):
    pt = prewitter()
    return pt(img_tensor)

def sobel(img_tensor):
    sb = sobeller()
    return histo_equa(sb(img_tensor)) # 再直方图均衡化来增加对比度

def laplacian(img_tensor):
    lp = laplacer()
    return lp(img_tensor)


