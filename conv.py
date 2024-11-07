from torch.nn import Module
from torch import tensor,zeros,mean
from torch import float32 as tfloat
from torch import var as tvar

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

class convolution(Module):
    def __init__(self,kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, picture, num_channel=3):
        shape = self.kernel.shape
        height = picture.shape[1] - shape[0] + 1
        width = picture.shape[2] - shape[1] + 1
        res = zeros((num_channel,height,width))
        for c in range(num_channel):
            for i in range(height):
                for j in range(width):
                    res[c,i,j] = sum(sum(
                        picture[c,i:i+shape[0],j:j+shape[1]] * self.kernel
                    ))
        return res
    
def denoising(img_tensor):
    # conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3)
    # conv.weight = nn.Parameter(0.125 * torch.tensor([[1,1,1],[1,0,1],[1,1,1]]))
    conv = convolution(0.125 * tensor([[1,1,1],[1,0,1],[1,1,1]]))
    conved = conv(img_tensor)

    for c in range(conved.shape[0]):
        for i in range(conved.shape[1]):
            for j in range(conved.shape[2]):
                if abs(conved[c][i][j] - img_tensor[c][i+1,j+1]) > 127.5:
                    img_tensor[c][i+1,j+1] = 225 - img_tensor[c][i+1,j+1]
    return img_tensor

def select_mask_smooth(img_tensor):
    for c in range(3):
        for i in range(img_tensor.shape[1]-4):
            for j in range(img_tensor.shape[2]-4):
                var = 1e8
                v_i = -1
                for i in range(9):
                    v = tvar((seven_mask[i] * img_tensor[c,i:i+5,j:j+5]).reshape(-1).to(tfloat))
                    if v < var:
                        var = v
                        v_i = i
                img_tensor[c][i+2][j+2] = \
                    mean(
                        (seven_mask[v_i] * img_tensor[c,i:i+5,j:j+5]).reshape(-1).to(tfloat)
                    ).item()
    return img_tensor