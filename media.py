from torch import median
def media_conv(img_tensor):
    shape = (img_tensor.shape[1]-4,img_tensor.shape[2]-4)
    for c in range(3):
        for i in range(shape[0]):
            for j in range(shape[1]):
                t = img_tensor[c,i:i+5,j:j+5].reshape(-1)
                value = median(img_tensor[c,i:i+5,j:j+5].reshape(-1))
                img_tensor[c,i+2,j+2] = value.item()
    return img_tensor
