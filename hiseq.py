from torch import zeros_like
def histo_equa(img_tensor):
    res = zeros_like(img_tensor)
    for channel in range(3):
        accumulate = 0
        for i in range(256):
            idx = img_tensor[channel]==i
            accumulate += sum(sum(idx))
            res[channel,:,:][idx] = \
                255 * (accumulate/idx.numel())
    return res
