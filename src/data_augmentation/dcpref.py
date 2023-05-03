import torch
from torch import nn
import kornia

class DarkChannelPrior(nn.Module):
    def __init__(self, device, kernel_size = 15):
        super(DarkChannelPrior, self).__init__()
        self.device = device
        self.ks = kernel_size
        self.kernel = torch.ones((self.ks,self.ks), device = self.device)

    def dark_channel_preproc(self, im):
        # pytorch function
        # input: im.size = (B,C,H,W) im.dtype = torch.float and im is in [0, 1]
        # output: dark.size = (B,1,H,W)
        b,g,r = torch.split(im, split_size_or_sections = 1, dim = 1)
        dark_pre = torch.min(torch.min(r,g),b)
        return dark_pre

    def guided_filter_plus(self, im, r = 60, eps = 0.0001):
        d_im = kornia.morphology.erosion(im,self.kernel, engine='convolution')

        mean_I = kornia.filters.box_blur(d_im,(r,r))
        mean_p = kornia.filters.box_blur(im,(r,r))
        mean_Ip = kornia.filters.box_blur(d_im*im,(r,r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = kornia.filters.box_blur(d_im*d_im,(r,r))
        var_I   = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = kornia.filters.box_blur(a,(r,r))
        mean_b = kornia.filters.box_blur(b,(r,r))

        q = mean_a*d_im + mean_b

        return q

    def forward(self, I):
        # I.dtype = torch.float, I is in [0., 1.] range

        dark = self.dark_channel_preproc(I)
        dark_ref = self.guided_filter_plus(dark)

        return dark_ref