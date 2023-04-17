import torch
import torch.nn.functional as F
from torchvision import transforms
import kornia

from math import exp
import numpy as np

from typing import List, Any, Tuple, Union

def center_crop_to_align_images(img1:torch.Tensor, img2:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Crop dim -2
    if img1.shape[-2] > img2.shape[-2]:
        img1 = transforms.CenterCrop((img2.shape[-2], img1.shape[-1]))(img1)
    else:
        img2 = transforms.CenterCrop((img1.shape[-2], img2.shape[-1]))(img2)

    # Crop dim -1
    if img1.shape[-1] > img2.shape[-1]:
        img1 = transforms.CenterCrop((img1.shape[-2], img2.shape[-1]))(img1)
    else:
        img2 = transforms.CenterCrop((img2.shape[-2], img1.shape[-1]))(img2)
    return img1, img2

def gaussian(window_size:int, sigma:float) -> torch.Tensor:
    '''
        Creates a normalized gaussian window of size [window_size x window_size] and 
        with a standard deviation of sigma
    '''
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size:int, channel:int=1, sigma:float=1.5) -> torch.Tensor:
    '''
        Creates guassian windows of size [ channel , 1 , x window_size x window_size ]
        with a standard deviation of sigma. This is used for SSIM and MS-SSIM computations
    '''
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1:torch.Tensor, img2:torch.Tensor, img_range:float, window:Any=None, 
          sigma:float=1.5, window_size:int=11, size_average:bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Computes SSIM metric between img1 and img2
        Inputs:
            img1/2 [B x C x H x W]: are image pairs to compute the SSIM between
            img_range: min - max values img1/2 can take on
            window: optional input of a window from create_window() to reduce computation time
                If no input, function will create internally with window and sigma
            window_size: window size if making window
            sigma: standard deviation if making window
            size_average: if true, averages across all batches in input [ 1 ]
                else, returns ssim per batch element [B x 1]
            
        Outputs:
            ssim: SSIM computation
            cs : contrast sensitivity computation
    '''
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel, sigma=sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.pow(2), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.pow(2), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * img_range) ** 2
    C2 = (0.03 * img_range) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs


def msssim(img1:torch.Tensor, img2:torch.Tensor, img_range:float, window_size:int=11, 
            size_average:bool=True, 
            weights:torch.Tensor = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])) -> torch.Tensor:
    '''
        For M = len(w) (w is weights) levels to compute SSIM, where each level downsamples by a factor of 2, the
        img1/2 ms-ssim can be computed as:
            l_M^{w[M]} * prod_{i=1}^M cs_i^{w[i]}
        where l_M is the luminance of the M level (so most downsampled image)
        cs_i is the constrant*structural of the i-th level for i = 1,...,M

        Inputs:
            img1/2 [B x C x H x W]: are image pairs to compute the SSIM between
            img_range: min - max values img1/2 can take on
            window_size: window size for SSIM computation
            size_average: if true, averages across all batches in input [ 1 ]
                else, returns ssim per batch element [B x 1]
            weights: applied to the MS-SSIM. Default comes from original implementation
                https://www.cns.nyu.edu/pub/eero/wang03b.pdf
            
        Outputs:
            ms-ssim of the images across the paired batch

        Idea for TODO: Approximates ms-ssim using https://arxiv.org/pdf/1511.08861.pdf
        Code is: https://github.com/NVlabs/PL4NN/blob/master/src/loss.py
    '''
    weights = weights.to(img1.device)
    
    levels = weights.size()[0]
    ssims_list:List[torch.Tensor] = []
    for _ in range(levels):
        sim, mcs = ssim(img1, img2, img_range, window_size = window_size, 
                        size_average = size_average)

        ssims_list.append(sim)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims_list)

    output = (mcs ** weights[-1]) * torch.prod(ssims ** weights)
    return output


def psnr(img1:torch.Tensor, img2:torch.Tensor, img_range:float) -> torch.Tensor:
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img_range / torch.sqrt(mse))


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, img_range:float, window_size:int=11):
        '''
            Module for mssim loss.
            Inputs:
                img_range: range of input images (i.e. 2 for [-1, 1])
                window_size for ssim computation
        '''
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.img_range = img_range

    def forward(self, img1:torch.Tensor, img2:torch.Tensor) -> torch.Tensor:
        ms_ssim = msssim(img1, img2, self.img_range, window_size=self.window_size, size_average=True)
        return 1 - ms_ssim


class L1andMSSSIM(torch.nn.Module):
    def __init__(self, img_range:float, alpha:float = 0.84, beta:float = 0.16, window_size:int=11, crop:bool = True):
        '''
        
            alpha*l_mss-sim + beta*l1
        '''
        super(L1andMSSSIM, self).__init__()
        self.msssim = MSSSIMLoss(img_range = img_range, window_size = window_size)
        self.l1 = torch.nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.crop = crop

    def forward(self, img1:torch.Tensor, img2:torch.Tensor) -> torch.Tensor:
        if self.crop:
            img1, img2 = center_crop_to_align_images(img1, img2)
        if self.alpha == 0:
            return self.beta * self.l1(img1, img2)
        else:
            return self.alpha * self.msssim(img1, img2) + self.beta * self.l1(img1, img2)


class L1andMSSSIMandAlignment(torch.nn.Module):
    def __init__(self, img_range:float, alignment_channels:int, alpha:float = 0.0,
                beta:float = 1.0, gamma:float = 1.0, window_size:int=11, crop:bool = True):
        '''
            alpha*l_mss-sim + beta*l1 + gamma*l1_alignment
        '''
        super(L1andMSSSIMandAlignment, self).__init__()
        self.l1_msssim = L1andMSSSIM(img_range = img_range, 
                                    alpha = alpha,
                                    beta = beta,
                                    window_size = window_size, 
                                    crop = crop)
        self.l1 = torch.nn.L1Loss()
        self.c = alignment_channels
        self.gamma = gamma
    
    def forward(self, img1:Tuple[torch.Tensor, torch.Tensor], img2:torch.Tensor) -> torch.Tensor:
        '''
            img1 is Tuple where:
                img1[0] is [B x C x W x H] image to be passed into l_mss-ssim and l1
                img1[1] is [B x N*C x W x H] images to be passed into l1_alignment:
                    1/(N-1) sum_{i=1}^{N-1} l1 ( img1[1][:, i*C:(i+1)*C], img2)
                    where N is number of images

            img2 is [B x C x W x H] image to be passed into l_mss-ssim and l1
        '''
        num_img = int(img1[1].shape[1]/self.c)
        l1_alignment = [self.l1(img1[1][:, i*self.c:self.c*(i+1)], img2) \
                            for i in range(1, num_img)]        

        return self.l1_msssim(img1[0], img2) + self.gamma * torch.mean(torch.stack(l1_alignment))


class PSNR(torch.nn.Module):
    def __init__(self, img_range:float, crop:bool = True):
        '''
            Computes PSNR metric
        '''
        super(PSNR, self).__init__()
        self.img_range = img_range
        self.crop = crop
    
    def forward(self, img1:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], img2:torch.Tensor) -> torch.Tensor:
        if isinstance(img1, tuple):
            img1 = img1[0]
        if self.crop:
            img1, img2 = center_crop_to_align_images(img1, img2)
        return psnr(img1, img2, self.img_range)


class LaplacianVar(torch.nn.Module):
    def __init__(self, kernel_size:int = 5, denormalize_input:bool = True):
        '''
            Computes the laplacian variance of an input image.
            This is a metric used to estimate the bluriness of an image
                kernel_size: size of kernel for laplacian filter
                denormalize_input: denormalizes the images from [-1, 1] -> [0, 1] before computing
        '''
        super(LaplacianVar, self).__init__()
        self.laplacian = kornia.filters.Laplacian(kernel_size, border_type='reflect', normalized=True)
        self.denormalize_input = denormalize_input
    
    def forward(self, img1:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        '''
            In the instance img1 is a single tensor:
                img1 [B x 3 x H x W] or img1 [B x 1 x H x W] where in the 3 channel case the
                order is expceted to be RGB
            In the instance img1 is a tuple of tensors: only the first input tensor is used.
                This is convenient when the output of a network has multiple tensors and
                the others are just used for auxilary losses
        '''
        if isinstance(img1, tuple):
            img1 = img1[0]

        if img1.shape[1] == 3:
            kornia.color.rgb_to_grayscale(img1)
        
        if self.denormalize_input:
            img1 = (img1 + 1.0)/2.0
            
        return torch.var(self.laplacian(img1))