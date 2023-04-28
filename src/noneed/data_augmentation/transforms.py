import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import kornia
import kornia.color as kc
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.utils import _range_bound

from typing import Tuple, Dict, Any, Optional
import cv2

class BaseTransformClass:
    def __str__(self) -> str:
        out_string = type(self).__name__ + "("
        for idx, (key, value) in enumerate(vars(self).items()):
            out_string = out_string + key + "=" + str(value)
            if idx < len(vars(self)) - 1:
                out_string = out_string + ", "
        out_string = out_string + ")"
        return out_string

class CropBBox(BaseTransformClass):
    """Crop with bounding box"""
    def __init__(self, top, left, height, width):
        self.top  = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)

class RGBA2RGB(BaseTransformClass):
    def __call__(self, x):
        return kc.rgba_to_rgb(x)

class BilateralFilter(BaseTransformClass):
    '''
    Uses OpenCV bilateral filter: 
        https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    '''
    def __init__(self, diameter, sigma_color, sigma_space):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        
    def __call__(self, x):
        '''
            Input is a tensor of shape (C, H, W) or (1, C, H, W)
            Note: Cannot do more than batch size 1 since it is opencv bilateral filter
                  Also this is not-differentiable
        '''
        input_shape = x.shape

        # Case for (B, C, H, W)
        if len(input_shape) == 4:
            # If batch is not of size 1, raise error
            if input_shape[0] != 1:
                raise ValueError("Input shape must be (1, C, H, W) or (C, H, W)")
            # If batch is size 1, remove batch dimension
            else:
                x = x.squeeze(0)
        elif len(input_shape) != 3:
            raise ValueError("Input shape must be (1, C, H, W) or (C, H, W)")

        # Save device to convert back later
        device = x.device

        # x is now (C, H, W), switch to (H, W, C) and convert to numpy
        x = x.permute(1, 2, 0).cpu().detach().numpy()

        # Run bilateral filter
        x = cv2.bilateralFilter(x, self.diameter, self.sigma_color, self.sigma_space)

        # x is now (H, W, C), switch to (C, H, W) and convert to tensor
        x = torch.from_numpy(x).to(device).permute(2, 0, 1)

        return x.reshape(input_shape)

class Darken(IntensityAugmentationBase2D):
    ''' 
        Darkens an image with beta * (alpha * I)^gamma 
        from here: https://arxiv.org/pdf/1908.00682.pdf
    '''
    def __init__(self, 
                alpha: Tuple[float, float]=(0.9, 1.0), 
                beta:  Tuple[float, float]=(0.5, 1.0), 
                gamma: Tuple[float, float]=(1.5, 5),
                same_on_batch: bool = False,
                p: float = 1.0,
                keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((alpha, "alpha_factor", None, None),
                                                         (beta,  "beta_factor",  None, None),
                                                         (gamma, "gamma_factor", None, None)
                                                        )

    def apply_transform(self, x: torch.Tensor, 
                        params: Dict[str, torch.Tensor], 
                        flags: Dict[str, Any], 
                        transform: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        alpha = params["alpha_factor"].to(x)
        beta  = params["beta_factor"].to(x)
        gamma = params["gamma_factor"].to(x)
        output = x
        output[:, :3] = beta * torch.pow( alpha * x[:, :3], gamma)
        return output.reshape(input_shape)

class SpeckleImage(IntensityAugmentationBase2D):
    '''
        Adds speckle to image:
            x + N(0, std^2) + x * N(0, speckle^2)
    '''
    def __init__(self, 
                std: Tuple[float, float] = (0.0, 0.025), 
                speckle: Tuple[float, float] = (0.0, 0.025),
                same_on_batch: bool = False,
                p: float = 1.0,
                keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((std, "std", None, None),
                                                         (speckle,  "speckle",  None, None),
                                                        )

    def apply_transform(self, x: torch.Tensor, 
                        params: Dict[str, torch.Tensor], 
                        flags: Dict[str, Any], 
                        transform: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:

        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        x  =  x + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["std"].to(x)).unsqueeze(0).repeat(1, x.shape[1], 1, 1)\
                + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["speckle"].to(x)).unsqueeze(0).repeat(1, x.shape[1], 1, 1)*x
        x[x > 1] = 1
        x[x < 0] = 0
        return x.reshape(input_shape)

class UniformlyIncreaseChannel(IntensityAugmentationBase2D):
    '''
        Increases the channel_idx values of an image by a uniform amount
        This is useful if input image has no color and all RGB channels are the same value
    '''
    def __init__(self, channel_idx:int, 
                increase_amount:Tuple[float, float] = (0.0, 0.05),
                same_on_batch: bool = False,
                p: float = 1.0,
                keepdim: bool = False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.channel_idx = channel_idx
        self._param_generator = rg.PlainUniformGenerator((increase_amount, "increase_amount", None, None))

    def apply_transform(self, x: torch.Tensor, 
                        params: Dict[str, torch.Tensor], 
                        flags: Dict[str, Any], 
                        transform: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        increase_amount = params["increase_amount"].to(x)
        x[:, self.channel_idx] += increase_amount
        x[x > 1] = 1
        x[x < 0] = 0
        return x.reshape(input_shape)

class RandomHueRGBA(BaseTransformClass):
    '''
        Wrapper for kornia's augmentation.RandomHue for RGBA images
    '''
    def __init__(self, hue = (-0.5, 0.5)):
        self.random_hue = kornia.augmentation.RandomHue(hue = hue, keepdim=True)

    def __call__(self, x):
        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        x[:, :3, :, :] = self.random_hue(x[:, :3, :, :])
        return x.reshape(input_shape)

class HazeImage_RGBA(IntensityAugmentationBase2D):
    '''
        Adjusts an rgba image such that when it alpha composed, 
        it will hazes image  with the following model(https://arxiv.org/pdf/1601.07661.pdf):
            out = in*t + a*(1 - t)
        Input:
            t : low and high values for random transmission to be applied
            std_t: low and high values for gaussian noise on transmission to be applied
            a: low and high values for random light value to be applied
            std_a: low and high values for gaussian noise on light value to be applied
            p: probability to apply the transform
            epsilon: added to the denominator in alpha compose and avoids NaN's
    '''
    def __init__(self, 
                 t:Tuple[float, float]=(0.5, 1.0),
                 std_t:Tuple[float, float]=(0.0,0.05), 
                 speckle_t:Tuple[float, float]=(0, 0.05),
                 a:Tuple[float, float]=(0.5, 1.0),
                 std_a:Tuple[float, float]=(0,0.01),
                 speckle_a:Tuple[float, float]=(0, 0.05),
                 epsilon:float=1e-3,
                 same_on_batch: bool = False,
                 p: float = 1.0,
                 keepdim: bool = False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.epsilon =epsilon
        self._param_generator = rg.PlainUniformGenerator((t, "t", None, None),
                                                         (std_t, "std_t", None, None),
                                                         (speckle_t, "speckle_t", None, None),
                                                         (a, "a", None, None),
                                                         (std_a, "std_a", None, None),
                                                         (speckle_a, "speckle_a", None, None))

    def apply_transform(self, x: torch.Tensor, 
                        params: Dict[str, torch.Tensor], 
                        flags: Dict[str, Any], 
                        transform: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        '''
            x image size 4xWxH where the last channel is alpha
        '''
        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        # Initialize random t and a values
        t = params["t"].to(x)
        a = params["a"].to(x)

        t  =  t + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["std_t"].to(x).to(x.device))\
                + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["speckle_t"].to(x))*t

        a = a + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["std_a"].to(x))\
              + torch.normal(mean=torch.zeros_like(x[:,0,:,:]), std=params["speckle_a"].to(x))*a
        a = a.unsqueeze(0).repeat(1, 3, 1, 1)

        # Alpha compose the smoke over the haze
        alpha_o  = x[:, -1, :, :] + (1 - t)*(1 - x[:, -1, :, :])
        x[:, :-1, :, :] = ( x[:, :-1, :, :] * x[:, -1, :, :] + a * (1 - t)*(1 - x[:, -1, :, :]) ) \
                        / (alpha_o + self.epsilon)
        x[:, -1, :, :] = alpha_o

        # Can get NaN if alpha_o is 0. In this case, the color can be set 
        # arbitrarily to 0 since the corresponding alpha value is 0 (i.e. fully transparent)
        x = torch.nan_to_num(x) 

        # Clip values just in case
        x[x>1] = 1
        x[x<0] = 0
        
        return x.reshape(input_shape)

def warp(x, flo, padding_mode="zeros"):
        """
        Warps an image/tensor (im2) back to im1, according to the optical flow:
        https://arxiv.org/pdf/1612.01925.pdf
        Inputs:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W, device=x.device).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H, device=x.device).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = torch.nn.functional.grid_sample(x, vgrid, padding_mode=padding_mode)
        mask = torch.autograd.Variable(torch.ones(x.size(), device=x.device))
        mask = torch.nn.functional.grid_sample(mask, vgrid, padding_mode=padding_mode)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

class RandomScattering(BaseTransformClass):
    '''
        Randomly moves pixels according to a gaussian map.
        Input:
            sigma: low and high values for standard deviation applied to the gaussian map.
            padding_mode: padding used by warp function which causes the scattering
    '''
    def __init__(self, sigma=(0.0, 3.0), padding_mode="reflection"):
        self.sigma = sigma
        self.padding_mode = padding_mode

    def __call__(self, x):
        '''
            x image size CxHxW or BxCxHxW
        '''
        input_shape = x.shape
        if len(input_shape) == 3:
            x = x.unsqueeze(0)
        elif len(input_shape) != 4:
            raise ValueError("Input must be 3 or 4 dimensions (C H W) or (B C H W)")

        rand_map = torch.normal(mean=torch.zeros((1, 2, x.shape[-2], x.shape[-1]), device=x.device), 
                                std=torch.FloatTensor(1).uniform_(self.sigma[0], self.sigma[1]).to(x.device))
        return warp(x, rand_map, padding_mode=self.padding_mode).reshape(input_shape)