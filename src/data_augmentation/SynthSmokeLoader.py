import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, write_png

import numpy as np
import os
import logging
from typing import Union, Callable

from .VideoLoader import VideoLoader
from .SmokeDirectoryLoader import SmokeDirectoryLoader
from .utils import *

logger = logging.getLogger(__name__)


def compose_smoke_and_vid_img(vid_img:torch.tensor, smoke_img:torch.tensor) -> Union[torch.tensor, torch.tensor]:
    '''
        Both inputs need to be of shape 3xWxH with floating point RGB channels
        Coppied from equations 3 and 4: https://core.ac.uk/download/pdf/266980034.pdf

        Outputs are of form 3xWxH and WxH
    '''
    out_img = 0.5*vid_img + 0.5*smoke_img
    out_img[out_img > 1] = 1

    out_mask = 0.3*smoke_img[0, :, :] + 0.59*smoke_img[1, :, :] + 0.11*smoke_img[2, :, :]
    return out_img, out_mask

def compose_smoke_and_vid_img_alpha(vid_img:torch.tensor, smoke_img:torch.tensor) -> Union[torch.tensor, torch.tensor]:
    '''
        vid_img input need to be of shape 3xWxH with floating point RGB channels
        smoke_img input need to be of shape 4xWxH with floating point RGBA channels
        
        Using alpha compositing for these equations.

        Outputs are of form 3xWxH and WxH
    '''
    out_img = vid_img*(1 - smoke_img[-1, :, :]) + smoke_img[:-1, :, :]*smoke_img[-1, :, :]
    out_img[out_img > 1] = 1

    out_mask = smoke_img[-1, :, :]
    return out_img, out_mask


class SynthSmokeLoader(Dataset):
    '''
        Randomly matches each image between the from background_image_dataset to
        images in the synth_smoke_dataset. Therefore: 
            len(SynthSmokeLoader) = len(background_image_dataset)
        Note: In order for idx_of_prev_frame (i.e. multiframes) to have temporal consistency, ensure
            background_image_dataset has is temporally consistent.
        Note: If using the cache, the data is saved as PNG's which means:
            1. The saving adds uint8 quantization
            2. Add transforms.ConvertImageDtype(torch.float32) to output_transform for float output
        
        Inputs:
            background_image_dataset: Dataset that loads background images
            synth_smoke_dataset: Dataset that loads smoke images. Special use by setting to None.
                In that case, the __getitem__ tuple return (img, smoke_mask, bg_img) will have:
                    - img set to just the data from the background image dataset
                        so no alpha composing, but still apply appropriate transforms
                    - smoke_mask is set to 0 image with 1 channel and same resolution as img
                    - bg_img is set to img
                This is useful when using real smoke images in background_image_dataset
                but still wanting to use SynthSmokeLoader class for consistency
            compose_function: Function that compses an image from background_image_dataset 
                and an image from synth_smoke_dataset.
            bg_img_transform: PyTorch transform which are applied to the bg_img before composing
                but will NOT appear in the bg_img returned in __getitem__
                This is useful for applying additional models (e.g. HazeImage) that should be learned
            output_transform: Pytorch transform which is applied to ALL returned items in __getitem__
                and NOT saved in the cached data. This is useful for applying random 
                rotations/flips to do image normalization (i.e. [0,1] -> [-1, 1])
            idx_of_prev_frame: contains bg_img indices relative to the bg_img pointed that should be appended
                when accessing this data. For example, if idx_of_prev_frame=[15, 30] then
                the the following will be returned from the dataset when asking for idx:
                [dataset[idx], dataset[idx-15],  dataset[idx-30]] -> (3C x W x H)
                Note that the images are simply stacked on the color channels
            prev_frame_transform: transform applied to the prev_frame (from idx_of_prev_frame).
                This transform is applied AFTER bg_img_transform.
                NOTE: Since this transform is not applied to all bg images (i.e. not at idx),
                    its effects cannot be cached. Therefore, keep this operation as cheap as possible...
            cache_directory: Directory to cache the data. The data will be saved as uint8 png
                including raw background image, synthetically composed images, and smoke masks
            clear_cache_directory: Clears the cache saved in the directy and will re-write it
            to_float_when_loading_cache: if true, converts images loaded from cache to torch.float32
                if false, the images will be loaded from cache as uint8 since images
    '''
    def __init__(self, background_image_dataset:VideoLoader, synth_smoke_dataset:SmokeDirectoryLoader,
                 compose_function:Callable = compose_smoke_and_vid_img_alpha, 
                 bg_img_transform = None, output_transform = None, target_transform = None, 
                 idx_of_prev_frame:list[int] = [],
                 prev_frame_transform = None, cache_directory:str=None, clear_cache_directory:bool = True,
                 to_float_when_loading_cache:bool = True):
        self.background_image_dataset = background_image_dataset
        self.synth_smoke_dataset = synth_smoke_dataset
        self.compose_function = compose_function
        self.bg_img_transform = bg_img_transform
        self.output_transform = output_transform
        self.target_transform = target_transform
        self.idx_of_prev_frame = idx_of_prev_frame
        self.prev_frame_transform = prev_frame_transform
        self.cache_directory = cache_directory
        self.to_float_when_loading_cache = to_float_when_loading_cache

        if len(self.idx_of_prev_frame):
            self.idx_offset_for_dataset = max(self.idx_of_prev_frame)
        else:
            self.idx_offset_for_dataset = 0

        if self.idx_offset_for_dataset > len(background_image_dataset):
            raise ValueError("Length VIA dataset is {} by \
                                multi-frame requires at least {}".format(len(background_image_dataset), 
                                                                         self.idx_offset_for_dataset))

        self.association = np.random.randint(low=0, high=len(self.synth_smoke_dataset) if self.synth_smoke_dataset else 1, 
                                             size=len(self.background_image_dataset) )

        # Option to create cache!
        self.initialize_complete = False
        if not self.cache_directory is None:
            logger.info("Preparing cache")

            # Variable that handles if png's should be made into cache
            make_png = False

            # Create directory if needed
            if not os.path.isdir(self.cache_directory):
                os.mkdir(self.cache_directory)
                make_png = True

            # Delete all the contents in the cache directory if clear_cache_directory
            elif os.path.isdir(self.cache_directory) and clear_cache_directory:
                logger.info("Clearing all png's in cache_folder {}".format(self.cache_directory))
                for file in os.listdir(self.cache_directory):
                    if file.endswith('.png'):
                        os.remove(os.path.join(self.cache_directory, file)) 
                
                make_png = True

            # Case occurs if the self.chache_directory exists and clear_cache_directory = False
            # Want to just double check some sort of png's exist and raise warning if not
            else:
                logger.info("Re-using cache at {}".format(self.cache_directory))
                num_of_pngs = len([file for file in os.listdir(self.cache_directory) if file.endswith('.png')])
                if num_of_pngs != len(self.background_image_dataset) and \
                    num_of_pngs != 3*len(self.background_image_dataset):
                    logger.warn("Re-using cached png's in {}, ".format(self.cache_directory) + \
                                "but only found {} images ".format(num_of_pngs) + \
                                "and len(bg_ds) = {}. ".format(len(self.background_image_dataset)) + \
                                "Recommend setting clear_cache_directory = True")

            # Export video as png's in the cache directory
            if make_png:
                logger.info("Generating png's in {}".format(self.cache_directory))
                for idx in range(len(self.background_image_dataset)):
                    # Load images
                    img, smoke_mask, bg_img = self._load_and_compose(idx)
                    # Save 
                    tf = transforms.ConvertImageDtype(torch.uint8)
                    write_png(tf(img), os.path.join(self.cache_directory, "img_%05d.png" % idx))
                    if self.synth_smoke_dataset is not None:
                        write_png(tf(smoke_mask), os.path.join(self.cache_directory, "smoke_mask_%05d.png" % idx))
                        write_png(tf(bg_img), os.path.join(self.cache_directory, "bg_img_%05d.png" % idx))
                logger.info("Finished generating png's")
            
            # Get list of all the png's
            self.filelist = [file for file in os.listdir(self.cache_directory) if file.endswith('.png')]
            self.filelist.sort()
            self.initialize_complete = True

    def __len__(self):
        return (len(self.association) - self.idx_offset_for_dataset)

    def _load_image_pair(self, idx:int) -> Union[torch.tensor, torch.tensor]:
        ''' Internal function used to load the bg_img and smoke_img'''
        bg_img = self.background_image_dataset[idx]
        if isinstance(bg_img, tuple):
            bg_img = bg_img[0]

        if self.synth_smoke_dataset is not None:
            smoke_img = self.synth_smoke_dataset[self.association[idx]]
        else:
            smoke_img = torch.zeros_like(bg_img)

        return bg_img, smoke_img

    def _compose_function(self, vid_img:torch.tensor, smoke_img:torch.tensor) -> Union[torch.tensor, torch.tensor]:
        ''' Internal function that wraps self.compose_function'''
        if self.synth_smoke_dataset is not None:
            return self.compose_function(vid_img, smoke_img)
        else:
            return vid_img, smoke_img
    
    def _load_and_compose(self, idx:int) -> Union[torch.tensor, torch.tensor, torch.tensor]:
        # Cache case
        if not self.cache_directory is None and self.initialize_complete:
            img = read_image(os.path.join(self.cache_directory, "img_%05d.png" % idx))

            if self.synth_smoke_dataset is not None:
                smoke_mask = read_image(os.path.join(self.cache_directory, "smoke_mask_%05d.png" % idx))
                bg_img = read_image(os.path.join(self.cache_directory, "bg_img_%05d.png" % idx))
            else:
                smoke_mask = torch.zeros_like(img)
                bg_img = torch.zeros_like(img)

            if self.to_float_when_loading_cache:
                tf = transforms.ConvertImageDtype(torch.float32)
                img = tf(img)
                smoke_mask = tf(smoke_mask)
                bg_img = tf(bg_img)

            return img, smoke_mask, bg_img

        # None cache case
        bg_img, smoke_img = self._load_image_pair(idx)

        # Apply bg_img_transform
        if self.bg_img_transform is not None:
            t_bg_img = self.bg_img_transform(bg_img)
        else:
            t_bg_img = bg_img

        # Compose images
        img, smoke_mask = self._compose_function(t_bg_img, smoke_img)
        img = ensureChannelAvailableInImgTensor(img)
        smoke_mask = ensureChannelAvailableInImgTensor(smoke_mask)

        # Apply target_transform
        if self.target_transform is not None:
            bg_img = self.target_transform(bg_img)

        return img, smoke_mask, bg_img
            
    def __getitem__(self, idx:int) -> torch.tensor:
        # Get images from both datasets
        img, smoke_mask, bg_img = self._load_and_compose(idx)

        # Repeat loading trailing images (with some smoke composition)
        for rel_idx in self.idx_of_prev_frame:
            e_img, e_smoke_mask, _ = self._load_and_compose(idx + self.idx_offset_for_dataset - rel_idx)
            if self.prev_frame_transform is not None:
                e_img = self.prev_frame_transform(e_img)
            img = torch.cat([img, e_img])
            smoke_mask = torch.cat([smoke_mask, e_smoke_mask])

        # Output transform applied in single pass so all tensors are changed together
        if self.output_transform:

            c_img = img.shape[0]
            c_smoke_mask = smoke_mask.shape[0]
            # c_bg_img = bg_img.shape[0] # never used but left for completeness

            temp_stack = torch.cat([img, smoke_mask, bg_img], dim=0)
            temp_stack = self.output_transform(temp_stack)

            img = ensureChannelAvailableInImgTensor(temp_stack[:c_img, :, :])
            smoke_mask = ensureChannelAvailableInImgTensor(temp_stack[c_img:(c_img + c_smoke_mask), :, :])
            bg_img = ensureChannelAvailableInImgTensor(temp_stack[(c_img + c_smoke_mask):, :, :])

        return img, smoke_mask, bg_img