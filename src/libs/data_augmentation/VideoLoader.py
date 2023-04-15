from functools import cache
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor, ConvertImageDtype, Compose

import kornia
import cv2
import numpy as np

import os
import pims
import warnings
import logging
import ffmpeg

from typing import Optional, Type
from types import TracebackType

from .transforms import CropBBox

logger = logging.getLogger(__name__)


def inscribedBBox(center:tuple[float,float], radius:float, img_shape:tuple[int, int]):
    x = int(center[0] - radius/np.sqrt(2))
    y = int(center[1] - radius/np.sqrt(2))
    side = int(2*radius/np.sqrt(2))
    h = side
    w = side

    # Clip bounding box so it is within bounds of the image_shape
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img_shape[-1]:
        w = img_shape[-1] - x
    if y + h > img_shape[-2]:
        h = img_shape[-2] - y
    
    # Keep square dimension
    side = np.min((h, w))
    return x,y,side,side

def getScopeImgCircleParameters(img: torch.Tensor, threshold:float = 0.25):
    '''
    Input: img CxHxW
    Output: (x,y),r, conf location and radius in pixels to define the circlular endoscopic image
        and a confidence value that is computed by seeing the overlap between the enclosing circle and
        thresholded image
    '''
    # Get contours
    img = ConvertImageDtype(torch.float32)(img)
    thresh_img = kornia.utils.tensor_to_image(kornia.color.rgb_to_grayscale(img) > threshold)
    thresh_img = thresh_img.astype(np.uint8)*255
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Remove contours that are too small (less than 0.5% of the image)
    contours = [c for c in contours if cv2.contourArea(c)/(img.shape[-1]*img.shape[-2]) > 0.005]
    contours = np.vstack(contours)

    # Find minimum encolosing circle
    (x,y),r = cv2.minEnclosingCircle(contours)

    # Draw circle and get its overlap with the thresholded image
    circle_img = cv2.circle(np.zeros((img.shape[-2], img.shape[-1])), (int(x), int(y)), int(r), (1), 1)
    conf = np.sum(thresh_img*circle_img)/(np.sum(circle_img)*255)

    return (x,y), r, conf


class VideoLoader(Dataset):
    def __init__(self, video_file_path:str, transform=None, scope_cropping:bool = False,
                cache_directory:str=None, clear_cache_directory:bool = True,
                subsample:int = 1):
        '''
        Class for video loader datasets.
        Inputs:
            video_file_path: path to video file (e.g. mp4)
            transform: transform from torchvision.transforms to apply on the image
            scope_cropping: If true a scope cropping operation BEFORE the transform.
                            This is useful to remove the black parts of the scope
            cache_directory: If none, pims will be used to read the video
                             If set, will store png's generated from the video in this path
                                - This speeds up accessing the image data by a LOT
                             By default, if the cache_directory doesn't exist, FFMPEG will be
                             used to generate the png's in the cache_directory
            clear_cache_directory: If the cache_directory is set and exists, the contents of the
                             directory are cleared and new png's are generated with FFMPEG
            subsample: Directly subsample the video by this amount. This is useful if you want
                          to reduce memory usage from caching the video.

        Additional Attribute:
            target_transform: can be set by inherited classes to apply a torchvision transfrom
                              on the label for the data
        '''

        self.video_file_path = video_file_path
        self.cache_directory = cache_directory
        self.pim_video = None
        self.subsample = subsample

        if self.cache_directory is None:
            self.pim_video = pims.Video(self.video_file_path)
        else:
            logger.info("Preparing cache")

            # Variable that handles if png's should be made via ffmpeg
            make_png = False

            # Create directory and use FMMPEG to generate the pngs
            if not os.path.isdir(cache_directory):
                os.mkdir(self.cache_directory)
                make_png = True

            # Delete all the contents in the cache directory
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
                if len([file for file in os.listdir(self.cache_directory) if file.endswith('.png')]) == 0:
                    logger.warn("Re-using cached png's in {} but didn't find any png's. \
                                 Recommend setting clear_cache_directory = True".format(self.cache_directory))

            # Export video as png's in the cache directory using FFMPEG
            if make_png:
                logger.info("Running ffmpeg to generate png's in {}".format(self.cache_directory))

                # Prepare ffmpeg stream with subsampling
                (
                    ffmpeg
                    .input(self.video_file_path)
                    #.filter('select', 'not(mod(n\,{})'.format(int(self.subsample)))
                    .filter('select',  'not(mod(n,{}))'.format(int(self.subsample)))
                    .filter('setpts', '1/{}*PTS'.format(self.subsample))
                    .output(os.path.join(self.cache_directory, "out-%05d.png"))
                    .run(quiet=False)
                )
                logger.info("Finished generating png's")
            
            # Get list of all the png's
            self.filelist = [file for file in os.listdir(self.cache_directory) if file.endswith('.png')]
            self.filelist.sort()

        self.transform = transform
        self.target_transform = None # Put here for compatibility with other dataloader classes

        if scope_cropping:
            #  -- Loop over images until a bright image is found to compute inscribed bounding box
            thres = 0.15 # Threshold on grayscale pixels used to compute inscribed bounding box
            for img_idx in range(0, self.__len__(), 30):
                img = self._load_image(img_idx)
                #  -- Check if the image is bright enough (25% of the pixels are greater than thres)
                if torch.sum(kornia.color.rgb_to_grayscale(img) > thres) / (img.shape[-2]*img.shape[-1]) < 0.25:
                    continue
                #  -- Compute Scope Parameters and if confidence is greater than 25% keep these parameters
                c, r, conf = getScopeImgCircleParameters(img, threshold = thres)
                if conf > 0.25:
                    break

            #  --Find inscribed bbox
            x,y,w,h = inscribedBBox(c, r*0.9, (img.shape[-2], img.shape[-1]))
            logger.info("Created scope square scrope from detected circle {} with bb {}".format((c,r),(x,y,w,h)))

            #  -- Add CropBBox to transform list
            if self.transform is not None:
                self.transform = Compose([CropBBox(y,x,h,w),
                                                    self.transform])
            else:
                self.transform = CropBBox(y,x,h,w)

    # Helps fix pims issue
    def reloadVideo(self):
        if self.pim_video:
            self.pim_video = pims.Video(self.video_file_path)
        else:
            logger.warn("reloadVideo() was ran but not using pims which requires this function.")

    def __len__(self):
        if self.pim_video:
            # Subtracted 1 (so losing a frame) because the last idx causes error for some reason...
            return int((len(self.pim_video) - 1)/self.subsample)
        else:
            return len(self.filelist)

    def _load_image(self, idx):
        ''' Used internally to manage loading from PIMS or cache'''
        if self.pim_video:
            # To supress the following warning:
            #    MethodDeprecationWarning: VideoStream.seek is deprecated.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    image = ToTensor()(self.pim_video[int(idx * self.subsample)])
                except:
                    self.reloadVideo()
                    image = ToTensor()(self.pim_video[int(idx * self.subsample)])
        else:
            image_file_path = os.path.join(self.cache_directory, self.filelist[idx])
            image = read_image(image_file_path)

        return image

    def __getitem__(self, idx):
        idx = int(idx)

        image = self._load_image(idx)
        if self.transform:
            image = self.transform(image)

        return image

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType] ) -> bool:
        return False
