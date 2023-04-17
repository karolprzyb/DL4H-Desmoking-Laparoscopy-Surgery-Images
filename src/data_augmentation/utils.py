import torch
from torch.utils.data import Dataset
import kornia
import cv2
import numpy as np

import os
import logging
from typing import Any, Literal, Union
from torchvision import transforms

from .transforms import CropBBox
from .VideoLoader import VideoLoader
from .VIAVideoLoader import VIAVideoLoader, isCSVImgViaLoaderData
from .ImageQuality import ImgQualityData, isCSVImgQualityData
from .Cholec80Loader import Cholec80SmokeLoader, isCSVCholecSmokeLoaderData

logger = logging.getLogger(__name__)

def getVidData(vid_dataset_path:str,
               loader_type:Literal["vid", "via", "qual", "cholec_smoke"],
               vid_dataset_cache_path:str = None,
               **kwargs) -> list[Union[VideoLoader,VIAVideoLoader,ImgQualityData, Cholec80SmokeLoader]] :
    '''
    Returns list of Video Dataset Loaders for each folder in vid_dataset_path that contains a .mp4 file
    Inputs:
        vid_dataset_path: str path which points to the top folder holding VIA-temporally labelled datasets
            Each sub-folder contains a single mp4 and csv file for the video and labels respectively.
            See VIAVideoLoader in data_augmentation for more details.
        loader_type: type for video loader being used
        vid_dataset_cache_path: top folder to cache the mp4 --> .png files for video loader
        kwargs: keywords passed to the video loader class for initialization
    '''
    output:list[Union[VideoLoader,VIAVideoLoader,ImgQualityData, Cholec80SmokeLoader]] = []
    # Loop through folders containing VIA-temporally labelled datasets
    for idx, directory in enumerate(os.listdir(vid_dataset_path)):

        directory_path = os.path.join(vid_dataset_path, directory)
        if os.path.isdir(directory_path):
            # Find mp4 file
            mp4_files = [f for f in os.listdir(directory_path) if f.endswith('MP4') or f.endswith('mp4')]
            if len(mp4_files) != 1:
                logger.warning("Found {} *.mp4 files in directory {}. \
                    There should only be one VIA labelling mp4 file \
                    in the dataset sub-directory. Skipping folder!".format(len(mp4_files), directory_path))
                continue
            
            # Prepare cache directory 
            if vid_dataset_cache_path is not None:
                cache_directory = os.path.join(vid_dataset_cache_path, directory)
            else:
                cache_directory = None

            # via loader cases
            if loader_type == "via":
                # Find the VIA csv file
                csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('csv')]
                via_csv_files = [f for f in csv_files if isCSVImgViaLoaderData(f)]
                if len(via_csv_files) != 1:
                    logger.warning("Found {} *.csv files in directory {}. \
                        There should only be one csv labelling CSV file \
                        in the dataset sub-directory. Skipping folder!".format(len(via_csv_files), directory_path))
                    continue
                
                # Load via video dataset
                t_vid_data = VIAVideoLoader(via_csv_files[0], 
                                            video_file_path=os.path.join(directory_path, mp4_files[0]), 
                                            cache_directory=cache_directory,
                                            **kwargs)
            
            # img quality loader case
            elif loader_type == "qual":
                # Find the Image Quality csv file
                # If not found, will create one called "quality_metrics.csv" in the directory_path
                csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('csv')]
                img_quality_csv_files = [f for f in csv_files if isCSVImgQualityData(f)]
                if len(img_quality_csv_files) == 1:
                    img_quality_csv = img_quality_csv_files[0]
                elif len(img_quality_csv_files) == 0:
                    img_quality_csv = os.path.join(directory_path, 'quality_metrics.csv')
                else:
                    print("CSV FOUND SO THAT IS GOOD")
                    logger.warning("Found {} *.csv files in directory {}. \
                        There should only be one or zero image quality CSV files \
                        in the dataset sub-directory. Skipping folder!".format(len(img_quality_csv_files), directory_path))
                    continue

                # Load image quality video dataset
                t_vid_data = ImgQualityData(img_quality_csv, 
                                            video_file_path=os.path.join(directory_path, mp4_files[0]), 
                                            cache_directory=cache_directory, **kwargs)
            
            # cholec80 smoke data set
            elif loader_type == 'cholec_smoke':
                # Find the cholec80 smoke csv file
                csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('csv')]
                smoke_csv_files = [f for f in csv_files if isCSVCholecSmokeLoaderData(f)]
                if len(smoke_csv_files) != 1:
                    logger.warning("Found {} *.csv files in directory {}. \
                        There should only be one smoke CSV file \
                        in the dataset sub-directory. Skipping folder!".format(len(smoke_csv_files), directory_path))
                    continue
                
                # Load cholec 80 smoke video dataset
                t_vid_data = Cholec80SmokeLoader(smoke_csv_files[0], 
                                                 video_file_path=os.path.join(directory_path, mp4_files[0]), 
                                                 cache_directory=cache_directory, 
                                                 **kwargs)

            # video only loader case
            else:
                # Load video dataset
                t_vid_data = VideoLoader(os.path.join(directory_path, mp4_files[0]), 
                                         cache_directory=cache_directory,
                                         **kwargs)

            output.append(t_vid_data)
        
    return output

def ensureChannelAvailableInImgTensor(img:torch.tensor) -> torch.tensor:
    '''
        Input: must be of shape CxWxH or WxH
        Output: If input is of shape CxWxH, returns input
                If input is of shape WxH, returns input unsqueezed (1xWxH)
    '''
    if len(img.shape) == 3:
        return img
    elif len(img.shape) == 2:
        return img.unsqueeze(0)
    else:
        raise TypeError("Input tensor must be of shape CxWxH or WxH")