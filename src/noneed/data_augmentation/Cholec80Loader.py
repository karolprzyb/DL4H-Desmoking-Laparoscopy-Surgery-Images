import torch
import csv

import numpy as np
from .VideoLoader import VideoLoader
import logging

def isCSVCholecSmokeLoaderData(csv_file_path:str):
    '''
        Checks if header matches:
            'frame_start', 'frame_end', 'label'
    '''
    with open(csv_file_path, mode='r') as csv_file:
        csv_file = csv.reader(csv_file)
        return ['frame_start', 'frame_end', 'label'] == next(csv_file)


class Cholec80SmokeLoader(VideoLoader):
    def __init__(self, smoke_csv_file_path:str, video_file_path:str,
                 transform=None, target_transform=None, scope_cropping:bool = False,
                 cache_directory:str=None, clear_cache_directory:bool = True,):
        '''
        Data loader for tool labels in a single video from the cholec80 dataset:
            smoke_csv_file_path: path to smoke labels which should be a csv file
            video_file_path: path to mp4 file which holds the cholec80 video
            cache_directory: If none, pims will be used to read the video
                    If set, will store png's generated from the video in this path
                    - This speeds up accessing the image data by a LOT
                    By default, if the cache_directory doesn't exist, FFMPEG will be
                    used to generate the png's in the cache_directory
            clear_cache_directory: If the cache_directory is set and exists, the contents of the
                directory are cleared and new png's are generated with FFMPEG
            
        Outputs the following labels:
            0 for no smoke
            1 for smoke
            -1 for no label
        '''

        # Load csv data
        self.label_data = None
        self.target_transform = target_transform

        # Prepare video
        super().__init__(video_file_path, transform, scope_cropping, cache_directory, clear_cache_directory)

        # Prepare label data
        self.label_data = np.stack([np.arange(super().__len__()), -1*np.ones((super().__len__()))], axis=1)
        # -- Load csv data and iterate through each label
        with open(smoke_csv_file_path) as csv_file_obj:
            csv_dict_reader = csv.DictReader(csv_file_obj)
            for csv_row in csv_dict_reader:
                frame_start = int(csv_row['frame_start'])
                frame_end = int(csv_row['frame_end'])
                label = int(csv_row['label'])
                self.label_data[frame_start:frame_end, 1] = label
        self.label_data = torch.from_numpy(self.label_data)

    def getLabels(self):
        return self.label_data[:, 1:]

    def __len__(self):
        # Special case for initializing and should be the same as "normal" case
        if self.label_data is None:
            return super().__len__()
        # Normal case
        else:
            return len(self.label_data[:, 0])

    def __getitem__(self, idx):
        image = super().__getitem__(int(self.label_data[idx, 0]))
        label = self.label_data[idx, 1:]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class Cholec80ToolLoader(VideoLoader):
    def __init__(self, tool_txt_file_path:str, video_file_path:str,
                 transform=None, target_transform=None, scope_cropping:bool = False, 
                 cache_directory:str=None, clear_cache_directory:bool = True):
        '''
        Data loader for tool labels in a single video from the cholec80 dataset:
            tool_txt_file_path: path to tool labels which should be a txt file
            video_file_path: path to mp4 file which holds the cholec80 video
            cache_directory: If none, pims will be used to read the video
                    If set, will store png's generated from the video in this path
                    - This speeds up accessing the image data by a LOT
                    By default, if the cache_directory doesn't exist, FFMPEG will be
                    used to generate the png's in the cache_directory
            clear_cache_directory: If the cache_directory is set and exists, the contents of the
                directory are cleared and new png's are generated with FFMPEG
        '''

        # Load text file 
        self.label_data = torch.from_numpy(np.loadtxt(tool_txt_file_path, skiprows=1))
        self.target_transform = target_transform

        # Prepare video
        super().__init__(video_file_path, transform, scope_cropping, cache_directory, clear_cache_directory)
    
    def getLabels(self):
        return self.label_data[:, 1:]

    def __len__(self):
        return len(self.label_data[:, 0])

    def __getitem__(self, idx):
        image = super().__getitem__(int(self.label_data[idx, 0]))
        label = self.label_data[idx, 1:]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label