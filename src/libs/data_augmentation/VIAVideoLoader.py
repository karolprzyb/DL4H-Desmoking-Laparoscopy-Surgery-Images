import torch
import pandas as pd

import warnings
import ast
import os
import csv
import ffmpeg
import logging

from typing import List
from .VideoLoader import VideoLoader


# This class is to load video datasets labelled by VIA:
#   https://www.robots.ox.ac.uk/~vgg/software/via/
#   Works with via-3.0.11

logger = logging.getLogger(__name__)

def isCSVImgViaLoaderData(csv_file_path:str):
    '''
        Checks if header matches the header from VIA 3.0.11:
            # Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)
    '''
    with open(csv_file_path, mode='r') as csv_file:
        csv_file = csv.reader(csv_file)
        return ['# Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)'] \
                == next(csv_file)

class VIAVideoLoader(VideoLoader):
    def __init__(self, via_csv_file_path:str, video_file_path:str, temporal_labels:List[str], 
                 transform=None, target_transform=None, scope_cropping:bool = False, cache_directory:str=None,
                 clear_cache_directory:bool = True):
        '''
        Data loader for VIAVideo temporal labelling (3.0.11). Each element is a tuple where
        the first element is an image and the second element is a tensor whose length
        is equal to the number of classes in temporal_label_dictionary.
        If the image labels are elements in temporal_label_dictionary, then
        the correspondings element (by index) is set to 1, meanwhile others are set to 0.

        Inputs:
            via_csv_file_path (str) is the file path to the csv annotation file from VIA
            video_file_path (str) is the file path that holds the video which were annotated in VIA
            temporal_labels (list of str) keywords are temporal labels used for labelling
                and it's order corresponds to the output label.
                Ex:
                    'Outside body', 'Smoke','Flush Camera'
                Also if csv file has a label not found in list, that label is ignored.
            cache_directory: If none, pims will be used to read the video
                If set, will store png's generated from the video in this path
                - This speeds up accessing the image data by a LOT
                By default, if the cache_directory doesn't exist, FFMPEG will be
                used to generate the png's in the cache_directory
            clear_cache_directory: If the cache_directory is set and exists, the contents of the
                directory are cleared and new png's are generated with FFMPEG
        Notes:
            Only supports 1 video file per csv file
            Only outputs "TEMPORAL-SEGMENTS" for labels. This refers to csv file in via_csv_file_path
        '''
        if not isCSVImgViaLoaderData(via_csv_file_path):
                logger.warning("The csv file is not recognized as \
                                temporal via data from 3.0.11: {}".format(via_csv_file_path))

        # Load VIA csv
        # -- Find lines that start with a #. Want to skip these lines
        skip_lines = []
        with open(via_csv_file_path) as f:
            for l_num, l in enumerate(f):
                if l[0] == '#':
                    if not 'CSV_HEADER' in l:
                        skip_lines.append(l_num)
                        
        # -- Load with skipped lines
        #       Note: cannot use "comment" command from pd.read_csv because the header also uses a comment "#"...
        csv_data = pd.read_csv(via_csv_file_path, skiprows=skip_lines, header = 0)

        # Go through the csv data and confirm the video file in via_csv_file_path matches with video_file_path
        video_file_path_tail = os.path.split(video_file_path)[1]
        for file_list_str in csv_data["file_list"]:
            file = ast.literal_eval(file_list_str)[0]
            if file != video_file_path_tail:
                warnings.warn("video_file_path is set to {} but file in VIA csv ({}) is {}".format(video_file_path, 
                                                                                                   via_csv_file_path,
                                                                                                   file), UserWarning)
                break

        # Load video
        super().__init__(video_file_path, transform, scope_cropping, cache_directory, clear_cache_directory)


        # Load temporal label data
        self.temporal_labels  = temporal_labels        
        self.temporal_label_data = torch.zeros((super().__len__(), len(self.temporal_labels)))
        frame_to_dur_ratio = float(super().__len__()) / float(ffmpeg.probe(video_file_path)['format']['duration'])

        for idx, start_ts in enumerate(csv_data["temporal_segment_start"]):
            
            label_data = ast.literal_eval(csv_data["metadata"][idx])["TEMPORAL-SEGMENTS"]

            if label_data in self.temporal_labels:
                start_idx = int(start_ts*frame_to_dur_ratio)
                end_idx   = int(csv_data["temporal_segment_end"][idx]*frame_to_dur_ratio)
                self.temporal_label_data[start_idx:end_idx, self.temporal_labels.index(label_data)] = 1
            else:
                logger.info("Found label {} in csv file {}, but not in temporal_labels".format(label_data, 
                                                                                               via_csv_file_path))

        # Transform information
        self.target_transform = target_transform

    def getLabels(self):
        return self.temporal_label_data

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        label = self.temporal_label_data[idx, :]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label