from torch.utils.data import Dataset
import torchvision
import torch
import os
import logging

logger = logging.getLogger(__name__)

class SmokeDirectoryLoader(Dataset):
    def __init__(self, directory:str, transform=None):
        '''Dataset
            Loads a single folder of smoke images generated by SyntheticDataGeneration:
            https://github.com/FloDri-SI/SyntheticDataGeneration/blob/main/examples/smoke_dataset.py

            Assumes the directory has png's ordered by filename.
        '''
        self.directory = directory
        self.transform = transform
        self.filelist = [file for file in os.listdir(self.directory) if file.endswith('.png')]
        self.filelist.sort()

        if not self.filelist:
            logger.warning("Found no *.png files in directory {}".format(self.directory))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx:int) -> torch.tensor:
        '''
            Expected default tensor size is 4,W,H where there is an alpha channel.
            Can add transform to change to 3 channel (e.g. RGB)
        '''
        image_file_path = os.path.join(self.directory, self.filelist[idx])
        image = torchvision.io.read_image(image_file_path)
        
        if self.transform:
            image = self.transform(image)


        return image
