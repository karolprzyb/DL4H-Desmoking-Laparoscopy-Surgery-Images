from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from os import listdir
from os.path import splitext
from pathlib import Path
import re


class SegLoader(Dataset):
    def __init__(self, images_dir:str, masks_dir:str, sort_by_num_in_file_name = True, 
                transform=None, target_transform=None):
        '''
            Inputs:
                images_dir: path to folder that contains raw images only
                masks_dir: path to folder that contains masks images only 
                transform/target_transform: are pytorch transforms for data agumentation 
            Note: the file names in images_dir must be the same as masks_dir so the label corresponds to the raw img
            
        '''
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Sorts according to the first number found in filename
        if sort_by_num_in_file_name:
            # Extract list of ints in the file names
            list_num = [int(re.search(r'\d+', id)[0]) for id in self.ids]
            self.ids = [x for _, x in sorted(zip(list_num, self.ids))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        mask = ToTensor()(Image.open(mask_file[0]))
        img = ToTensor()(Image.open(img_file[0]))

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask
