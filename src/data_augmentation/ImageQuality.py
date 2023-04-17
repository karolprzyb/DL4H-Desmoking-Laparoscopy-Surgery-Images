import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import csv
import logging

import kornia
from skimage.segmentation import slic

from .VideoLoader import VideoLoader

logger = logging.getLogger(__name__)

class laplacianVar(nn.Module):
    '''
        Used to measure the "sharpness" of an image
    '''
    def __init__(self, kernel_size:int=7, border_type:str='reflect', normalize:bool=False,
                rgb_weights:torch.tensor=torch.tensor([0.299, 0.587, 0.114])) -> None:
        super().__init__()
        self.rgb_to_gray = kornia.color.RgbToGrayscale(rgb_weights)
        self.laplacian_filter = kornia.filters.Laplacian(kernel_size, border_type, normalize)


    def forward(self, x: torch.Tensor):
        """
        Inputs (float with range 0 to 1):
            x (Bx3xWxH) Image batch where B is batch, image is RGB with W width and H height
        Output:
            var (B) variance of laplacian output per image
        """
        x = self.rgb_to_gray(x) # x -> Bx1xWxH
        x = self.laplacian_filter(x).squeeze(1) # x -> BxWxH
        return torch.var(x, dim=(1,2))


class colorMetric(nn.Module):
    '''
        M^3 color metric proposed here: file:///home/florian/Downloads/HaslerS03.pdf
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        """
        Inputs (float with range 0 to 1):
            x (Bx3xWxH) Image batch where B is batch, image is RGB with W width and H height
        Output:
            sigma + 0.3*mean
        """
        rg = x[:, 0] - x[:, 1]
        yb = 0.5*(x[:, 0] + x[:, 1]) - x[:, 2]
        
        sigma = torch.sqrt(torch.var(rg, dim=(-2, -1))  + torch.var(yb, dim=(-2, -1)))
        mean = torch.sqrt(torch.mean(rg, dim=(-2, -1))**2 + torch.mean(yb, dim=(-2, -1))**2)
        return sigma + 0.3*mean


def brigthnessMetric(img:torch.Tensor, n_segments:int = 50, compactness:int = 10, sigma:float = 1.0):
    """
    Inputs (float with range 0 to 1):
        x (3xWxH) Image is RGB with W width and H height
        n_segments, compactness, and sigma are parameters for slic from ski image
    Output:
        mean_v_list: 1D torch tensor of the mean values in each super-pixel selected by slic
        var_v_list: 1D torch tensor of the variance in each super-pixel selected by slic
        segments_slic: 2D torch tensor holding the segmented super-pixels by slic of the input images
    """
    hsv_x = kornia.color.rgb_to_hsv(img)

    x = img.transpose(0, -1)
    segments_slic = slic(x, n_segments=n_segments, compactness=compactness, 
                         sigma=sigma, start_label=1)
    segments_slic = torch.tensor(segments_slic)

    mean_v_list = []
    var_v_list = []
    for segmentation_idx in range(torch.min(segments_slic), torch.max(segments_slic)):
        masked_v = torch.masked_select(hsv_x[-1].transpose(0, -1), segments_slic == segmentation_idx)
        mean_v_list.append(torch.mean(masked_v))
        var_v_list.append(torch.var(masked_v))
        
    return torch.stack(mean_v_list), torch.stack(var_v_list), segments_slic


img_quality_csv_header = ['img_idx', 'Laplacian Variance', 'Color Metric', 'Mean Value List']

def isCSVImgQualityData(csv_file_path:str):
    '''
        Checks if header matches the current img_quality_csv_header:
            img_idx , Laplacian Variance, Color Metric, Mean Value List
    '''
    with open(csv_file_path, mode='r') as csv_file:
        csv_file = csv.reader(csv_file)
        return img_quality_csv_header == next(csv_file)

class ImgQualityData(VideoLoader):
    def __init__(self, csv_file_path:str, clear_csv:bool = False, **kwargs):
        '''
            Class that computes/loads image quality data for a corresponding VideoLoader
            _getitem__ returns (img, list[metrics]) where the img is from VideoLoader
            and metrics are laplacian variance, color metric, and brigthnessMetric
                vid_loader: VideoLoader to have image quality metrics computed for
                csv_file_path: path where the image quality metrics are stored
                    if it exists, then uses the data from the file
                    if it does not exists, then it is automatically generated
                clear_csv: If set to true, will delete the old csv (if it exists)
                    and re-write
                kwargs: keywords passed to initialize VideoLoader parent class (including video_path)
        '''
        # Load video
        super().__init__(**kwargs)

        self.lap = []
        self.col = []
        self.bright = []

        if os.path.exists(csv_file_path) and clear_csv:
            os.remove(csv_file_path)

        # Case where the csv_file_path exists so data is loaded from it
        if os.path.exists(csv_file_path):
            print("CSV IS STILL GOOD SO GOOD")
            if not isCSVImgQualityData(csv_file_path):
                print("CSV is not quality data!?!?")
                logger.warning("Csv file exists but is not img quality data. " + \
                                "Recommend deleting so it is recomputed: {}".format(csv_file_path))
            with open(csv_file_path, mode='r') as img_quality_csv_file:
                print("CSV is good?")
                img_quality_csv_reader = csv.reader(img_quality_csv_file)
                next(img_quality_csv_reader) # skip header
                for row in img_quality_csv_reader:
                    self.lap.append(float(row[1]))
                    self.col.append(float(row[2]))
                    self.bright.append( torch.tensor( [ float(b) for b in row[3:] ] ) )

            if len(self.lap) != super().__len__():
                print("NUM ROWS IN CSV WRONG")
                logger.warning("Number of rows in csv file is {} ".format(len(self.lap)) + \
                                "but video length is {}. ".format(super().__len__()) + \
                                "Recommend deleting csv file so it is recomputed: {}.".format(csv_file_path))
        # Case where csv_file_does not exist so data is computed and saved in csv file
        else:
            print("CSV does not exist?")
            logger.info("Computing image quality metrics for {} ".format(self.video_file_path) + \
                        "and saving too: {}".format(csv_file_path))
            lap_model = laplacianVar()
            color_model = colorMetric()
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(img_quality_csv_header)
                for i in range(super().__len__()):
                    img = super().__getitem__(i)
                    # Compute metrics
                    lap = lap_model(img.unsqueeze(0)).cpu().numpy()[0]
                    col = color_model(img.unsqueeze(0)).cpu().numpy()[0]
                    mean_v_list, _, _ = brigthnessMetric(img)

                    # Store in data loader object
                    self.lap.append(lap)
                    self.col.append(col)
                    self.bright.append(mean_v_list)

                    # Save in csv
                    csv_writer.writerow([str(i), str(lap), str(col)] + [str(v) for v in mean_v_list.cpu().numpy()])

        self.lap = torch.tensor(self.lap)
        self.col = torch.tensor(self.col)

    def __getitem__(self, idx:int):
        return super().__getitem__(idx), self.lap[idx], self.col[idx], self.bright[idx]