import torch
import os
from torchvision import transforms
import kornia
from typing import Union

from torch.utils.data import random_split, ConcatDataset, Subset
from data_augmentation import SpeckleImage, UniformlyIncreaseChannel, \
                                RandomHueRGBA, Darken, SynthSmokeLoader, HazeImage_RGBA
from utils import getDatasets

# Settings for generating synthetic  smoke data
intermediate_size = 512
img_size = 256
pre_path = "C:/Users/Karol/Documents/DL4H"

# Choose between via and cholec80
cholec80 = True
if cholec80:
  # -- cholec dataset option
  loader_type = 'cholec_smoke'
  dataset_path = pre_path + '/datasets/cholec80/input_formatted'
  cache = None
  cache_path_synth_smoke = os.path.join(pre_path + '/datasets/cholec80/cache/', "synth_smoke")
  cache_path_real_smoke  = os.path.join(pre_path + '/datasets/cholec80/cache/', "real_smoke")

  l_th = 15
  c_th = 0.15
  b_th = 0.4
  b_th_th = 0.8

else:
  # -- via dataset option
  loader_type = 'via'
  dataset_path = '/home/flodri/Datasets/temporal_data/'
  cache = '/home/flodri/Datasets/cache/smoke_detection/'
  cache_path_synth_smoke = os.path.join(cache, "synth_smoke")
  cache_path_real_smoke  = os.path.join(cache, "real_smoke")

  l_th = 8
  c_th = 0.15
  b_th = 0.4
  b_th_th = 0.55


# Other settings
smoke = pre_path + "/datasets/cholec80/synthetic_smoke/"
via_labels = ['Outside body', 'Smoke', 'Flush Camera']
multi_frame = [1, 2, 3]
subsample = 5

# The many many transforms
vid_transform = transforms.Compose([
                                    transforms.Resize((intermediate_size, intermediate_size)),
                                    transforms.ConvertImageDtype(torch.float32),
                                   ])


synth_smoke_transforms = transforms.Compose([
                                            transforms.ConvertImageDtype(torch.float32),
                                            transforms.CenterCrop((512, 512)),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # HazeImage_RGBA(t=(0.8, 1.0), std_t=(0.0, 0.0), speckle_t=(0.0, 0.0), 
                                            #                a=(0.6, 0.8), std_a=(0.0, 0.0), speckle_a=(0.0, 0.0),
                                            #                keepdim=True, p = 0.5),
                                            # SpeckleImage(std = (0.01, 0.01), speckle = (0.025, 0.025),
                                            #             keepdim=True, p = 0.5),
                                            # UniformlyIncreaseChannel(channel_idx = 2, 
                                            #                          increase_amount = (0.0, 0.15),
                                            #                          keepdim = True, p = 0.5),
                                            # RandomHueRGBA(hue=(-0.15, 0.15)),
                                            # Darken(alpha = (0.95, 1.05), beta = (0.95, 1.05), 
                                            #        gamma = (0.9, 1.1), keepdim=True, p = 0.5)
                                            ])

bg_img_transform = transforms.Compose([
                                       transforms.GaussianBlur(9, sigma=(1.0, 1.0)),
                                       kornia.augmentation.RandomBoxBlur(p=0.5, keepdim=True),
                                       kornia.augmentation.RandomMotionBlur(p=0.5, 
                                                                            kernel_size = (3, 7), 
                                                                            angle = (0, 360), 
                                                                            direction = (-1.0, 1.0),
                                                                            border_type = 'replicate',
                                                                            keepdim=True),
                                       Darken(alpha = (0.9, 1.0), beta = (0.9, 1.0), 
                                              gamma = (1.0, 2.0), keepdim=True, p = 0.5),
                                      ])


prev_frame_transform = transforms.Compose([
                                            transforms.ConvertImageDtype(torch.float32),
                                            kornia.augmentation.RandomPerspective(distortion_scale=0.1,
                                                                                  p = 0.5,
                                                                                  keepdim=True,
                                                                                  sampling_method='area_preserving',
                                                                                  ),
                                           ])

output_transform_synth = transforms.Compose([
                                            transforms.ConvertImageDtype(torch.float32),
                                            transforms.RandomRotation((-5.0, 5.0)),
                                            transforms.RandomResizedCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.Normalize(mean = 0.5, std = 0.5),
                                            ])

output_transform_real = transforms.Compose([
                                            transforms.ConvertImageDtype(torch.float32),
                                            transforms.Resize((img_size, img_size)),
                                            transforms.Normalize(mean = 0.5, std = 0.5),
                                            ])

target_transform = transforms.Compose([
                                        kornia.augmentation.RandomSaturation(saturation=(1.2, 1.2), 
                                                                             p=1.0, keepdim=True),


                                        ])

def atmospheric_scattering_compose(vid_img:torch.tensor, smoke_img:torch.tensor) -> Union[torch.tensor, torch.tensor]:
    '''
        vid_img input need to be of shape 3xWxH with floating point RGB channels
        smoke_img input need to be of shape 4xWxH with floating point RGBA channels
        
        Using alpha compositing for these equations.

        Outputs are of form 3xWxH and WxH
    '''
    
    out_img = vid_img*(1 - 0.5*smoke_img[-1, :, :]) + smoke_img[:-1, :, :]*smoke_img[-1, :, :]*0.5
    out_img[out_img > 1] = 1

    out_mask = smoke_img[-1, :, :]
    return out_img, out_mask

# Get the datasets
dataset_list, real_smoke_dataset_list = getDatasets(vid_dataset_path = dataset_path,
                                                    loader_type = loader_type,
                                                    vid_dataset_cache_path = cache,
                                                    cache_path_synth_smoke = cache_path_synth_smoke,
                                                    cache_path_real_smoke  = cache_path_real_smoke,
                                                    vid_transform = vid_transform,
                                                    vid_dataset_apply_scope_cropping = True,
                                                    smoke_dataset_path = smoke,
                                                    synth_smoke_transforms = synth_smoke_transforms,
                                                    multi_frame = multi_frame,
                                                    bg_img_transform = bg_img_transform,
                                                    output_transform_synth = output_transform_synth,
                                                    output_transform_real = output_transform_real,
                                                    prev_frame_transform = prev_frame_transform,
                                                    vid_dataset_labels = via_labels,
                                                    target_transform = target_transform,
                                                    compose_function = atmospheric_scattering_compose,
                                                    l_th = l_th, c_th = c_th, b_th = b_th, b_th_th = b_th_th)

dataset:SynthSmokeLoader = ConcatDataset(dataset_list)
dataset = Subset( dataset, range(0, len(dataset), subsample) )
real_smoke_dataset:SynthSmokeLoader = ConcatDataset(real_smoke_dataset_list)

# Split train and validation
val_percent = 0.005
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_dataset, val_dataset = random_split(dataset, [n_train, n_val], 
                                            generator=torch.Generator().manual_seed(0))