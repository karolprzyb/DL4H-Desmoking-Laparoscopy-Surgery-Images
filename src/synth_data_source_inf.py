import torch
import os
from torchvision import transforms
import kornia

from torch.utils.data import random_split, ConcatDataset, Subset
from data_augmentation import SpeckleImage, RandomHueRGBA, Darken, \
                                SynthSmokeLoader, HazeImage_RGBA
from utils import getDatasets

def loadData(args):
       # Settings for generating synthetic  smoke data
       intermediate_size = 512
       img_size = 256

       # Cholec dataset location
       loader_type = 'cholec_smoke'

       # pre_path = "C:/Users/Karol/Documents/DL4H"
       # cache_subfolder = '/datasets/cholec80/cache'
       # cache_subfolder_test = '/datasets/cholec80/cache_testset'
       # syn_smoke_subfolder = '/datasets/cholec80/synthetic_smoke/'
       # dataset_subfolder = '/datasets/cholec80/input_formatted'
       # dataset_subfolder_test = '/datasets/cholec80/input_formatted_test'

       pre_path = args['pre_path']
       cache_subfolder = args['cache_subfolder']
       cache_subfolder_test = args['cache_subfolder_test']
       syn_smoke_subfolder = args['syn_smoke_subfolder']
       dataset_subfolder = args['dataset_subfolder']
       dataset_subfolder_test = args['dataset_subfolder_test']

       dataset_path = pre_path + dataset_subfolder
       dataset_path_test = pre_path + dataset_subfolder_test
       cache = pre_path + cache_subfolder
       cache_test = pre_path + cache_subfolder_test
       cache_path_synth_smoke = os.path.join(cache, "new_synth_smoke")
       cache_path_real_smoke  = os.path.join(cache, "real_smoke")
       cache_path_synth_smoke_test = os.path.join(cache_test, "new_synth_smoke")
       cache_path_real_smoke_test  = os.path.join(cache_test, "real_smoke")

       smoke = pre_path + syn_smoke_subfolder


       l_th = 15
       c_th = 0.15
       b_th = 0.4
       b_th_th = 0.8


       # Other settings
       via_labels = ['Outside body', 'Smoke', 'Flush Camera']
       multi_frame = []
       subsample = 5
       subsample_test = 5



       # Prepare all the transforms and synthetic smoke loaders
       vid_transform = transforms.Compose([
                                          transforms.Resize((intermediate_size, intermediate_size)),
                                          transforms.ConvertImageDtype(torch.float32),
                                          ])


       synth_smoke_transforms = transforms.Compose([
                                                 transforms.ConvertImageDtype(torch.float32),
                                                 transforms.CenterCrop((intermediate_size, intermediate_size)),
                                                 HazeImage_RGBA(t=(0.8, 1.0), std_t=(0.0, 0.0), speckle_t=(0.0, 0.0), # Haze only works on rgba and must be applied to the alpha composited layer.
                                                               a=(0.6, 0.8), std_a=(0.0, 0.0), speckle_a=(0.0, 0.0),
                                                               keepdim=True, p = 0.5),
                                                 SpeckleImage(std = (0.00, 0.01), speckle = (0.0, 0.025),
                                                               keepdim=True, p = 0.5),
                                                 RandomHueRGBA(hue=(-0.15, 0.15)),
                                                 Darken(alpha = (0.95, 1.05), beta = (0.95, 1.05), 
                                                        gamma = (0.9, 1.1), keepdim=True, p = 0.5),
                                                 ])


       bg_img_transform = transforms.Compose([
                                          transforms.GaussianBlur(9, sigma=(0.01, 2.0)),
                                          kornia.augmentation.RandomBoxBlur(p=0.5, keepdim=True),
                                          kornia.augmentation.RandomMotionBlur(p=0.5, 
                                                                             kernel_size = (3, 13), 
                                                                             angle = (0, 360), 
                                                                             direction = (-1.0, 1.0),
                                                                             border_type = 'replicate',
                                                                             keepdim=True),
                                          Darken(alpha = (0.9, 1.0), beta = (0.9, 1.0), 
                                                 gamma = (1.0, 2.0), keepdim=True, p = 0.5),
                                          ])


       # Unimportant for our use. This transform would be useful if we were using temporally consistent
       prev_frame_transform = None # We are only using single frame input so anything here is wasted processing.

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

       target_frame_transform = transforms.Compose([
                                                 kornia.augmentation.RandomSaturation(saturation=(1.2, 1.2), 
                                                                                    p=1.0, keepdim=True),
                                                 ])

       # Get the datasets
       _, dataset_list = getDatasets(vid_dataset_path = dataset_path,
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
                                                        target_transform = target_frame_transform,
                                                        l_th = l_th, c_th = c_th, b_th = b_th, b_th_th = b_th_th)

       synthetic_list_test, dataset_list_test = getDatasets(vid_dataset_path = dataset_path_test,
                                                        loader_type = loader_type,
                                                        vid_dataset_cache_path = cache_test,
                                                        cache_path_synth_smoke = cache_path_synth_smoke_test,
                                                        cache_path_real_smoke  = cache_path_real_smoke_test,
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
                                                        target_transform = target_frame_transform,
                                                        l_th = l_th, c_th = c_th, b_th = b_th, b_th_th = b_th_th)

       dataset:SynthSmokeLoader = ConcatDataset(dataset_list)
       dataset = Subset( dataset, range(0, len(dataset), subsample) )

       test_dataset:SynthSmokeLoader = ConcatDataset(dataset_list_test)
       test_dataset = Subset( test_dataset, range(0, len(test_dataset), subsample_test) )

       synth_test_dataset:SynthSmokeLoader = ConcatDataset(synthetic_list_test)

       return dataset_list, dataset_list_test, synth_test_dataset