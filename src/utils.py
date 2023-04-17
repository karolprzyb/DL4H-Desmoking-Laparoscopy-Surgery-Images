import os
import numpy as np
import sys
from typing import Tuple, List, Any, Optional, Literal
import logging
from tqdm import tqdm
import shutil

import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset, Subset, DataLoader
import cv2
import kornia


from data_augmentation import SmokeDirectoryLoader, VideoLoader,\
                              SynthSmokeLoader, BilateralFilter, getVidData

from data_augmentation.SynthSmokeLoader import compose_smoke_and_vid_img

logger = logging.getLogger()

def generateVideo(smoke_dataset:SynthSmokeLoader, net:torch.nn.Module,
                  save_file_path:str, device:torch.device, denormalize_imgs:bool = True,
                  single_frame:bool=False, apply_bf:bool=True,
                  smoke_net:torch.nn.Module = None):
    '''
        Generates a video where the two images are put side to side. The left
        image is the input from smoke_dataset. The righ timage is the network
        applied to the left image.
        
        Inputs:
            smoke_dataset: dataset to generate video from
            net: Network to run inference with
            save_file_path: file-path for the video to be saved
            device: device to run inference with
    '''
    out_video:Any = None
    val_loader = DataLoader(smoke_dataset, shuffle=False, drop_last=True, 
                            batch_size=1, num_workers=4, pin_memory=False)

    net.eval()
    with torch.no_grad():
        with tqdm(total=len(smoke_dataset), desc='Real World Validation', unit='img') as pbar:
            for smoke_img, _, _ in val_loader:
                smoke_img = smoke_img.to(device=device, dtype=torch.float32)

                # Single frame case
                if single_frame:
                    smoke_img = smoke_img[:, :3, :, :]

                # Inference
                if smoke_net is not None:
                    smoke_mask = smoke_net(smoke_img)
                    pred = net(torch.cat([smoke_img, smoke_mask], dim=1))
                else:
                    pred = net(smoke_img)

                # Case where network outputs more than one image.
                if isinstance(pred, tuple):
                    additional_outputs = pred[1].squeeze(0)
                    pred = pred[0].squeeze(0)
                else:
                    additional_outputs = None
                    pred = pred.squeeze(0)

                # Denormalize images if desired
                if denormalize_imgs:
                    smoke_img = ( (smoke_img + 1) / 2)
                    pred   = ( (pred   + 1) / 2)

                # Move to CPU to begin opencv operations
                smoke_img = smoke_img.cpu()
                pred = pred.cpu()

                # Case where network is smoke detector, make sure to only take one channel
                if pred.shape[-3] == 1:
                    pred = kornia.color.grayscale_to_rgb(pred[0].unsqueeze(0))

                # Apply bilateral filter if desired
                if apply_bf:
                    pred = BilateralFilter(-1, 0.15, 15)(pred.squeeze(0))

                # Concat input vs output for left and right by:
                # --- cropping to align images
                smoke_img = smoke_img[:, :3, :, :]
                smoke_img = transforms.CenterCrop((pred.shape[-2], pred.shape[-1]))(smoke_img).squeeze(0)
                # --- concatenating images
                out_img = torch.cat((smoke_img, pred), dim=1)
                # --- switch rgb to bgr
                out_img = kornia.color.rgb_to_bgr(out_img)
                # --- conver to uint8 and transpose from CHW -> HWC
                out_img[out_img > 1] = 1
                out_img[out_img < 0] = 0
                out_img = transforms.ConvertImageDtype(torch.uint8)(out_img).transpose(0, -1).cpu().detach().numpy()
                
                # Use video writer to save output
                if out_video is None:
                    out_video = cv2.VideoWriter(save_file_path,
                                                cv2.VideoWriter_fourcc('M','J','P','G'), 30, 
                                                (out_img.shape[1], out_img.shape[0]))
                out_video.write(out_img)
                
                pbar.update(1)
                
    out_video.release()


def synthDatafromVidData(vid_data_list:List[VideoLoader],
                            mask_vid_data_list:List[np.ndarray],
                            smoke_dataset_path:Optional[str],
                            multi_frame:List[int],
                            syn_dataset_cache_path:str,
                            synth_smoke_transforms:Any = None,
                            num_synth_smoke_per_bg_img:int = 10,
                            clear_cache:bool = False,
                            bg_img_transform:Any = None,
                            output_transform:Any = None,
                            prev_frame_transform:Any = None,
                            target_transform=None,
                            compose_function=compose_smoke_and_vid_img,
                            ) -> List[SynthSmokeLoader]:
    '''
        Returns a concatenation of SynthSmokeLoader's where each SynthSmokeLoader corresponds
        to one temporally consistent video clip. The video clips are generated from vid_data_list
        where an additional mask, mask_vid_data_list, is applied. I.e. only vid_data corresponding to
        a 1 in mask_vid_data will be used.
        Inputs:
            vid_data_list: list of VideoLoader's for bg images to create SynthSmokeLoader
            mask_vid_data_list: list of masks for vid_data_list. Only vid_data corresponding to a 1
                will be used to create clips for the SynthSmokeLoader

            smoke_dataset_path: str path which points to the top fodler holding synthetic smoke images
                See SmokeDirectoryLoader for more details on how this dataset is created.
                If smoke_dataset_path = '', then an synth_smoke_data is None
            synth_smoke_transforms: transforms to augment the synthetic smoke data

            multi_frame: list of indices which denote additional images to be retrieved from the
                output SynthSmokeLoader (e.g. [1,2,3])
            syn_dataset_cache_path: top folder to cache the SynthSmokeLoaders
            num_synth_smoke_per_bg_img: Repeats making SynthSmokeLoader per clip this many times
                hence increasing the dataset.
            clear_cache: Clear cache and re-cache VIAVideoLoader's
            bg_img_transform: transform applied to the background images in SynthSmokeLoaders
            output_transform: transformed applied to output images in SynthSmokeLoaders
            prev_frame_transform: transform applied to previous frames. See SynthSmokeLoader for more details.
            target_transform: transform applied to target when creating SynthSmokeLoader
    '''

    # Get synthetic smoke dataset
    if smoke_dataset_path:
        synth_smoke_data:Any = ConcatDataset([SmokeDirectoryLoader(os.path.join(smoke_dataset_path, directory), 
                                                                                    synth_smoke_transforms) 
                                                    for directory in os.listdir(smoke_dataset_path)  
                                                    if os.path.isdir(os.path.join(smoke_dataset_path, directory))])
    else:
        synth_smoke_data = None

    # Prepare cache folder
    if syn_dataset_cache_path:
        if not os.path.isdir(syn_dataset_cache_path):
            os.mkdir(syn_dataset_cache_path)

    # Dump parameters for the synth data
    with open(os.path.join(syn_dataset_cache_path, "generation_parameters.txt"), "w") as text_file:
        text_file.write("Smoke dataset {}\n".format(smoke_dataset_path))
        text_file.write("Multi-frame {}\n".format(multi_frame))
        text_file.write("Synth smoke transforms {}\n".format(synth_smoke_transforms))
        text_file.write("Num synth smoke per bg img {}\n".format(num_synth_smoke_per_bg_img))
        text_file.write("Bg img transform {}\n".format(bg_img_transform))
        text_file.write("Output transform {}\n".format(output_transform))
        text_file.write("Previous frame transform {}\n".format(prev_frame_transform))
        for idx, vid_data in enumerate(vid_data_list):
            text_file.write("Video loader {} video file {}\n".format(idx, vid_data.video_file_path))
            text_file.write("Video loader {} cache directory {}\n".format(idx, vid_data.cache_directory))
            text_file.write("Video loader {} transform {}\n".format(idx, vid_data.transform))

    # Internal function used to subset VIAVideoLoader from mask information
    def mask_to_clip_idx(mask:np.ndarray) -> np.ndarray:
        '''
            mask is 1D array of 0's and 1's where 1 is a mask for the clip of interest
            returns indices of start and end clip in the following format:
                [s_1, e_1, ...]
            where [s_1, e_1) bounds the clip.                
        '''
        clip_idx = np.where(mask[:-1] != mask[1:])[0] + 1
        # If the clip starts with 1, then starting idx is 0
        if mask[0] == 1:
            clip_idx = np.insert(clip_idx, 0, 0)

        # If the clip ends with a 1, then ending idx is len(mask),
        #   so append len(mask) to the end.
        if mask[-1] == 1:
            clip_idx = np.append(clip_idx, len(mask))
        
        return clip_idx

    # Loop through VIAVideoLoaders and return syn_data
    syn_data:List[SynthSmokeLoader] = []
    for vid_idx, vid_data in enumerate(vid_data_list):

        # Get clips from mask for this vid_data
        clip_indices = mask_to_clip_idx(mask_vid_data_list[vid_idx])

        # Loop over clips
        for clip_idx in range(0, len(clip_indices), 2):
            start_idx = clip_indices[clip_idx]
            end_idx   = clip_indices[clip_idx + 1]

            clip_data = Subset(vid_data, list(range(start_idx, end_idx)))

            # Skip clip if it is too short
            if max(multi_frame) >= len(clip_data):
                continue
            
            # Make t_syn_data                
            for synth_smoke_idx in range(num_synth_smoke_per_bg_img):
                if syn_dataset_cache_path:
                    cache_directory = os.path.join(syn_dataset_cache_path, 'synth_smoke_v%03d_c%04d_s%03d' 
                                                                            % (vid_idx, int(clip_idx/2), 
                                                                                synth_smoke_idx))
                else:
                    cache_directory = None

                syn_data.append(SynthSmokeLoader(clip_data, synth_smoke_data, 
                                                bg_img_transform = bg_img_transform,
                                                idx_of_prev_frame = multi_frame,
                                                output_transform = output_transform,
                                                prev_frame_transform = prev_frame_transform,
                                                cache_directory = cache_directory,
                                                clear_cache_directory = clear_cache,
                                                target_transform=target_transform,
                                                compose_function=compose_function))

    # Searching through the list of syn_data uses bisect_right to index data.
    # This is a recursive search which we need to increase if list of syn_data is long
    if sys.getrecursionlimit() < 2*len(syn_data):
        sys.setrecursionlimit(2*len(syn_data))
    return syn_data


def getDatasets(vid_dataset_path:str,
                loader_type:Literal["via", "cholec_smoke"],
                smoke_dataset_path:str,
                vid_dataset_cache_path:str = None,
                cache_path_synth_smoke:str = None,
                cache_path_real_smoke:str = None,
                multi_frame:List[int] =[],
                vid_dataset_labels:List[str] = None,
                num_synth_smoke_per_bg_img:int = 10,
                vid_dataset_apply_scope_cropping:bool = False,
                vid_transform:Any = None,
                clear_cache_directory:bool = False,
                synth_smoke_transforms:Any = None,
                bg_img_transform:Any = None,
                output_transform_synth:Any = None,
                output_transform_real:Any = None,
                prev_frame_transform:Any = None,
                target_transform = None,
                compose_function = compose_smoke_and_vid_img,
                l_th:int = 8, c_th:float = 0.15, b_th:float = 0.4, b_th_th:float = 0.55) \
                        -> Tuple[List[SynthSmokeLoader], List[SynthSmokeLoader]]:
    '''
    Returns two SynthSmokeLoader datasets to train and real-world test de-smoking networks.
    Inputs:
        vid_dataset_path: str path which points to the top folder holding temporally labelled datasets
            Each sub-folder contains a single mp4 and csv file for the video and labels respectively.
        loader_type: type of video loader used. Can be VIAVideo or cholec80 smoke 
        vid_dataset_cache_path: top folder to cache the mp4 --> .png files for VideoLoader
        cache_path_synth_smoke: top folder to cache the synth smoke data
        cache_path_real_smoke: top folder to cache the real smoke data
        clear_cache_directory: clear cache directory for VideoLoader AND eventual SynthSmokeLoader

        vid_dataset_apply_scope_cropping: applies a cropping on the vid images for scope to NOT have the 
            black-scope isn't there. Uses getScopeImgCircleParameters to compute cropping dimensions
        vid_transform: transforms applied to the images from the vid_dataset
        vid_dataset_labels: list which holds the label key for the VIAVideoLoader
            MUST contain 'Smoke' label. Not needed for Cholec80 smoke

        smoke_dataset_path: str path which points to the top fodler holding synthetic smoke images
            See SmokeDirectoryLoader for more details on how this dataset is created
        synth_smoke_transforms: transforms to augment the synthetic smoke data

        bg_img_transform: transform applied to the background images in returned dataset SynthSmokeLoader
        output_transform_synth: transformed applied to output images in returned dataset SynthSmokeLoader for synth data
        output_transform_real: transformed applied to output images in returned dataset SynthSmokeLoader for real smoke data
        prev_frame_transform: transform applied to previous frames. See SynthSmokeLoader for more details.
            This is ONLY applied to the synthetic data, NOT the real smoke data
        target_transform: transform applied to the target images in returned dataset SynthSmokeLoader for synth data
        multi_frame: list of indices which denote additional images to be retrieved from the
            output SynthSmokeLoader (e.g. [1,2,3])
    
    Outputs:
        dataset: SynthSmokeLoader from vid_dataset_path with None labels and smoke_dataset_path
        real_dataset: SynthSmokeLoader from vid_dataset_path with Smoke labels and NO synthetic smoke
    '''
    # Get video data for real smoke data
    if loader_type == 'via':
        vid_data_list = getVidData(vid_dataset_path = vid_dataset_path,
                                   vid_dataset_cache_path = vid_dataset_cache_path,
                                   temporal_labels = vid_dataset_labels,
                                   transform = vid_transform,
                                   scope_cropping = vid_dataset_apply_scope_cropping,
                                   loader_type = 'via',
                                   clear_cache_directory = clear_cache_directory)
    elif loader_type == 'cholec_smoke':
        vid_data_list = getVidData(vid_dataset_path = vid_dataset_path,
                                   vid_dataset_cache_path = vid_dataset_cache_path,
                                   transform = vid_transform,
                                   scope_cropping = vid_dataset_apply_scope_cropping,
                                   loader_type='cholec_smoke',
                                   clear_cache_directory = clear_cache_directory)
    else:
        raise ValueError("loader_type must be 'via' or 'cholec_smoke'")

    # Get real_smoke_dataset
    mask_vid_data_list = []
    for vid_data in vid_data_list:
        if loader_type == "via":
            smoke_label_mask = vid_data.getLabels()[:, vid_dataset_labels.index('Smoke')]
        elif loader_type == "cholec_smoke":
            smoke_label_mask = vid_data.getLabels() == 1
        mask_vid_data_list.append(smoke_label_mask.cpu().numpy())

    real_smoke_dataset = synthDatafromVidData(vid_data_list, mask_vid_data_list,
                                              None,
                                              multi_frame = multi_frame,
                                              syn_dataset_cache_path = cache_path_real_smoke,
                                              num_synth_smoke_per_bg_img = 1,
                                              clear_cache = clear_cache_directory,
                                              output_transform = output_transform_real)

    # Get video data for background images
    img_quality_data = getVidData(vid_dataset_path = vid_dataset_path,
                                  vid_dataset_cache_path = vid_dataset_cache_path,
                                  transform = vid_transform,
                                  scope_cropping = vid_dataset_apply_scope_cropping,
                                  clear_cache_directory = clear_cache_directory,
                                  loader_type = 'qual')

    # Get synth_smoke_dataset
    mask_vid_data_list = []
    for qual_data, vid_data in zip(img_quality_data, vid_data_list):
        if loader_type == "via":
            label_mask = ~torch.sum(vid_data.getLabels(), 1).type(torch.bool)
        elif loader_type == "cholec_smoke":
            label_mask = vid_data.getLabels()[:, 0] != 1

        brightness_data = (torch.tensor([ torch.sum(img_d > b_th)/len(img_d) 
                                                for img_d in qual_data.bright]))
        quality_mask = (qual_data.lap > l_th) * (qual_data.col > c_th) * (brightness_data > b_th_th)
        print(label_mask.size()[0])
        print(quality_mask.size()[0])
        if label_mask.size()[0] == quality_mask.size()[0]+1: # TODO THIS IS AN IMPROVED IDEA BUT SHOULD FIX ROOT CAUSE IN FUTURE
            label_mask = label_mask[:-1]
        mask_vid_data_list.append(torch.logical_and(label_mask, quality_mask).cpu().numpy()) # TODO I DO NOT THINK THIS IS A GREAT IDEA BUT I GUESS WE WILL SEE!!

    synth_smoke_dataset = synthDatafromVidData(vid_data_list, mask_vid_data_list,
                                                smoke_dataset_path,
                                                multi_frame = multi_frame,
                                                syn_dataset_cache_path = cache_path_synth_smoke,
                                                synth_smoke_transforms = synth_smoke_transforms,
                                                num_synth_smoke_per_bg_img = num_synth_smoke_per_bg_img,
                                                clear_cache = False,
                                                bg_img_transform = bg_img_transform,
                                                output_transform = output_transform_synth,
                                                prev_frame_transform = prev_frame_transform,
                                                target_transform = target_transform,
                                                compose_function = compose_function, # TODO this should be integrated more nicely
                                                )

    return synth_smoke_dataset, real_smoke_dataset
