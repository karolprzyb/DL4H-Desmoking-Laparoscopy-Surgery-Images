import torch
from torch.utils.data import Dataset, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer

import numpy as np
from typing import List, Tuple

from data_augmentation import Trainer

from losses import PSNR
from losses import SSIMLoss
#from libs.pytorch_fsim.fsim import FSIMc
from torchmetrics.functional import universal_image_quality_index

class SynthTrainer(Trainer):
    '''
        Handles unified logging to tensorboard, validation metrics, and test metrics
        When using, need to fill in:
            trainIteration()  : runs a training iteration for the network, should optimize networks within
            networkInference(): runs a validation/test inference for the network,
                                should return a tuple of inferenced image(s)
                                See validationIteration and testIteration for how this function is used
                                  
    '''
    def __init__(self, device:torch.device, 
                 training_criterion:torch.nn.Module, 
                 writer:SummaryWriter,
                 optimizer:Optimizer,
                 val_crit_list:List[torch.nn.Module] = [PSNR(2.0), SSIMLoss(2.0), universal_image_quality_index], # [PSNR(2.0), torch.nn.MSELoss(), torch.nn.L1Loss()]
                 single_frame:bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.training_criterion = training_criterion
        self.optimizer = optimizer
        self.writer = writer
        self.val_crit_list = val_crit_list
        self.single_frame = single_frame

    def trainIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, *args, **kwargs):
        '''
            Fill in this function to run a training iteration for the network(s)
        '''
        raise NotImplementedError

    def networkInference(self, net: torch.nn.Module, dataset_tuple: Tuple, *args, **kwargs) -> torch.Tensor:
        '''
            Output should be a tuple where:
                1. First element is the inferenced image
                2. Second element is the inferenced smoke mask (if applicable otherwise None)
        '''
        raise NotImplementedError

    def validation(self, net:torch.nn.Module, validation_set:Dataset, *args, **kwargs):
        '''
            Wrapper of trainer validation function to save validation losses to tensorboard
        '''
        # Keeps track of validation losses. See validationIteration for how this is used
        self.total_validation_loss = [0 for _ in range(len(self.val_crit_list))]
        self.total_validation_reference = [0 for _ in range(len(self.val_crit_list))]

        # Keeps track of images to show in tensorboard.
        # Images according to val_indices_to_show should be filled in networkInference function 
        self.bg_img_list:List[torch.Tensor] = []
        self.smoke_img_list:List[torch.Tensor] = []
        self.true_mask_list:List[torch.Tensor] = []
        self.bg_pred_list:List[torch.Tensor] = []
        self.mask_pred_list:List[torch.Tensor] = []
        self.val_indices_to_show = np.linspace(0, len(validation_set), 5, endpoint=False).astype(int)  # type: ignore
        
        super().validation(net, validation_set, *args, **kwargs)

        # Save current epoch
        self.writer.add_scalar('epoch_val', self.epoch_step, self.global_step)

        # Save validation losses to tensorboard
        for val_crit, val_loss, val_loss_ref in zip(self.val_crit_list, self.total_validation_loss, 
                                                        self.total_validation_reference):
            if val_loss != 0:
                self.writer.add_scalar('Synthetic: Validation Loss - {}'.format(val_crit.__class__.__name__), 
                                        val_loss/len(validation_set),   # type: ignore
                                        self.global_step)
            if val_loss_ref != 0:
                self.writer.add_scalar('Synthetic: Validation Loss - Reference - {}'.format(val_crit.__class__.__name__), 
                                        val_loss_ref/len(validation_set),   # type: ignore
                                        self.global_step)
            self.writer.add_scalar('Synthetic: Validation Loss - Delta - {}'.format(val_crit.__class__.__name__), 
                                        (val_loss - val_loss_ref)/len(validation_set),   # type: ignore
                                        self.global_step)

        # Save images to tensorboard
        if self.bg_img_list:
            self.writer.add_images('Synthetic: Raw Img', torch.stack(self.bg_img_list), self.global_step)
        if self.smoke_img_list:
            self.writer.add_images('Synthetic: Composed Img', torch.stack(self.smoke_img_list), self.global_step)
        if self.bg_pred_list:
            self.writer.add_images('Synthetic: Predicted Desmoked', torch.stack(self.bg_pred_list), self.global_step)
        if self.true_mask_list:
            self.writer.add_images('Synthetic: Smoke Mask', torch.stack(self.true_mask_list),  self.global_step)
        if self.mask_pred_list:
            self.writer.add_images('Synthetic: Predicted Smoke Mask', torch.stack(self.mask_pred_list), self.global_step)

    def validationIteration(self, net: torch.nn.Module, dataset_tuple: Tuple, *args, **kwargs):
        smoke_img, true_mask, bg_img = dataset_tuple

        # Run network inference
        bg_pred = self.networkInference(net, dataset_tuple, *args, **kwargs)

        # Compute validation loss and save statistic (total)
        for idx, val_crit in enumerate(self.val_crit_list):
            if bg_pred is not None:
                self.total_validation_loss[idx] += val_crit(bg_pred, bg_img.to(bg_pred.device)).item()
                self.total_validation_reference[idx] += val_crit(smoke_img.to(bg_pred.device), bg_img.to(bg_pred.device)).item()

        # Save images to show in tensorboard
        if self.validation_step in self.val_indices_to_show:
            bg_img =  ( (bg_img[0].cpu() + 1) / 2)
            smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
            true_mask = ( (true_mask[0].cpu() + 1) / 2 )
            self.bg_img_list.append(bg_img)
            self.smoke_img_list.append(smoke_img[:3, : ,:])
            self.true_mask_list.append(true_mask[0,  : ,:].unsqueeze(0))

            if bg_pred is not None:
                bg_pred = ( (bg_pred[0].cpu() + 1) / 2 )
                self.bg_pred_list.append(bg_pred)

    def testIteration(self, net: torch.nn.Module, dataset_tuple: Tuple, *args, **kwargs):
        smoke_img, true_mask, bg_img = dataset_tuple

        # Run network inference
        bg_pred = self.networkInference(net, dataset_tuple, *args, **kwargs)

        # Compute validation loss and save statistic (total)
        for idx, val_crit in enumerate(self.val_crit_list):
            if bg_pred is not None:
                self.total_validation_loss[idx] += val_crit(bg_pred, bg_img.to(bg_pred.device)).item()
                self.total_validation_reference[idx] += val_crit(smoke_img.to(bg_pred.device), bg_img.to(bg_pred.device)).item()

        # Save images to show in tensorboard
        if self.validation_step in self.val_indices_to_show:
            bg_img =  ( (bg_img[0].cpu() + 1) / 2)
            smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
            true_mask = ( (true_mask[0].cpu() + 1) / 2 )
            self.bg_img_list.append(bg_img)
            self.smoke_img_list.append(smoke_img[:3, : ,:])
            self.true_mask_list.append(true_mask[0,  : ,:].unsqueeze(0))

            if bg_pred is not None:
                bg_pred = ( (bg_pred[0].cpu() + 1) / 2 )
                self.bg_pred_list.append(bg_pred)

    def test(self, net:torch.nn.Module, test_set:Dataset, *args, **kwargs):
        '''
            Wrapper of test validation function to save validation losses to tensorboard
        '''
         # Keeps track of validation losses. See validationIteration for how this is used
        self.total_validation_loss = [0 for _ in range(len(self.val_crit_list))]
        self.total_validation_reference = [0 for _ in range(len(self.val_crit_list))]

        # Keeps track of images to show in tensorboard.
        # Images according to val_indices_to_show should be filled in networkInference function 
        self.bg_img_list:List[torch.Tensor] = []
        self.smoke_img_list:List[torch.Tensor] = []
        self.true_mask_list:List[torch.Tensor] = []
        self.bg_pred_list:List[torch.Tensor] = []
        self.mask_pred_list:List[torch.Tensor] = []
        self.val_indices_to_show = np.linspace(0, len(test_set), 5, endpoint=False).astype(int)  # type: ignore
        
        super().test(net, test_set, *args, **kwargs)

        # Save current epoch
        self.writer.add_scalar('epoch_test', self.epoch_step, self.global_step)

        # Save validation losses to tensorboard
        for val_crit, val_loss, val_loss_ref in zip(self.val_crit_list, self.total_validation_loss, 
                                                        self.total_validation_reference):
            if val_loss != 0:
                self.writer.add_scalar('TEST: Validation Loss - {}'.format(val_crit.__class__.__name__), 
                                        val_loss/len(test_set),   # type: ignore
                                        self.global_step)
            if val_loss_ref != 0:
                self.writer.add_scalar('TEST: Validation Loss - Reference - {}'.format(val_crit.__class__.__name__), 
                                        val_loss_ref/len(test_set),   # type: ignore
                                        self.global_step)
            self.writer.add_scalar('TEST: Validation Loss - Delta - {}'.format(val_crit.__class__.__name__), 
                                        (val_loss - val_loss_ref)/len(test_set),   # type: ignore
                                        self.global_step)

        # Save images to tensorboard
        if self.bg_img_list:
            self.writer.add_images('TEST: Raw Img', torch.stack(self.bg_img_list), self.global_step)
        if self.smoke_img_list:
            self.writer.add_images('TEST: Composed Img', torch.stack(self.smoke_img_list), self.global_step)
        if self.bg_pred_list:
            self.writer.add_images('TEST: Predicted Desmoked', torch.stack(self.bg_pred_list), self.global_step)
        if self.true_mask_list:
            self.writer.add_images('TEST: Smoke Mask', torch.stack(self.true_mask_list),  self.global_step)
        if self.mask_pred_list:
            self.writer.add_images('TEST: Predicted Smoke Mask', torch.stack(self.mask_pred_list), self.global_step)
