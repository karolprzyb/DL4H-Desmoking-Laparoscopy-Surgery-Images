import torch
from torch.utils.data import Dataset, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer

import numpy as np

from typing import List

from data_augmentation import Trainer

class SynthTrainer(Trainer):
    def __init__(self, device:torch.device, training_criterion:torch.nn.Module, 
                 validation_criterion:torch.nn.Module, writer:SummaryWriter, optimizer:Optimizer,
                 single_frame:bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.training_criterion = training_criterion
        self.optimizer = optimizer
        self.writer = writer
        self.validation_criterion = validation_criterion
        self.single_frame = single_frame

    def validation(self, net:torch.nn.Module, validation_set:Dataset, *args, **kwargs):
        self.total_validation_loss = 0
        self.total_validation_loss_smoke_detector = 0
        self.bg_img_list:List[torch.Tensor] = []
        self.smoke_img_list:List[torch.Tensor] = []
        self.true_mask_list:List[torch.Tensor] = []
        self.bg_pred_list:List[torch.Tensor] = []
        self.mask_pred_list:List[torch.Tensor] = []
        self.val_indices_to_show = np.linspace(0, len(validation_set), 5, endpoint=False).astype(int)  # type: ignore
        
        super().validation(net, validation_set, *args, **kwargs)

        if self.total_validation_loss != 0:
            self.writer.add_scalar('Synthetic: Validation Loss', 
                                    self.total_validation_loss/len(validation_set),   # type: ignore
                                    self.global_step)
        if self.total_validation_loss_smoke_detector != 0:
            self.writer.add_scalar('Synthetic: Validation Loss - Smoke Detection', 
                                    self.total_validation_loss_smoke_detector/len(validation_set),   # type: ignore
                                    self.global_step)
        self.writer.add_scalar('epoch', self.epoch_step, self.global_step)
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

    def test(self, net:torch.nn.Module, test_set:Dataset, *args, **kwargs):
        self.total_test_loss = 0
        self.smoke_img_list = []
        self.bg_pred_list = []
        self.mask_pred_list = []
        self.additional_img_list:List[torch.Tensor] = []

        idx_to_show = np.linspace(0, len(test_set), 50, endpoint=False).astype(int)  # type: ignore
        dataset = Subset(test_set, list(idx_to_show))
        super().test(net, dataset, *args, **kwargs)

        if self.smoke_img_list:
            self.writer.add_images('Real: Raw Img', torch.stack(self.smoke_img_list), self.global_step)
        if self.bg_pred_list:
            self.writer.add_images('Real: Predicted Desmoked', torch.stack(self.bg_pred_list), self.global_step)
        if self.mask_pred_list:
            self.writer.add_images('Real: Predicted Smoke Mask', torch.stack(self.mask_pred_list), self.global_step)
        if len(self.additional_img_list):
            self.writer.add_images('Real: Predicted Additional Images', torch.stack(self.additional_img_list), 
                                self.global_step)
