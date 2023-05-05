import os
import logging
import json
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim import Optimizer                 
from synth_data_source import loadData
from dcpref import DarkChannelPrior


from utils import *
from SynthTrainer import SynthTrainer


class GANTrainer(SynthTrainer):
    def __init__(self, optimzer_d:Optimizer, use_dark_channel:bool=False, *args, **kwargs):
        # Just save what we need.
        self.optimizer_d = optimzer_d
        self.use_dark_channel = use_dark_channel
        if self.use_dark_channel:
            self.dcp_gen = DarkChannelPrior(kwargs['device'])
        super().__init__(*args, **kwargs)
        #torch.autograd.set_detect_anomaly(True)

    def trainIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, net_d:torch.nn.Module):
        smoke_img, true_mask, bg_img = dataset_tuple
        
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        bg_img = bg_img.to(device=self.device, dtype=torch.float32)

        if self.use_dark_channel:
            dark_img = self.dcp_gen(smoke_img).to(device=self.device)
            smoke_img = torch.cat((smoke_img, dark_img), dim=1)

        if torch.isnan(smoke_img).any():
            logging.warning("Training data, smoke_img, has NaN!!!!")
            return
        if torch.isnan(bg_img).any():
            logging.warning("Training data, bg_img, has NaN!!!!")
            return
        
        # Train D
        # -- Inference on G and D w/ both predicted "fake" data and real data to train D
        bg_pred = net(smoke_img)
        
        # -- Inference on D to train it
        d_pred_fake = net_d(bg_pred)
        d_pred_real = net_d(bg_img)
        d_pred = torch.cat([d_pred_fake, d_pred_real])
        labels = torch.cat([torch.ones_like(d_pred_fake), torch.zeros_like(d_pred_real)])

        # -- Compute loss and back-propogate to update D
        d_loss = self.training_criterion(d_pred, labels)
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()

        # Train G
        bg_pred_g = net(smoke_img)
        d_pred_fake_g = net_d(bg_pred_g)

        # -- Compute loss and back-propogate to update G
        g_loss = self.training_criterion(d_pred_fake_g, torch.ones_like(d_pred_fake_g))  \
                            + 100*torch.nn.L1Loss()(bg_pred_g, bg_img)
        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()
        
        # Save loss in tensorboard
        self.writer.add_scalar('training loss', g_loss.item(), self.global_step)
        self.writer.add_scalar('training loss: d', d_loss.item(), self.global_step)

    def networkInference(self, net: torch.nn.Module, dataset_tuple: Tuple, net_d:torch.nn.Module) -> torch.Tensor:
        
        smoke_img, _, _ = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)

        if self.use_dark_channel:
            dark_img = self.dcp_gen(smoke_img).to(device=self.device, dtype=torch.float32)
            smoke_img = torch.cat((smoke_img, dark_img), dim=1)

        bg_pred = net(smoke_img)
        
        return bg_pred

def run(args:dict, UNET:torch.nn.Module, discriminator:torch.nn.Module,
        train_dataset:SynthSmokeLoader, val_dataset:SynthSmokeLoader, test_dataset:SynthSmokeLoader):
    
    os.makedirs(args['save'])
    with open(os.path.join(args['save'], 'commandline_args.txt'), 'w') as f:
        json.dump(args, f, indent=2)

    # Save Tensorboard
    writer = SummaryWriter(args['save'])

    # Save log output in file-path too
    logging.basicConfig(filename=os.path.join(args['save'], 'run.log'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    # Pick device, prefer GPU if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:{}'.format(args['gpu']))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device('cpu')

    # Prepare networks
    UNET = UNET.to(device)
    discriminator = discriminator.to(device)
    logging.info(UNET)
    logging.info(discriminator)

    # Prepare traiing
    training_criterion  = torch.nn.L1Loss()

    # # Load saved network
    # if args['load_net'] is not None:
    #     net.load_state_dict(torch.load(args['load_net']))

    # Initialize optimizer
    optimizer = Adam(UNET.parameters(), lr=args['lr'], betas=[0.5, 0.999]) 
    optimzer_d = Adam(discriminator.parameters(), lr=args['lr'], betas=[0.5, 0.999]) 


    # Initialize and run trainer
    trainer = GANTrainer(device = device, 
                          training_criterion = training_criterion, 
                          writer = writer, 
                          optimizer = optimizer, 
                          optimzer_d = optimzer_d,
                          save_path = args['save'], num_epoch = args['epochs'], batch_size = args['batch'],
                          single_frame = True,
                          use_dark_channel=args['use_dark_channel'],
                           run_val_and_test_every_steps = args['run_val_and_test_every_steps']
                          )

    trainer.train(UNET, train_dataset, val_dataset, test_dataset, net_d = discriminator)

    return trainer