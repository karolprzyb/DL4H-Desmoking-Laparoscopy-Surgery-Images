import os
import logging
import argparse
import json
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim import Optimizer
                              
from models import UNet
from utils import *
from losses import PSNR
from SynthTrainer import SynthTrainer
import arg_parser

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels:int = 3):
        super().__init__()
        # self.encoder =  Encoder(input_channels = 3,
        #                         output_channels = 512,
        #                         layer_num_channels = [64, 128, 256],
        #                         drop_out_layers = [0 for _ in range(3)],
        #                         save_skips = True,
        #                         kernel_size = 4,
        #                         leaky_relu = True,
        #                         batch_norm = False)


        # self.conv = torch.nn.Conv2d(512, 1, kernel_size=4, stride=1)
        sequence = [torch.nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), 
                    torch.nn.LeakyReLU(0.2, True)]
        sequence += [torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
                     torch.nn.BatchNorm2d(128),
                     torch.nn.LeakyReLU(0.2, True) 
                     ]
        sequence += [torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
                     torch.nn.BatchNorm2d(256),
                     torch.nn.LeakyReLU(0.2, True) 
                     ]
        sequence += [torch.nn.ZeroPad2d(2)]
        sequence += [torch.nn.Conv2d(256, 512, kernel_size=4, stride=1), 
                     torch.nn.BatchNorm2d(512),
                     torch.nn.LeakyReLU(0.2, True) 
                     ]
        sequence += [torch.nn.ZeroPad2d(2)]
        sequence += [torch.nn.Conv2d(512, 1, kernel_size=4, stride=1)]
       # sequence += [torch.nn.Sigmoid()] #IT APPEARS THIS WAS NOT USED IN THE PAPER. COULD LOOK AT ITS IMPACT LATER POSSIBLY
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

def prepareModel(args:arg_parser.argparse.Namespace, num_images:int, dataset_tuple:Tuple) \
        -> Tuple[torch.nn.Module, Union[torch.nn.Module, None], torch.nn.Module]:
    # Desmoke network
    if args.load_net_smoke_detect:
        input_channels_for_desmoke = 3*num_images + num_images
    else:
        input_channels_for_desmoke = 3*num_images
    desmoke_net =  UNet(output_channels = dataset_tuple[-1].shape[0],
                        input_channels = input_channels_for_desmoke,
                        layer_num_channels = args.layers,
                        drop_out_layers_dec = args.drop_out,
                        drop_out_layers_enc = [0 for _ in range(len(args.layers))])

    # Discriminator
    if args.pix2pix_condition:
        d_in_channels = dataset_tuple[-1].shape[0] + 3*num_images
    else:
        d_in_channels = dataset_tuple[-1].shape[0]
    discriminator_net = Discriminator(input_channels=d_in_channels)

    # Smoke detector
    if args.load_net_smoke_detect:
        # -- Load arguments
        smoke_args_path = os.path.join( os.path.dirname(args.load_net_smoke_detect), "commandline_args.txt")
        if not os.path.isfile(smoke_args_path):
            raise ValueError('Could not find commandline_args.txt file in the same directory as {}'.format(args.load_net_smoke_detect))
        smoke_args = argparse.Namespace()
        with open(smoke_args_path) as smoke_cmd_file:
            smoke_args_dict = json.load(smoke_cmd_file)
            smoke_args.__dict__.update(smoke_args_dict)
        # -- Check if args from load_net_smoke_detect has the same single_frame setting as here
        if  smoke_args.single_frame != args.single_frame:
            raise ValueError('Smoke network needs to be {} but the arg is: {} '.format(args.single_frame, 
                                                                                       smoke_args.single_frame))
        # -- Load smoke detect network
        smoke_detect_net = UNet(output_channels = num_images,
                                input_channels = 3*num_images,
                                layer_num_channels = smoke_args.layers, 
                                drop_out_layers_dec = smoke_args.drop_out,
                                drop_out_layers_enc = [0 for _ in range(len(smoke_args.layers))])
        smoke_detect_net.load_state_dict(torch.load(args.load_net_smoke_detect))
        smoke_detect_net.eval()
        for param in smoke_detect_net.parameters():
            param.requires_grad = False
    else:
        smoke_detect_net = None

    return desmoke_net, smoke_detect_net, discriminator_net                   

class GANTrainer(SynthTrainer):
    def __init__(self, optimzer_d:Optimizer, pix2pix_condition:bool = False,
                    *args, **kwargs):
        self.optimizer_d = optimzer_d
        self.pix2pix_condition = pix2pix_condition
        super().__init__(*args, **kwargs)

    def trainIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, net_d:torch.nn.Module,
                         net_smoke:torch.nn.Module = None):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        bg_img = bg_img.to(device=self.device, dtype=torch.float32)

        if torch.isnan(smoke_img).any():
            logging.warning("Training data, smoke_img, has NaN!!!!")
            return
        if torch.isnan(bg_img).any():
            logging.warning("Training data, bg_img, has NaN!!!!")
            return
        
        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = net_smoke(smoke_img)
        else:
            mask_pred = None
        
        # Train D
        # -- Inference on G and D w/ both predicted "fake" data and real data to train D
        if mask_pred is not None:
            bg_pred = net(torch.cat([smoke_img, mask_pred], dim = 1))
        else:
            bg_pred = net(smoke_img)

        # -- In pix2pix the discriminator is conditioned with the input image
        if self.pix2pix_condition:
            d_input_fake = torch.cat([bg_pred, smoke_img], dim = 1)
            d_input_real = torch.cat([bg_img , smoke_img], dim = 1)
        else:
            d_input_fake = bg_pred
            d_input_real = bg_img
        
        # -- Inference on D to train it
        d_pred_fake = net_d(d_input_fake)
        d_pred_real = net_d(d_input_real)
        d_pred = torch.cat([d_pred_fake, d_pred_real])
        labels = torch.cat([torch.ones_like(d_pred_fake), torch.zeros_like(d_pred_real)])

        # -- Compute loss and back-propogate to update D
        d_loss = self.training_criterion(d_pred, labels)
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train G
        # --  Re-inference G and D to train G
        if mask_pred is not None:
            bg_pred_g = net(torch.cat([smoke_img, mask_pred], dim = 1))
        else:
            bg_pred_g = net(smoke_img)

        # --  In pix2pix the discriminator is conditioned with the input image
        if self.pix2pix_condition:
            d_input = torch.cat([bg_pred_g, smoke_img], dim = 1)
        else:
            d_input = bg_pred_g
        d_pred_fake_g = net_d(d_input)

        # -- Compute loss and back-propogate to update G
        g_loss = self.training_criterion(d_pred_fake_g, torch.ones_like(d_pred_fake_g))  \
                            + 100*torch.nn.L1Loss()(bg_pred_g, bg_img)
        self.optimizer.zero_grad()
        g_loss.backward()
        self.optimizer.step()
        
        # Save loss in tensorboard
        self.writer.add_scalar('training loss', g_loss.item(), self.global_step)
        self.writer.add_scalar('training loss: d', d_loss.item(), self.global_step)

    def validationIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, net_d:torch.nn.Module,
                            net_smoke:torch.nn.Module = None):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        bg_img = bg_img.to(device=self.device, dtype=torch.float32)

        # Single frame case
        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = net_smoke(smoke_img)
        else:
            mask_pred = None

        # Run inference
        if mask_pred is not None: 
            bg_pred = net(torch.cat([smoke_img, mask_pred], dim = 1))
        else:
            bg_pred = net(smoke_img)

        # Get loss
        loss = self.validation_criterion(bg_pred, bg_img)
        if not torch.isnan(loss).any():
            self.total_validation_loss += loss.item()

        # Save images to show
        if self.validation_step in self.val_indices_to_show:
            bg_img =  ( (bg_img[0].cpu() + 1) / 2)
            smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
            true_mask = ( (true_mask[0].cpu() + 1) / 2 )
            bg_pred = ( (bg_pred[0].cpu() + 1) / 2 )
            
            # Note: Moved image tensor to CPU to save gpu memory.
            self.bg_img_list.append(transforms.Resize(int(bg_img.shape[-1]))(bg_img))
            self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
            self.true_mask_list.append(transforms.Resize(int(true_mask.shape[-1]))(true_mask[0,  : ,:].unsqueeze(0)))
            self.bg_pred_list.append(transforms.Resize(int(bg_pred.shape[-1]))(bg_pred))

            if mask_pred is not None:
                mask_pred = ( (mask_pred[0].cpu() + 1) / 2 )
                self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))
            

    def testIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, net_d:torch.nn.Module, 
                        net_smoke:torch.nn.Module = None):
        smoke_img, _, _ = dataset_tuple

        # Set device for images
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)

        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Detect smoke from smoke detector network if it exists
        if net_smoke:
            mask_pred = net_smoke(smoke_img)
        else:
            mask_pred = None

        # Run inference
        if mask_pred is not None:
            bg_pred = net(torch.cat([smoke_img, mask_pred], dim = 1))
        else:
            bg_pred = net(smoke_img)

        # Save images to show 
        smoke_img = transforms.CenterCrop((bg_pred.shape[-2], bg_pred.shape[-1]))(smoke_img)
        smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
        bg_pred   = ( (bg_pred[0].cpu()   + 1) / 2)

        self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
        self.bg_pred_list.append(transforms.Resize(int(bg_pred.shape[-1]))(bg_pred))
        if mask_pred is not None:
                mask_pred = ( (mask_pred[0].cpu() + 1) / 2 )
                self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))

if __name__ == '__main__':
    # Command line arguments
    parser = arg_parser.create()

    # --- Network architecture and training commands for model
    parser.add_argument('--mfunet', action = "store_true")
    parser.add_argument('--layers', type=int, metavar='unet_layers', nargs='+',
                        default=[64, 128, 256, 512, 512, 512, 512, 1024])
    parser.add_argument('--drop_out', type=float, metavar='drop_out_layers_decoder', nargs='+',
                        default=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument('--load_net_smoke_detect', type=str, metavar='path to *.pth holding saved smoke network' + \
                                                                   ' and in the same directory as commandline_args.txt. ' + \
                                                                    'This net should be traijned by smoke_detector.py')
    parser.add_argument('--pix2pix_condition', action = "store_true")


    # --- Parse args and save
    args = parser.parse_args()
    args.__dict__["run_file"] = os.path.abspath(__file__)
    arg_parser.save(args)

    # Save Tensorboard
    writer = SummaryWriter(args.save)

    # Save log output in file-path too
    logging.basicConfig(filename=os.path.join(args.save, 'run.log'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # Pick device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device('cpu')

    # Prepare datasets
    from synth_data_settings import train_dataset, val_dataset, real_smoke_dataset, multi_frame
    
    # Prepare model
    if args.single_frame or True: #THIS SHOULD BE args.single_frame, need to change back later
        num_images = 1
    else:
        num_images = len(multi_frame) + 1
    net, net_smoke, net_d = prepareModel(args, num_images, train_dataset[0])
    net = net.to(device)
    if net_smoke:
        net_smoke = net_smoke.to(device)
    net_d = net_d.to(device)
    logging.info(net)
    logging.info(net_d)

    # Prepare traiing
    training_criterion  = torch.nn.L1Loss()
    validation_criterion  = PSNR(2)

    # Load saved network
    if args.load_net:
        net.load_state_dict(torch.load(args.load_net))

    # Initialize optimizer
    optimizer = Adam(net.parameters(), lr=args.lr, betas=args.adam) 
    optimzer_d = Adam(net_d.parameters(), lr=args.lr, betas=args.adam) 


    # Initialize and run trainer
    trainer = GANTrainer(device = device, 
                          training_criterion = training_criterion, 
                          validation_criterion = validation_criterion,
                          writer = writer, 
                          optimizer = optimizer, 
                          optimzer_d = optimzer_d,
                          save_path = args.save, num_epoch = args.epochs, batch_size = args.batch,
                          single_frame = True, # THIS SHOULD BE args.singl_frame
                          pix2pix_condition = args.pix2pix_condition,
                          num_workers = 0, # NEEDED FOR PYAV SINCE IT CANNOT BE PICKLED. FAILS WITH WORKERS
                        #   run_val_and_test_every_steps = 1,
                          )

    trainer.train(net, train_dataset, val_dataset, real_smoke_dataset, net_d = net_d, net_smoke = net_smoke)