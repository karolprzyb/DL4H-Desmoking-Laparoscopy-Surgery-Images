import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
                              
from models import UNet
from utils import *
from losses import L1andMSSSIM, PSNR
from SynthTrainer import SynthTrainer
import arg_parser

def prepareModel(args:arg_parser.argparse.Namespace, num_images:int, dataset_tuple:Tuple) -> torch.nn.Module:
    return UNet(output_channels = num_images,
                input_channels = num_images*3,
                layer_num_channels = args.layers, 
                drop_out_layers_dec = args.drop_out,
                drop_out_layers_enc = [0 for _ in range(len(args.layers))])

class SmokeDetectorTrainer(SynthTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trainIteration(self, net:torch.nn.Module, dataset_tuple:Tuple):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        true_mask = true_mask.to(device=self.device, dtype=torch.float32)

        # Limit smoke_img to 1 image if set:
        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]
            true_mask = true_mask[:, 0, :, :].unsqueeze(1)

        # Logs to help debug dataset issues
        if torch.isnan(smoke_img).any():
            logging.warning("Training data, smoke_img, has NaN!!!!")
            return
        if torch.isnan(true_mask).any():
            logging.warning("Training data, true_mask, has NaN!!!!")
            return

        # Run inference
        mask_pred = net(smoke_img)

        # Compute loss and back-propogate
        loss = self.training_criterion(mask_pred, true_mask)

        if torch.isnan(loss).any():
            logging.warning("Loss has NaN!!!!")
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('training loss - smoke detection', loss, self.global_step)

    def validationIteration(self, net:torch.nn.Module, dataset_tuple:Tuple):
        smoke_img, true_mask, bg_img = dataset_tuple
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)
        true_mask = true_mask.to(device=self.device, dtype=torch.float32)

        # Limit smoke_img to 1 image if set:
        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]
            true_mask = true_mask[:, 0, :, :].unsqueeze(1)

        # Run inference
        mask_pred = net(smoke_img)

        # Get loss
        loss = self.validation_criterion(mask_pred, true_mask)
        if not torch.isnan(loss).any():
            self.total_validation_loss_smoke_detector += loss.item()

        # Save images to show
        if self.validation_step in self.val_indices_to_show:
            bg_img =  ( (bg_img[0].cpu() + 1) / 2)
            smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
            true_mask = ( (true_mask[0].cpu() + 1) / 2 )
            mask_pred = ( (mask_pred[0].cpu() + 1) / 2 )
            
            # Note: Moved image tensor to CPU to save gpu memory.
            self.bg_img_list.append(transforms.Resize(int(bg_img.shape[-1]))(bg_img))
            self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
            self.true_mask_list.append(transforms.Resize(int(true_mask.shape[-1]))(true_mask[0,  : ,:].unsqueeze(0)))
            self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))

    def testIteration(self, net:torch.nn.Module, dataset_tuple:Tuple):
        smoke_img, _, _ = dataset_tuple

        # Set device for images
        smoke_img = smoke_img.to(device=self.device, dtype=torch.float32)

        # Limit smoke_img to 1 image if set:
        if self.single_frame:
            smoke_img = smoke_img[:, :3, :, :]

        # Run inference
        mask_pred = net(smoke_img)

        # Save images to show 
        smoke_img = ( (smoke_img[0].cpu() + 1) / 2)
        mask_pred = ( (mask_pred[0].cpu()   + 1) / 2)

        self.smoke_img_list.append(transforms.Resize(int(smoke_img.shape[-1]))(smoke_img[:3, : ,:]))
        self.mask_pred_list.append(transforms.Resize(int(mask_pred.shape[-1]))(mask_pred[0,  : ,:].unsqueeze(0)))



if __name__ == '__main__':
    # Command line arguments
    parser = arg_parser.create()

    # --- Network Architecture Commands for model
    parser.add_argument('--layers', type=int, metavar='unet_layers_', nargs='+',
                        default=[64, 128, 256, 512])
    parser.add_argument('--drop_out', type=float, metavar='drop_out_layers_decoder', nargs='+',
                        default=[0.05, 0.0, 0.0, 0.0])

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
    if args.single_frame:
        num_images = 1
    else:
        num_images = len(multi_frame) + 1
    net = prepareModel(args, num_images, train_dataset[0]).to(device)
    logging.info(net)
    training_criterion = torch.nn.L1Loss()
    validation_criterion  = PSNR(2)

    # Load saved network
    if args.load_net:
        net.load_state_dict(torch.load(args.load_net))

     # Initialize optimizer
    optimizer = Adam(net.parameters(), lr=args.lr, betas=args.adam) 

    # Initialize and run trainer
    smoke_trainer = SmokeDetectorTrainer(device = device, 
                                         training_criterion = training_criterion, 
                                         validation_criterion = validation_criterion, 
                                         writer = writer, optimizer = optimizer,
                                         save_path = args.save, 
                                         num_epoch = args.epochs, 
                                         batch_size = args.batch,
                                        # run_val_and_test_every_steps = 1,
                                        single_frame = args.single_frame
                                         )
    smoke_trainer.train(net, train_dataset, val_dataset, real_smoke_dataset)