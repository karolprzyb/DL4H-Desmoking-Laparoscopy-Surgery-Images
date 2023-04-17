import argparse
import datetime
import os
import json

def create():
    parser = argparse.ArgumentParser(description="Training of Smoke Detection Algorithms")

    # --- Dataset Commands
    parser.add_argument('--save', type=str, metavar='save_directory',
                        default='C:/Users/Karol/Documents/DL4H/runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    parser.add_argument('--load_net', type=str, metavar='load_file_for_network', 
                        default='')
    parser.add_argument('--single_frame', action = "store_true")
    parser.add_argument('--smoke_detector', action = "store_true")
                        
    # --- Training commands
    parser.add_argument('--batch', type=int, metavar='batch_size', default=6) # TRY BATCH SIZE 6 ON 8GB AND SEE IF IT FITS!
    parser.add_argument('--lr', type=float, metavar='learning_rate', default=0.0002)

    parser.add_argument('--adam', type=float, metavar='adam_beta_params', nargs=2, default=[0.5, 0.999])
    parser.add_argument('--epochs', type=int, metavar='num_epochs', default=20)
    parser.add_argument('--gpu', type=int, metavar='gpu_index', default=0)

    return parser

def save(args):
    os.makedirs(args.save)
    with open(os.path.join(args.save, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
