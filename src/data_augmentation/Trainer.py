import torch
from torch.utils.data import Dataset, Dataset, DataLoader

import os
import sys
import logging
from tqdm import tqdm
from typing import Tuple


class Trainer:
    '''
        Abstract Trainer class that helps with organizing code by containing the 3 main
        loops in training networks: train, validate, test. Users of this class can fill
        in the iteration functions.
            save_path: Path to save network parameters
            num_epoch: number of training epoch
            batch_size: batch size for training
            num_workers: number of workers to load data from datasets
            pin_memory: pin memory when loading from datasets
            run_val_and_test_every_steps: Number of training steps per validation and 
                test runs
            val_batch_size: batch size for validation
            test_batch-size: batch size for test

        Internal variables of interest when filling in functions are
            self.global_step: current total number of training datapoints used
            self.epoch_step: current epoch
            self.train_step: current number of training iterations on the current epoch
            self.validation_step: current number of validation iterations
            self.test_step: current number of test iterations
    '''
    def __init__(self, save_path:str, num_epoch:int = 10, batch_size:int = 16, 
                num_workers:int = 4, pin_memory:bool = True, run_val_and_test_every_steps:int = 500,
                val_batch_size:int = 1, test_batch_size:int = 1):
        self.save_path = save_path
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.run_val_and_test_every_steps = run_val_and_test_every_steps
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.validation_loader = None
        self.test_loader = None
        
        self.global_step = 0
        self.train_step = 0
        self.validation_step = 0
        self.test_step = 0
        self.epoch_step = 0

    def train(self, net:torch.nn.Module, train_set:Dataset, validation_set:Dataset, test_set:Dataset, *args, **kwargs):
        # Set up training loader
        train_loader = DataLoader(train_set, shuffle = True, batch_size = self.batch_size, 
                                 num_workers = self.num_workers, pin_memory = False)

        # Training loop
        net.train()
        self.global_step = 0
        try:
            for epoch in range(self.num_epoch):
                # Iterate through training dataset
                self.train_step = 0
                self.epoch_step = epoch
                with tqdm(total = len(train_set), desc=f'Epoch {epoch + 1}/{self.num_epoch}', unit='batch') as pbar:  # type: ignore
                    for itx_train, dataset_tuple in enumerate(train_loader):
                        # Run training iteration
                        self.trainIteration(net, dataset_tuple, *args, **kwargs)

                        # Clear training data
                        del dataset_tuple

                        pbar.update(self.batch_size)
                        self.train_step = itx_train
                        self.global_step = ( epoch * len(train_loader) + itx_train ) * self.batch_size

                        # Run validation and test
                        if self.train_step % self.run_val_and_test_every_steps == 0 and self.train_step != 0:
                            self.validation(net, validation_set, *args, **kwargs)
                            self.test(net, test_set, *args, **kwargs)
                            net.train()

                # Run validation and test after every epoch
                self.validation(net, validation_set, *args, **kwargs)
                self.test(net, test_set, *args, **kwargs)
                net.train()
                
                self.save_network(net, os.path.join(self.save_path, "epoch_{}.pth".format(epoch)))

        except KeyboardInterrupt:
            self.save_network(net, os.path.join(self.save_path,"interrupt.pth"))
            logging.info('Saved interrupt')
            sys.exit(0)

    def trainIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def validation(self, net:torch.nn.Module, validation_set:Dataset, *args, **kwargs):
        # Set up validation loader
        if self.validation_loader is None:
            self.validation_loader = DataLoader(validation_set, shuffle = False, batch_size = 1, 
                                     num_workers = self.num_workers, pin_memory = True)

        # Loop through validation data
        net.eval()
        self.validation_step = 0
        with tqdm(total=len(validation_set), desc='Running Validation', unit='batch') as pbar:  # type: ignore
            for itx_val, dataset_tuple in enumerate(self.validation_loader):
                # Run validation iteration
                self.validationIteration(net, dataset_tuple, *args, **kwargs)

                # Clear training data
                del dataset_tuple

                pbar.update(self.val_batch_size)
                self.validation_step = itx_val

    def validationIteration(self, net:torch.nn.Module, dataset_tuple:Tuple, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, net:torch.nn.Module, test_set:Dataset, *args, **kwargs):
        # Set up test loader
        if self.test_loader is None:
            self.test_loader = DataLoader(test_set, shuffle = False, batch_size = 1, 
                                     num_workers = self.num_workers, pin_memory = True)

        # Loop through validation data
        net.eval()
        self.test_step = 0
        with tqdm(total=len(self.test_loader), desc='Running Test', unit='batch') as pbar:
            for itx_test, dataset_tuple in enumerate(self.test_loader):
                # Run validation iteration
                self.testIteration(net, dataset_tuple, *args, **kwargs)

                # Clear training data
                del dataset_tuple
                
                pbar.update(self.test_batch_size)
                self.test_step = itx_test

    def save_network(self, net:torch.nn.Module, path:str):
        torch.save(net.state_dict(), path)

    def testIteration(self, net:torch.nn.Module, dataset_tuple:Tuple):
        raise NotImplementedError