from ..humanml.scripts.motion_process import (process_file, recover_from_ric)
from ..a2m.humanact12poses import HumanAct12Poses
from torch.utils.data import DataLoader
import torch
import os

class Humanact12DataModule(object):
    def __init__(self, 
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 **kwargs) -> None:
        
        self.Dataset = HumanAct12Poses
        self.hparams = kwargs
        self.cfg = cfg
        self.dataloader_options = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }
        
        self.nfeats = 150
        self.njoints = 25
        self.nclasses = 12

    def train_dataloader(self):
        # split = eval(f"self.cfg.train.split")
        # split_file = os.path.join(eval(f"self.cfg.data.root"), eval(f"self.cfg.train.split") + ".txt")
        self.dataloader_options['batch_size'] = self.cfg.train.batchsize
        dataset = self.Dataset(**self.hparams)
        return DataLoader(
            dataset,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            **self.dataloader_options,
        )

    def test_dataloader(self):
        # split = eval(f"self.cfg.test.split")
        # split_file = os.path.join(eval(f"self.cfg.data.root"), eval(f"self.cfg.test.split") + ".txt")
        self.dataloader_options['batch_size'] = self.cfg.test.batchsize
        dataset = self.Dataset(**self.hparams)
        return DataLoader(
            dataset,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            **self.dataloader_options,
        )