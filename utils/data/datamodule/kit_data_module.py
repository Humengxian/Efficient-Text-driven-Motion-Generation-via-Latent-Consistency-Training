from ..humanml.scripts.motion_process import (process_file, recover_from_ric)
from ..humanml.data.dataset import Text2MotionDatasetV2
from torch.utils.data import DataLoader
import torch
import os

class KITDataModule(object):
    def __init__(self, 
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 **kwargs) -> None:
        
        self.Dataset = Text2MotionDatasetV2
        self.hparams = kwargs
        self.cfg = cfg
        self.dataloader_options = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }
        self.njoints = cfg.data.joints_num

    def train_dataloader(self):
        split = eval(f"self.cfg.train.split")
        split_file = os.path.join(eval(f"self.cfg.data.root"), eval(f"self.cfg.train.split") + ".txt")
        self.dataloader_options['batch_size'] = self.cfg.train.batchsize
        dataset = self.Dataset(split_file=split_file, split=split, **self.hparams)
        return DataLoader(
            dataset,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            **self.dataloader_options,
        )

    def test_dataloader(self):
        split = eval(f"self.cfg.test.split")
        split_file = os.path.join(eval(f"self.cfg.data.root"), eval(f"self.cfg.test.split") + ".txt")
        self.dataloader_options['batch_size'] = self.cfg.test.batchsize
        dataset = self.Dataset(split_file=split_file, split=split, **self.hparams)
        return DataLoader(
            dataset,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            **self.dataloader_options,
        )

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams['mean']).to(features)
        std = torch.tensor(self.hparams['std']).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams['mean']).to(features)
        ori_std = torch.tensor(self.hparams['std']).to(features)
        eval_mean = torch.tensor(self.hparams['mean_eval']).to(features)
        eval_std = torch.tensor(self.hparams['std_eval']).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features