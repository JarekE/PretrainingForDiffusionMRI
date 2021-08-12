import pytorch_lightning as pl
#import rising.transforms as rtr
#from rising.loading import default_transform_call
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import config_experiment
import dMRIDataset


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def train_dataloader(self):

        self.train_dataset_pre = dMRIDataset.HCPUKADataset(ds_type="training",
                                                           use_preprocessed=config_experiment.preprocessed)
        self.dataloader = DataLoader(self.train_dataset_pre, batch_size=config_experiment.batch_size,
                                     shuffle=False, num_workers=0)

        return self.dataloader


    def val_dataloader(self):

        self.val_dataset_pre = dMRIDataset.HCPUKADataset(ds_type="validation",
                                                         use_preprocessed=config_experiment.preprocessed)
        self.val_dataloader = DataLoader(self.val_dataset_pre, batch_size=config_experiment.batch_size,
                                         shuffle=False, num_workers=0)

        return self.val_dataloader

    def test_dataloader(self):

        self.test_dataset = dMRIDataset.HCPUKADataset(ds_type="test", use_preprocessed=config_experiment.preprocessed)

        return DataLoader(self.test_dataset, batch_size=config_experiment.batch_size,
                          shuffle=False, num_workers=0, batch_transforms=self.composed)
