import pytorch_lightning as pl
from torch.utils.data import DataLoader

import config
from ExperimentDataset import UKADataset


class DataModule(pl.LightningDataModule):

    def __init__(self, learning_mode):
        super().__init__()
        self.learning_mode = learning_mode

    def train_dataloader(self):
        self.train_dataset_pre = UKADataset(type="training", learning_mode=self.learning_mode)
        return DataLoader(self.train_dataset_pre, batch_size=config.batch_size,
                                     shuffle=False, num_workers=0)

    def val_dataloader(self):
        self.val_dataset_pre = UKADataset(type="validation", learning_mode=self.learning_mode)
        return DataLoader(self.val_dataset_pre, batch_size=config.batch_size,
                                         shuffle=False, num_workers=0)

    def test_dataloader(self):
        self.test_dataset = UKADataset(type="test", learning_mode=self.learning_mode)
        return DataLoader(self.test_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=0)
