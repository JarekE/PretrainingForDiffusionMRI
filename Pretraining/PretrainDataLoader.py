import pytorch_lightning as pl
from torch.utils.data import DataLoader

import config_pretrain
import sys

if sys.argv[1] == "server":
    import PretrainDataset
elif sys.argv[1] == "pc_leon":
    from Pretraining import PretrainDataset
else:
    raise Exception("unknown first argument")


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def train_dataloader(self):

        self.train_dataset_pre = PretrainDataset.PretrainDataset(ds_type="pretraining",
                                                                 use_preprocessed=config_pretrain.preprocessed)
        self.dataloader = DataLoader(self.train_dataset_pre, batch_size=config_pretrain.batch_size,
                                     shuffle=False, num_workers=0)

        return self.dataloader


    def val_dataloader(self):

        self.val_dataset_pre = PretrainDataset.PretrainDataset(ds_type="validation",
                                                               use_preprocessed=config_pretrain.preprocessed)
        self.val_dataloader = DataLoader(self.val_dataset_pre, batch_size=config_pretrain.batch_size,
                                         shuffle=False, num_workers=0)

        return self.val_dataloader
