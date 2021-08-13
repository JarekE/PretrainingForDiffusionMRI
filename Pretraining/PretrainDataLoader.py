import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchio as tio

import config_pretrain
import sys

# if sys.argv[1] == "server":
#     from PretrainDataset import PretrainDataset
# elif sys.argv[1] == "pc_leon":
#     from Pretraining.PretrainDataset import PretrainDataset
# else:
#     raise Exception("unknown first argument")
from PretrainDataset import PretrainDataset

class PretrainDataModule(pl.LightningDataModule):

    def __init__(self, distortions):
        super().__init__()

        if distortions:
            tio_motion = tio.RandomMotion()
            tio_ghosting = tio.RandomGhosting(num_ghosts=(0, 2), intensity=(0,0.9), restore=0.02)
            # RandomGhosting>0.9 => zu starke Verzerrungen
            tio_biasfield = tio.RandomBiasField()
            tio_noise = tio.RandomNoise()
            self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])
            #self.tio_transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield])
        else:
            self.transforms = None

    def train_dataloader(self):
        train_dataset_pre = PretrainDataset(type="pretraining", transform=self.transforms)
        dataloader = DataLoader(train_dataset_pre, batch_size=config_pretrain.batch_size,
                                     shuffle=False, num_workers=6)

        return dataloader

    def val_dataloader(self):
        val_dataset_pre = PretrainDataset(type="validation", transform=self.transforms)
        val_dataloader = DataLoader(val_dataset_pre, batch_size=config_pretrain.batch_size,
                                         shuffle=False, num_workers=6)

        return val_dataloader
