import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchio as tio
import config
from PretrainDataset import PretrainDataset
import sys


class PretrainDataModule(pl.LightningDataModule):

    def __init__(self, distortions):
        super().__init__()

        if distortions: 

            if sys.argv[2] == "all":
                tio_motion = tio.RandomMotion()
                tio_ghosting = tio.RandomGhosting(num_ghosts=(0, 2), intensity=(0,0.9), restore=0.02)
                tio_biasfield = tio.RandomBiasField()
                tio_noise = tio.RandomNoise()
                self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])
            if sys.argv[2] == "motion":
                self.transforms = tio.RandomMotion()
            if sys.argv[2] == "ghosting":
                self.transforms = tio.RandomGhosting()
            if sys.argv[2] == "bias":
                self.transforms = tio.RandomBiasField()
            if sys.argv[2] == "noise":
                self.transforms = tio.RandomNoise()
            if sys.argv[2] == "light":
                tio_motion = tio.RandomMotion(degrees=5, translation=5, num_transforms=1)
                tio_ghosting = tio.RandomGhosting(num_ghosts=(0, 1), intensity=(0,0.5), restore=0.02)
                tio_biasfield = tio.RandomBiasField(coefficients=0.3)
                tio_noise = tio.RandomNoise(std=0.1)
                self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])
            if sys.argv[2] == "normal":
                tio_motion = tio.RandomMotion()
                tio_ghosting = tio.RandomGhosting(num_ghosts=(0, 2), intensity=(0,0.9), restore=0.02)
                tio_biasfield = tio.RandomBiasField()
                tio_noise = tio.RandomNoise()
                self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])
                #self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield])
            if sys.argv[2] == "strong":
                tio_motion = tio.RandomMotion(degrees=15, translation=15, num_transforms=3)
                tio_ghosting = tio.RandomGhosting(num_ghosts=(1, 3), intensity=(0.5,0.9), restore=0.02)
                tio_biasfield = tio.RandomBiasField(coefficients=0.7)
                tio_noise = tio.RandomNoise(std=0.3)
                self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])

        else:
            self.transforms = None

    def train_dataloader(self):
        train_dataset_pre = PretrainDataset(type="pretraining", transform=self.transforms)
        dataloader = DataLoader(train_dataset_pre, batch_size=config.batch_size,
                                shuffle=False, num_workers=4)

        return dataloader

    def val_dataloader(self):
        val_dataset_pre = PretrainDataset(type="validation", transform=self.transforms)
        val_dataloader = DataLoader(val_dataset_pre, batch_size=config.batch_size,
                                    shuffle=False, num_workers=4)

        return val_dataloader
