import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchio as tio
import config
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
            """
            add distortions from BA (Ranking of distortions may be relevant)
            """
            self.transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield, tio_noise])
            #self.tio_transforms = tio.Compose([tio_motion, tio_ghosting, tio_biasfield])
        else:
            self.transforms = None

    def train_dataloader(self):
        train_dataset_pre = PretrainDataset(type="pretraining", transform=self.transforms)
        dataloader = DataLoader(train_dataset_pre, batch_size=config.batch_size,
                                shuffle=False, num_workers=0)

        return dataloader

    def val_dataloader(self):
        val_dataset_pre = PretrainDataset(type="validation", transform=self.transforms)
        val_dataloader = DataLoader(val_dataset_pre, batch_size=config.batch_size,
                                    shuffle=False, num_workers=0)

        return val_dataloader
