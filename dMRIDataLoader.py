import pytorch_lightning as pl
import rising.transforms as rtr
from rising.loading import DataLoader as rDataLoader, default_transform_call
import dMRIconfig
import dMRIDataset


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.transforms = [
            rtr.SeqToMap("input", "label")
        ]
        self.composed = rtr.Compose(self.transforms, transform_call=default_transform_call)

    def train_dataloader(self):

        self.train_dataset_pre = dMRIDataset.HCPUKADataset(ds_type="training",
                                                       use_preprocessed=dMRIconfig.preprocessed)
        self.dataloader = rDataLoader(self.train_dataset_pre, batch_size=dMRIconfig.batch_size,
                                      shuffle=False, num_workers=0, batch_transforms=self.composed)

        return self.dataloader


    def val_dataloader(self):

        self.val_dataset_pre = dMRIDataset.HCPUKADataset(ds_type="validation",
                                                     use_preprocessed=dMRIconfig.preprocessed)
        self.val_dataloader = rDataLoader(self.val_dataset_pre, batch_size=dMRIconfig.batch_size,
                           shuffle=False, num_workers=0, batch_transforms=self.composed)

        return self.val_dataloader

    def test_dataloader(self):

        self.test_dataset = dMRIDataset.HCPUKADataset(ds_type="test", use_preprocessed=dMRIconfig.preprocessed)

        return rDataLoader(self.test_dataset, batch_size=dMRIconfig.batch_size,
                                  shuffle=False, num_workers=0, batch_transforms=self.composed)
