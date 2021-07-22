import pytorch_lightning as pl
import rising.transforms as rtr
from rising.loading import DataLoader as rDataLoader, default_transform_call
import config
import sys

if sys.argv[1] == "server":
    import Dataset
elif sys.argv[1] == "pc_leon":
    from Pretraining import Dataset
else:
    raise Exception("unknown first argument")


class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.transforms = [
            rtr.SeqToMap("input", "label"),
        ]
        self.composed = rtr.Compose(self.transforms, transform_call=default_transform_call)

    def train_dataloader(self):

        self.train_dataset_pre = Dataset.HCPUKADataset(ds_type="pretraining",
                                                       use_preprocessed=config.preprocessed)
        self.dataloader = rDataLoader(self.train_dataset_pre, batch_size=config.batch_size,
                                      shuffle=False, num_workers=0, batch_transforms=self.composed)

        return self.dataloader


    def val_dataloader(self):

        self.val_dataset_pre = Dataset.HCPUKADataset(ds_type="validation",
                                                     use_preprocessed=config.preprocessed)
        self.val_dataloader = rDataLoader(self.val_dataset_pre, batch_size=config.batch_size,
                           shuffle=False, num_workers=0, batch_transforms=self.composed)

        return self.val_dataloader
