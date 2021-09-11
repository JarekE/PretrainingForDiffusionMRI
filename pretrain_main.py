import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything
from os.path import join as opj
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os

from PretrainModule import PretrainAutoencoder
import PretrainDataloader
import config

'''
Argument 1:

dist
Implements distortions in input data

nodist
Data has no distortions
'''


def main():

    if sys.argv[1] == "dist":
        distortions = True
    elif sys.argv[1] == "nodist":
        distortions = False
    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = PretrainAutoencoder()

    dataloader = PretrainDataloader.PretrainDataModule(distortions=distortions)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=config.dirpath,
                                          filename=config.filename,
                                          save_top_k=1)

    logger = TensorBoardLogger(config.log_dir, name=os.path.join('Pretrain', config.pre_version), default_hp_metric=False)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.max_epochs,
                         callbacks=[checkpoint_callback],
                         deterministic=True,
                         logger=logger,
                         log_every_n_steps=10)

    trainer.fit(model, datamodule=dataloader)

    if distortions:
        torch.save(model.unet.state_dict(), opj(config.dirpath, "pretrained_model_distortions.pt"))
    else:
        torch.save(model.unet.state_dict(), opj(config.dirpath, "pretrained_model.pt"))

if __name__ == '__main__':
    main()
