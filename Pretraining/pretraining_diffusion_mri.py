import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything
import sys


if sys.argv[1] == "server":
    from Pretraining.UNetModule import Unet3d
    import DataLoader
    import config
elif sys.argv[1] == "pc_leon":
    from Pretraining.UNetModule import Unet3d
    from Pretraining import DataLoader
    from Pretraining import config
else:
    raise Exception("unknown first argument")


def main():

    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = Unet3d(in_dim=config.in_dim,
                   out_dim=config.out_dim,
                   num_filter=config.num_filter,
                   out_dim_pretraining=config.out_dim_pretraining,
                   pretraining_on=config.pretraining_on,
                   out_dim_regression=config.out_dim_regression,
                   out_dim_classification=config.out_dim_classification)

    dataloader = DataLoader.DataModule()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filepath=config.save_path,
                                          save_top_k=1)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.max_epochs,
                         checkpoint_callback=checkpoint_callback,
                         deterministic=True)

    trainer.fit(model, datamodule=dataloader)

if __name__ == '__main__':
    main()

