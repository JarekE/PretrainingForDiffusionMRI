import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything
import sys


if sys.argv[1] == "server":
    from Pretraining.PretrainUNetModule import Unet3d
    import PretrainDataLoader
    import config_pretrain
elif sys.argv[1] == "pc_leon":
    from Pretraining.PretrainUNetModule import Unet3d
    from Pretraining import PretrainDataLoader
    from Pretraining import config_pretrain
else:
    raise Exception("unknown first argument")


def main():

    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = Unet3d(in_dim=config_pretrain.in_dim,
                   out_dim=config_pretrain.out_dim,
                   num_filter=config_pretrain.num_filter,
                   out_dim_pretraining=config_pretrain.out_dim_pretraining,
                   pretraining_on=config_pretrain.pretraining_on,
                   out_dim_regression=config_pretrain.out_dim_regression,
                   out_dim_classification=config_pretrain.out_dim_classification)

    dataloader = PretrainDataLoader.DataModule()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=config_pretrain.dirpath,
                                          filename=config_pretrain.filename,
                                          save_top_k=1)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config_pretrain.max_epochs,
                         callbacks=[checkpoint_callback],
                         deterministic=True)

    trainer.fit(model, datamodule=dataloader)

if __name__ == '__main__':
    main()

