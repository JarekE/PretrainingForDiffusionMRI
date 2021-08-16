import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything

from pytorch_lightning.loggers import TensorBoardLogger

# if sys.argv[1] == "server":
#     from Pretraining.PretrainUNetModule import PretrainAutoencoder
#     import PretrainDataLoader
#     import config_pretrain
# elif sys.argv[1] == "pc_leon":
#     from Pretraining.PretrainUNetModule import PretrainAutoencoder
#     from Pretraining import PretrainDataLoader
#     from Pretraining import config_pretrain
# else:
#     raise Exception("unknown first argument")
from PretrainModule import PretrainAutoencoder
import PretrainDataloader
import config_pretrain

def main():

    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = PretrainAutoencoder()

    dataloader = PretrainDataloader.PretrainDataModule(distortions=True)

    checkpoint_callback = ModelCheckpoint(monitor='Loss/Validation',
                                          dirpath=config_pretrain.dirpath,
                                          filename=config_pretrain.filename,
                                          save_top_k=1)

    logger = TensorBoardLogger(config_pretrain.log_dir, name="Pretrain", default_hp_metric=False)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config_pretrain.max_epochs,
                         callbacks=[checkpoint_callback],
                         deterministic=True,
                         logger=logger,
                         log_every_n_steps=10)

    trainer.fit(model, datamodule=dataloader)


if __name__ == '__main__':
    main()

