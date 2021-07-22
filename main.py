import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything

from dMRIUNetModule import Unet3d
import dMRIDataLoader
import dMRIconfig


def main():

    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = Unet3d(in_dim=dMRIconfig.in_dim,
                   out_dim=dMRIconfig.out_dim,
                   num_filter=dMRIconfig.num_filter,
                   out_dim_pretraining=dMRIconfig.out_dim_pretraining,
                   learning_modus=dMRIconfig.learning_modus,
                   out_dim_regression=dMRIconfig.out_dim_regression,
                   out_dim_classification=dMRIconfig.out_dim_classification,
                   pretraining=dMRIconfig.pretraining)

    dataloader = dMRIDataLoader.DataModule()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          filepath=dMRIconfig.save_path,
                                          save_top_k=1)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=dMRIconfig.max_epochs,
                         checkpoint_callback=checkpoint_callback,
                         deterministic=True,
                         resume_from_checkpoint=dMRIconfig.checkpoint)

    trainer.fit(model, datamodule=dataloader)


    where_is_checkpoint = checkpoint_callback.best_model_path

    test_model = Unet3d(in_dim=dMRIconfig.in_dim,
               out_dim=dMRIconfig.out_dim,
               num_filter=dMRIconfig.num_filter,
               out_dim_pretraining=dMRIconfig.out_dim_pretraining,
               learning_modus=dMRIconfig.learning_modus,
               out_dim_regression=dMRIconfig.out_dim_regression,
               out_dim_classification=dMRIconfig.out_dim_classification,
               pretraining=dMRIconfig.pretraining)

    test_trainer = pl.Trainer(gpus=1,
                              max_epochs=dMRIconfig.max_epochs,
                              checkpoint_callback=checkpoint_callback,
                              deterministic=True,
                              resume_from_checkpoint=where_is_checkpoint)

    test_trainer.test(test_model, test_dataloaders=dataloader.test_dataloader())


if __name__ == '__main__':
    main()