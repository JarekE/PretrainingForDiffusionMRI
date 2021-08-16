import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning import seed_everything

from ExperimentModule import Unet3d
import ExperimentDataloader
import config_experiment


def main():

    torch.cuda.empty_cache()

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = Unet3d(in_dim=config_experiment.in_dim,
                   out_dim=config_experiment.out_dim,
                   num_filter=config_experiment.num_filter,
                   out_dim_pretraining=config_experiment.out_dim_pretraining,
                   learning_modus=config_experiment.learning_modus,
                   out_dim_regression=config_experiment.out_dim_regression,
                   out_dim_classification=config_experiment.out_dim_classification,
                   pretraining=config_experiment.pretraining)

    dataloader = ExperimentDataloader.DataModule()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=config_experiment.dirpath,
                                          filename=config_experiment.filename,
                                          save_top_k=1)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config_experiment.max_epochs,
                         callbacks=[checkpoint_callback],
                         deterministic=True,
                         resume_from_checkpoint=config_experiment.checkpoint)

    trainer.fit(model, datamodule=dataloader)


    where_is_checkpoint = checkpoint_callback.best_model_path

    test_model = Unet3d(in_dim=config_experiment.in_dim,
                        out_dim=config_experiment.out_dim,
                        num_filter=config_experiment.num_filter,
                        out_dim_pretraining=config_experiment.out_dim_pretraining,
                        learning_modus=config_experiment.learning_modus,
                        out_dim_regression=config_experiment.out_dim_regression,
                        out_dim_classification=config_experiment.out_dim_classification,
                        pretraining=config_experiment.pretraining)

    test_trainer = pl.Trainer(gpus=1,
                              max_epochs=config_experiment.max_epochs,
                              checkpoint_callback=checkpoint_callback,
                              deterministic=True,
                              resume_from_checkpoint=where_is_checkpoint)

    test_trainer.test(test_model, test_dataloaders=dataloader.test_dataloader())


if __name__ == '__main__':
    main()