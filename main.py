import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import os
# from pytorch_lightning import seed_everything

from ExperimentModule import ExperimentModule
import ExperimentDataloader
import config


'''
Argument 1:

pre
Uses the pretrained network for the calculations (best available version)

nopre
Doe not use the pretraining

-------------------------------------------

Argument 2:  

segmentation (Experiment 1)
Normal classification of the brain components (WM, GM, CSF, Background)

n_peaks (Experiment 2)
Number of fiber directions in one voxel (0-3)

regression (Experiment 3)
Regression task with two parameters in polar coordinates (direction of strongest fiber)

-------------------------------------------

Argument 3:

dist
The pretrained network with distortions

nodist
Does not use distortions in the pretrained network

Special Cases for Argument 3: --------------

light
All distortions, but in a light version

normal
All distortions, but in a normal strong version

strong
All distortions, but in a strong version

-------------------------------------------

Argument 4:

Number 1-7
Cross-validation (Define train, validation and test data)

'''


def main():

    # system arguments
    if sys.argv[1] == "pre":
        pretrained = True
    elif sys.argv[1] == "nopre":
        pretrained = False
    if not sys.argv[1] in ("pre", "nopre"):
        print('unknown first argument')
        raise Exception

    learning_mode = sys.argv[2]
    if not learning_mode in ("segmentation", "n_peaks", "regression"):
        print('unknown learning mode')
        raise Exception

    dis_mode = sys.argv[3]
    if dis_mode == "nodist":
        distortions = False
    else:
        distortions = True
    if not dis_mode in ("dist", "nodist", "light", "normal", "strong"):
        print('unknown distortion mode')
        raise Exception

    if not sys.argv[4] in ("1", "2", "3", "4", "5", "6", "7"):
        print('unknown cross-validation number')
        raise Exception

    torch.cuda.empty_cache() 

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = ExperimentModule(learning_mode=learning_mode, pretrained=pretrained, distortions=distortions)

    dataloader = ExperimentDataloader.DataModule(learning_mode=learning_mode)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='',
                                          filename=config.filenameExperiment,
                                          save_top_k=1)

    logger = TensorBoardLogger(config.log_dir, name=os.path.join('Train', config.version), default_hp_metric=False)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.max_epochs,
                         callbacks=[checkpoint_callback],
                         deterministic=True,
                         logger=logger,
                         log_every_n_steps=10,
                         resume_from_checkpoint=0)

    trainer.fit(model, datamodule=dataloader)


if __name__ == '__main__':
    main()