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
Doesn't use the pretraining

-------------------------------------------

Argument 2:  

segmentation (Experiment 1)
Use for normal classification of the brain components (WM, GM, CSF, BG)

n_peaks (Experiment 2)
Number of fiber directions in one voxel (0-3)

regression (Experiment 3a)
Use for regression task with two parameters (direction of first fiber, e.g. polar coordinates)

-------------------------------------------

Argument 3:

dist
Uses the pretrained network with distortions

nodist
Doesn't use distortions

Special Cases: -----------------------------

light
Uses all distortions, but in a light version

normal
Uses all distortions, but in a normal strong version

strong
Uses all distortions, but in a light version

'''


def main():
    if sys.argv[1] == "pre":
        pretrained = True
    elif sys.argv[1] == "nopre":
        pretrained = False

    learning_mode = sys.argv[2]
    if not learning_mode in ("segmentation", "n_peaks", "regression"):
        print("unkown learning mode")
        raise Exception

    dis_mode = sys.argv[3]
    if dis_mode == "nodist":
        distortions = False
    else:
        distortions = True
    if not dis_mode in ("dist", "nodist", "light", "normal", "strong"):
        print("unkown distortion mode")
        raise Exception

    torch.cuda.empty_cache() 

    # Reproducibility for every run (important to compare pretraining)
    # seed_everything(42)

    model = ExperimentModule(learning_mode=learning_mode, pretrained=pretrained, distortions=distortions)

    dataloader = ExperimentDataloader.DataModule(learning_mode=learning_mode)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=config.dirpath,
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