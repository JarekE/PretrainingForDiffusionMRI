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

Link to module.

-------------------------------------------

Argument 2:  

MUST MATCH TO LINK (ARGUMENT 1)

segmentation (Experiment 1)
Use for normal classification of the brain components (WM, GM, CSF, BG)

n_peaks (Experiment 2)
Number of fiber directions in one voxel (0-3)

regression (Experiment 3a)
Use for regression task with two parameters (direction of first fiber, e.g. polar coordinates)

-------------------------------------------

Argument 3:

No usage in the moment. Important due to the number of 3 arguments in the general application.

'''


def main():

    learning_mode = sys.argv[2]
    if not learning_mode in ("segmentation", "n_peaks", "regression"):
        print("unknown learning mode")
        raise Exception

    #forFutureUsage = sys.argv[3]

    torch.cuda.empty_cache()

    model = ExperimentModule.load_from_checkpoint(config.checkpoint, learning_mode=learning_mode, pretrained=False, distortions=False)

    dataloader = ExperimentDataloader.DataModule(learning_mode=learning_mode)

    logger = TensorBoardLogger(config.log_dir, name=os.path.join('Test', config.pre_version), default_hp_metric=False)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.max_epochs,
                         deterministic=True,
                         logger=logger,
                         log_every_n_steps=1)

    trainer.test(model, test_dataloaders=dataloader)


if __name__ == '__main__':
    main()