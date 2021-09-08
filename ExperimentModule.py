import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as opj
from tabulate import tabulate
import sys

import config
from UNet3d import UNet3d


class ExperimentModule(LightningModule):

    def __init__(self, learning_mode, pretrained, distortions):
        super(ExperimentModule, self).__init__()

        self.unet = UNet3d()

        if pretrained:
            if distortions:
                self.unet.load_state_dict(torch.load(opj(config.dirpath, "pretrained_model_distortions.pt")))
            else:
                self.unet.load_state_dict(torch.load(opj(config.dirpath, "pretrained_model.pt")))

        self.learning_mode = learning_mode

        if self.learning_mode == "n_peaks":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_peaks, kernel_size=1)
            self.loss = nn.L1Loss()
        elif self.learning_mode == "regression":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_regression, kernel_size=1)
            # MAE (MSE is not working properly)
            self.loss = nn.L1Loss()
        elif self.learning_mode == "segmentation":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_segmentation, kernel_size=1)
            self.loss = nn.L1Loss()
        else:
            raise Exception("unknown learning modus")

    def forward(self, z):
        y = self.unet(z)
        return self.out_block(y)

    def training_step(self, batch, batch_idx):
        target = batch['target']
        input = batch['input']

        output = self.forward(input)

        train_loss = self.loss(output, target)
        self.log('Loss/Train', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        target = batch['target']
        input = batch['input']

        output = self.forward(input)

        if 0:
            axial_middle = output.shape[2] // 2
            plt.figure('Showing the datasets')

            plt.subplot(2, 2, 1).set_axis_off()
            plt.title("Input")
            plt.imshow(input.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 2, 2).set_axis_off()
            plt.title("Output 0.5")
            plt.imshow((output.cpu().detach().numpy() > 0.5).astype(int)[0, 1, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 2, 3).set_axis_off()
            plt.title("Target")
            plt.imshow(target.cpu().detach().numpy()[0, 1, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 2, 4).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[0, 1, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.show()

        if sys.argv[2] == "regression":
            #data: batch_idx, theta/phi, x,y,z
            data = [[1, output[0,0,32,40,25].item(), target[0,0,32,40,25].item(), output[0,1,32,40,25].item(),
                     target[0,1,32,40,25].item()],
                    [2, output[1,0,32,40,25].item(), target[1,0,32,40,25].item(), output[1,1,32,40,25].item(),
                     target[1,1,32,40,25].item()],
                    [3, output[0, 0, 5, 5, 5].item(), target[0, 0, 5, 5, 5].item(),
                     output[0, 1, 5, 5, 5].item(), target[0, 1, 5, 5, 5].item()],
                    [4, output[1, 0, 5, 5, 5].item(), target[1, 0, 5, 5, 5].item(),
                     output[1, 1, 5, 5, 5].item(), target[1, 1, 5, 5, 5].item()]]
            print("\n")
            print(tabulate(data, headers=["Number", "Output:Theta", "Target:Theta", "Output:Phi", "Target:Phi"]))
            print("\n")

        val_loss = self.loss(output, target)
        self.log('val_loss', val_loss)

        print("\n")
        print("Validation Loss:", val_loss)
        print("\n")

        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        target = batch['target']
        input = batch['input']

        output = self.forward(input)

        val_loss = self.loss(output, target)
        self.log('val_loss', val_loss)
        return val_loss

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):

        lr = config.lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return {'optimizer': optimizer}

