import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
import math
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as opj
from pandas import DataFrame
from openpyxl import load_workbook
import pandas as pd
import os

import config
from UNet3d import UNet3d


class ExperimentModule(LightningModule):

    def __init__(self, learning_mode, pretrained):
        super(ExperimentModule, self).__init__()

        self.unet = UNet3d()

        if pretrained:
            self.unet.load_state_dict(torch.load(opj(config.dirpath, "pretrained_model.pt")))

        self.learning_mode = learning_mode

        if self.learning_mode == "n_peaks":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_peaks, kernel_size=1)
            self.loss = nn.MSELoss()
        elif self.learning_mode == "regression":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_regression, kernel_size=1)
            self.loss = nn.MSELoss()
        elif self.learning_mode == "segmentation":
            self.out_block = nn.Conv3d(config.num_filter, config.out_dim_segmentation, kernel_size=1)
            self.loss = nn.MSELoss()
        else:
            raise Exception("unknown learning modus")

    def forward(self, z):
        y = self.unet(z)
        return self.out_block(y)

    def training_step(self, batch, batch_idx):
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
            plt.title("Input")
            plt.imshow(input.cpu().detach().numpy()[2, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 2, 3).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 2, 4).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[2, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.show()

        train_loss = self.loss(output, target)
        self.log('Loss/Train', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        target = batch['target']
        input = batch['input']

        output = self.forward(input)

        if 0:
            if self.learning_mode == "out_number":
                axial_middle = output.shape[2] // 2
                plt.figure('Showing the datasets')

                plt.subplot(3, 1, 1).set_axis_off()
                plt.title("Input")
                plt.imshow(input.cpu().detach().numpy()[0, 10, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.subplot(3, 1, 2).set_axis_off()
                plt.title("Output")
                plt.imshow(output.cpu().detach().numpy()[0, 2, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.subplot(3, 1, 3).set_axis_off()
                plt.title("Label")
                plt.imshow(label.cpu().detach().numpy()[0, 2, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.show()
            elif self.learning_mode == "out":
                axial_middle = output.shape[2] // 2
                plt.figure('Showing the datasets')

                plt.subplot(3, 1, 1).set_axis_off()
                plt.title("Input")
                plt.imshow(input.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.subplot(3, 1, 2).set_axis_off()
                plt.title("Output")
                plt.imshow((output.cpu().detach().numpy() > 0.5).astype(int)[0, 1, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.subplot(3, 1, 3).set_axis_off()
                plt.title("Label")
                plt.imshow(label.cpu().detach().numpy()[0, 1, :, axial_middle, :].T, cmap='gray', origin='lower')

                plt.show()
            else:
                print("This output will be here in the near future!")


        val_loss = self.loss(output, target)
        self.log('Loss/Validation', val_loss)
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, batch, batch_idx):
        target = batch['target']
        input = batch['input']

        output = self.forward(input)

        val_loss = self.loss(output, target)
        self.log('Loss/Validation', val_loss)
        return val_loss

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):

        lr = config.lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return {'optimizer': optimizer}

