import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics
from UNet3d import UNet3d

import config_pretrain

class PretrainAutoencoder(pl.LightningModule):

    def __init__(self):
        super(PretrainAutoencoder, self).__init__()
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.loss = nn.MSELoss()

        self.model = UNet3d()

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        input = batch['input']
        groundtruth = batch["original"]
        output = self.model(input)

        if 0:
            axial_middle = output.shape[2] // 2
            plt.figure('Showing the datasets')

            plt.subplot(2, 3, 1).set_axis_off()
            plt.title("Groundtruth")
            plt.imshow(groundtruth.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 3, 2).set_axis_off()
            plt.title("Input")
            plt.imshow(input.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 3, 3).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 3, 4).set_axis_off()
            plt.title("Groundtruth")
            plt.imshow(groundtruth.cpu().detach().numpy()[1, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 3, 5).set_axis_off()
            plt.title("Input")
            plt.imshow(input.cpu().detach().numpy()[1, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 3, 6).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[1, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.show()

        loss = self.loss(output, groundtruth)

        self.log('Loss/Train', loss)
        #self.log('Loss/Train', loss, self.current_epoch)

        return loss


    def validation_step(self, batch, batch_idx):

        input = batch['input']
        groundtruth = batch["original"]
        output = self.model(input)

        loss = self.mse_metric(input.cpu(), output.cpu())
        self.log('Loss/Validation', loss)

        if batch_idx == 0:
            #%% Save example image to Tensorboard logger
            z_middle = output.shape[4] // 2
            img0 = groundtruth.cpu().detach()[0, 11, :, :, z_middle].T
            img1 = input.cpu().detach()[0, 11, :, :, z_middle].T
            img2 = output.cpu().detach()[0, 11, :, :, z_middle].T

            grid = torch.cat((img0, img1, img2), axis=1)
            grid = torch.clamp(grid, 0, 1) # tensorboard visualization cannot handle negative numbers
            self.logger.experiment.add_image("Groundtruth - Input - Output", grid, self.current_epoch, dataformats="HW")

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def configure_optimizers(self):

        lr = config_pretrain.lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return {'optimizer': optimizer}