import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics

from UNet3d import UNet3d
import config


class PretrainAutoencoder(pl.LightningModule):

    def __init__(self):
        super(PretrainAutoencoder, self).__init__()
        self.metric = torchmetrics.MeanAbsoluteError()
        self.loss = nn.L1loss()

        self.unet = UNet3d()
        self.out_block = nn.Conv3d(config.num_filter, config.in_dim, kernel_size=1)

    def forward(self, z):
        y = self.unet(z)
        return self.out_block(y)

    def training_step(self, batch, batch_idx):
        input = batch['input']
        groundtruth = batch["original"]
        output = self.forward(input)

        loss = self.loss(output, groundtruth)

        self.log('Loss/Train', loss)
        #self.log('Loss/Train', loss, self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):

        input = batch['input']
        groundtruth = batch["original"]
        output = self.forward(input)

        loss = self.metric(input.cpu(), output.cpu())
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

        lr = config.lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return {'optimizer': optimizer}