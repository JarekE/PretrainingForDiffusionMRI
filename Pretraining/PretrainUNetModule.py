import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification import f_beta
import math
import sys
import matplotlib.pyplot as plt
import torchmetrics


if sys.argv[1] == "server":
    import config_pretrain
elif sys.argv[1] == "pc_leon":
    from Pretraining import config_pretrain
else:
    raise Exception("unknown first argument")


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        act_fn,
    )
    return model


def up_conv(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.InstanceNorm3d(out_dim),
        act_fn,
    )
    return model


def double_conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
    )
    return model


def out_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
    )
    return model


class Unet3d(LightningModule):

    def __init__(self, in_dim, out_dim, num_filter, out_dim_pretraining, out_dim_regression, out_dim_classification, pretraining_on=False):
        super(Unet3d, self).__init__()
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.loss = nn.MSELoss()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.pretraining_on = pretraining_on
        act_fn = nn.ReLU(inplace=True)

        self.down_1 = double_conv_block(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = double_conv_block(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = double_conv_block(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = double_conv_block(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = up_conv(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = double_conv_block(self.num_filter * 12, self.num_filter * 4, act_fn)

        self.trans_2 = up_conv(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = double_conv_block(self.num_filter * 6, self.num_filter * 2, act_fn)

        self.trans_3 = up_conv(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = double_conv_block(self.num_filter * 3, self.num_filter, act_fn)

        self.out_pretraining = out_block(self.num_filter, out_dim_pretraining)
        self.out = out_block(self.num_filter, out_dim)
        self.out_regression = out_block(self.num_filter, out_dim_regression)
        self.out_classification = out_block(self.num_filter, out_dim_classification)

        self.epochs_list = []
        self.f1_list = []
        self.acc_list = []
        self.loss_list = []



    def forward(self, x):

        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_3], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        up_3 = self.up_3(concat_3)

        if self.pretraining_on == True:
            out_pretraining = self.out_pretraining(up_3)
            return out_pretraining
        else:
            out = self.out(up_3)
            return out


    def training_step(self, batch, batch_idx):

        input = batch['input']
        input = input.cuda(input.device.index)

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

        #train_loss_pre = F.l1_loss(input, output)
        train_loss_pre = F.mse_loss(input, output)
        self.train_loss = train_loss_pre
        return train_loss_pre


    def validation_step(self, batch, batch_idx):

        input = batch['input']

        input = input.cuda(input.device.index)

        output = self.forward(input)

        if 1:
            axial_middle = output.shape[2] // 2
            plt.figure('Showing the datasets')

            plt.subplot(2, 1, 1).set_axis_off()
            plt.title("Input")
            plt.imshow(input.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.subplot(2, 1, 2).set_axis_off()
            plt.title("Output")
            plt.imshow(output.cpu().detach().numpy()[0, 11, :, axial_middle, :].T, cmap='gray', origin='lower')

            plt.show()

        loss = self.mse_metric(input.cpu(), output.cpu())

        self.log('val_loss', loss)

        return loss


    def validation_epoch_end(self, validation_step_outputs):

        print("\nAktuelle Epoche:",self.current_epoch)
        print("Pretraining:",self.pretraining_on)

        self.create_logger_file_training(validation_step_outputs[0].item())



    def configure_optimizers(self):

        lr = config_pretrain.lr
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        return {'optimizer': optimizer}


    def create_logger_file_training(self, loss=0):

        self.epochs_list.append(self.current_epoch)
        self.loss_list.append([loss])