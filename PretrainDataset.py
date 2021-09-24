from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_erosion
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from PretrainDistortions import Transformation
from random import random


import config

def crop_brain_and_switch_axis(dwi):

    patch = np.moveaxis(dwi, 3, 0)

    # crop to 56, 72, 56
    ch, x,y,z = patch.shape
    x1 = x//2-28
    x2 = x//2+28
    y1 = y//2-32
    y2 = y//2+40
    patch = patch[:, x1:x2,y1:y2,x1:x2]
    #every site of the picture has to be --> site mod 8 = 0

    return patch


class PretrainDataset(Dataset):
    def __init__(self, type="train", transform=None):

        self.type = type
        self.subject_ids = config.hcp_subjects[type]
        self.transform = transform

        self.patches = []

        print("\n Loading data...")
        for subject in tqdm(self.subject_ids):
            subdir = opj(config.img_path_hcp, subject)
            brainmask, _ = load_nifti(opj(subdir, "brainmask.nii.gz"))
            brainmask = binary_erosion(brainmask)

            dwi, aff = load_nifti(opj(subdir, "resampled_to_uka", "data_resampled.nii.gz"))
            dwi = dwi * np.expand_dims(brainmask, axis=-1)

            bvals, bvecs = read_bvals_bvecs(opj(subdir, "resampled_to_uka", "bvals"), opj(subdir, "resampled_to_uka", "bvecs"))
            bvals = np.around(bvals / 1000).astype(np.int) * 1000

            # scale b-values between 0 and 1
            meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
            with np.errstate(divide='ignore',invalid='ignore'):
                edw = np.divide(dwi, meanb0)
            edw[edw > 1] = 1
            edw[edw < 0] = 0
            edw[brainmask == 0] = 0
            edw[np.isnan(edw)] = 0

            # delete b0 layer
            edw = np.delete(edw, np.where(bvals == 0), axis=3)

            self.patches.append(crop_brain_and_switch_axis(edw.astype(np.float32)))

    def __len__(self):
        if self.type == "pretraining":
            return config.training_samples_per_epoch
        else:
            return len(self.patches)*2

    def __getitem__(self, idx):
        idx = idx % len(self.patches) # as __len__ can return a higher number than number of patches
        patch = self.patches[idx]

        if self.transform:
            # 64,56,72,56 -> "perfect" data
            # Always new computed!

            input2 = np.float32(Transformation([56, 72, 56]).forward((patch)))
            input = self.transform(input2)


            if 0:
                # cmap = colors.ListedColormap(['black', 'red', 'green', 'blue'])
                cmap = 'gray'
                plt.figure("Transformations")

                plt.subplot(2, 3, 1).set_axis_off()
                plt.title("Data")
                plt.imshow(patch[30, :, :, 25].T, cmap='gray', origin='lower')

                plt.subplot(2, 3, 2).set_axis_off()
                plt.title("Data")
                plt.imshow(patch[30, :, 35, :].T, cmap='gray', origin='lower')

                plt.subplot(2, 3, 3).set_axis_off()
                plt.title("Data")
                plt.imshow(patch[30, 25, :, :].T, cmap='gray', origin='lower')

                plt.subplot(2, 3, 4).set_axis_off()
                plt.title("Transformations")
                plt.imshow(input[30, :, :, 25].T, cmap=cmap, origin='lower')

                plt.subplot(2, 3, 5).set_axis_off()
                plt.title("Transformations")
                plt.imshow(input[30, :, 35, :].T, cmap=cmap, origin='lower')

                plt.subplot(2, 3, 6).set_axis_off()
                plt.title("Transformations")
                plt.imshow(input[30, 25, :, :].T, cmap=cmap, origin='lower')

                plt.show()

        else:
            input = patch
        a=0

        return {
            "input": input,
            "original": patch,
        }

