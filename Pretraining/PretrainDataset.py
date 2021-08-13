from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
import numpy as np

import sys
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_erosion
from tqdm import tqdm

# if sys.argv[1] == "server":
#     import config_pretrain
# elif sys.argv[1] == "pc_leon":
#     from Pretraining import config_pretrain
# else:
#     raise Exception("unknown first argument")

import config_pretrain

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
        self.subject_ids = config_pretrain.subjects[type]
        self.transform = transform

        self.patches = []

        print("\n Loading data...")
        for subject in tqdm(self.subject_ids):
            subdir = opj(config_pretrain.img_path_hcp, subject)
            brainmask, _ = load_nifti(opj(subdir, "brainmask.nii.gz"))
            brainmask = binary_erosion(brainmask)

            dwi, aff = load_nifti(opj(subdir, "dwi.nii.gz"))
            dwi = dwi * np.expand_dims(brainmask, axis=-1)

            bvals, bvecs = read_bvals_bvecs(opj(subdir, "bvals"), opj(subdir, "bvecs"))
            bvals = np.around(bvals / 1000).astype(np.int) * 1000

            # scale b-values between 0 and 1
            meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
            with np.errstate(divide='ignore',invalid='ignore'):
                edw = np.divide(dwi, meanb0)
            edw[edw > 1] = 1
            edw[edw < 0] = 0
            edw[brainmask == 0] = 0
            edw[np.isnan(edw)] = 0

            edw = np.delete(edw, np.where(bvals == 0), axis=3)

            self.patches.append(crop_brain_and_switch_axis(edw.astype(np.float32)))

    def __len__(self):
        if self.type == "pretraining":
            return config_pretrain.training_samples_per_epoch
        else:
            return len(self.patches)*2

    def __getitem__(self, idx):
        idx = idx % len(self.patches) # as __len__ can return a higher number than number of patches
        patch = self.patches[idx]

        if self.transform:
            input = self.transform(patch)
        else:
            input = patch

        return {
            "input": input,
            "original": patch,
        }

