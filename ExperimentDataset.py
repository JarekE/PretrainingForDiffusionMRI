from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import config

def crop_brain_and_switch_axis(dwi, target):

    patch_dwi = np.moveaxis(dwi, 3, 0)

    # crop to 56, 72, 56
    ch, x,y,z = patch_dwi.shape
    x1 = x//2-28
    x2 = x//2+28
    y1 = y//2-32
    y2 = y//2+40

    patch_dwi = patch_dwi[:, x1:x2,y1:y2,:]
    patch_target = target[:, x1:x2,y1:y2,:]
    patch_dwi = np.concatenate((np.zeros((*patch_dwi.shape[:3],1)), patch_dwi, np.zeros((*patch_dwi.shape[:3],1))), axis=-1, dtype=dwi.dtype)
    patch_target = np.concatenate((np.zeros((*patch_target.shape[:3],1)), patch_target, np.zeros((*patch_target.shape[:3],1))), axis=-1, dtype=dwi.dtype)    #every site of the picture has to be --> site mod 8 = 0

    return (patch_dwi, patch_target)


class UKADataset(Dataset):
    def __init__(self, type, learning_mode):

        self.type = type
        self.learning_mode = learning_mode
        self.subject_ids = config.uka_subjects[type]

        self.patches = []

        print("\n Loading data...")
        for subject in tqdm(self.subject_ids):
            subdir = opj(config.img_path_uka, subject)
            subdir_gt = opj(subdir, "groundtruth")

            brainmask, _ = load_nifti(opj(subdir, "brainmask.nii.gz"))

            dwi, aff = load_nifti(opj(subdir, "dwi.nii.gz"))
            dwi = dwi * np.expand_dims(brainmask, axis=-1)
            bvals, bvecs = read_bvals_bvecs(opj(subdir, "bvals"), opj(subdir, "bvecs"))
            bvals = np.around(bvals / 1000).astype(np.int) * 1000

            # scale the b-values between 1 and 0 (Diffusionsabschwächung)
            meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
            edw = np.divide(dwi, meanb0)
            edw[edw > 1] = 1
            edw[edw < 0] = 0
            edw[brainmask == 0] = 0
            edw[np.isnan(edw)] = 0

            # delete b0 values
            edw = np.delete(edw, np.where(bvals == 0), axis=3)

            # find your mask
            if self.learning_mode == "segmentation":
                segmentation, _ = load_nifti(opj(subdir, "fs_seg.nii.gz"))

                # 1-hot encoding of the mask
                new_csfmask = np.zeros(segmentation.shape)
                new_csfmask[:, :, :] = (segmentation == 1)

                new_greymask = np.zeros(segmentation.shape)
                new_greymask[:, :, :] = (segmentation == 2)

                new_whitemask = np.zeros(segmentation.shape)
                new_whitemask[:, :, :] = (segmentation == 3)

                new_backgroundmask = np.ones(segmentation.shape)
                new_backgroundmask[:, :, :] = (segmentation == 0)

                target = np.stack((new_backgroundmask, new_whitemask, new_greymask, new_csfmask), axis=0)

            if self.learning_mode == "regression":
                directions = np.load(opj(subdir_gt, "peak_direction.npy"))

                # Very basic implementation, no further/advanced calculations at the moment!
                new_directions = np.zeros(directions.shape)
                sqrtArray = (np.power(directions[..., 0], 2) + np.power(directions[..., 1], 2) + np.power(directions[..., 2],
                                                                                                        2))
                r = np.sqrt(sqrtArray)
                r[np.logical_and(r >= 0.99, r <= 1.01)] = 1
                new_directions[..., 0] = np.arccos(directions[..., 2] / r)  # theta
                new_directions[..., 1] = np.arctan2(directions[..., 1], brainmask[..., 0])  # phi

                target = np.moveaxis(new_directions, 3, 0)

            if self.learning_mode == "n_peaks":
                n_peaks = np.load(opj(subdir_gt, "n_peaks.npy"))
                number_of_peaks = np.count_nonzero(n_peaks, axis=3)

                # 1-hot encoding of the mask
                mask0 = np.zeros([76, 91, 76])
                mask0[:, :, :] = (number_of_peaks == 0)

                mask1 = np.zeros([76, 91, 76])
                mask1[:, :, :] = (number_of_peaks == 1)

                mask2 = np.zeros([76, 91, 76])
                mask2[:, :, :] = (number_of_peaks == 2)

                mask3 = np.ones([76, 91, 76])
                mask3[:, :, :] = (number_of_peaks == 3)

                target = np.stack((mask0, mask1, mask2, mask2), axis=0)

            self.patches.append(crop_brain_and_switch_axis(edw.astype(np.float32), target))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        a=0
        return {
            "input": self.patches[idx][0],
            "target": self.patches[idx][1],
        }