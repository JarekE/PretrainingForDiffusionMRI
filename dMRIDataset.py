from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
import numpy as np
from random import randint
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from torch.utils.data import Dataset

import sys

import config_experiment


def transformation_and_artifacts(data, type):
    # Can be copied from BA project (correct artifacts)
    # Probably only use artifacts, and use argument to choose if it should be calculated
    ...
    return data


def create_singlePatches_wholebrain(dwi, mask, type):

    #mask = mask.astype(np.float)
    patch = np.moveaxis(dwi, 3, 0)
    del dwi

    #every site of the picture has to be --> site mod 8 = 0
    if type == "validation" or type == "test":

        patch_new_8 = patch[:, 2:-2, 1:-2, 2:-2]
        patch = patch_new_8

        # Not in use yet, will be used for at least artifacts
        #patch = transformation_and_artifacts(patch, type)

        mask_new_8 = mask[:, 2:-2, 1:-2, 2:-2]
        mask = mask_new_8

    else:
        # here one can implement extensions to extend all possible pictures to the right size
        patch = patch
        mask = mask

    patches = [patch]
    masks = [mask]

    return patches, masks


def create_singlePatches_random(dwi, brainmask, type, n_patches=config_experiment.n_patches):

    s = brainmask.shape

    patch_size = config_experiment.patch_size
    ps2 = np.int(patch_size / 2)

    patches_a = []
    masks = []

    np_dwi = np.moveaxis(dwi, 3, 0)

    # Not in use yet, will be used for at least artifacts
    #np_dwi = transformation_and_artifacts(np_dwi, type)

    n = 0
    while n<n_patches:

        x = randint(ps2,s[1]-ps2)
        y = randint(ps2,s[2]-ps2)
        z = randint(ps2,s[3]-ps2)

        # white matter should be central (brain in frame)
        if brainmask[0,x,y,z] != 1:
            continue

        if type == "training":
            mask = brainmask[:, x - ps2:x + ps2, y - ps2:y + ps2, z - ps2:z + ps2]
            patch = np_dwi[:, x - ps2:x + ps2, y - ps2:y + ps2, z - ps2:z + ps2]

        patches_a.append(patch)
        masks.append(mask.astype(np.float32))
        n = patches_a.__len__()

    return patches_a, masks

class HCPUKADataset(Dataset):
    def __init__(self, ds_type="training", use_preprocessed=False):

        self.ds_type = ds_type
        self.subject_ids = config_experiment.subjects[ds_type]

        if use_preprocessed == False:
            data = self.load_subjects(self.subject_ids)
            self.patches, self.masks_train, self.bvals, self.bvecs = self.prepare_data(data)


    def __len__(self):
        return len(self.patches)


    def __getitem__(self, idx):
        return {
            "input": self.patches[idx],
            "label": self.masks_train[idx]
        }



    def prepare_data(self, data):
        patches = []
        wm_masks = []
        bvals = []
        bvecs = []
        for subject in data.values():

            if self.ds_type == "training":
                patches_ss, wm_masks_ss = create_singlePatches_random(subject['dwi'],
                                                                      subject['bm'],
                                                                      type=self.ds_type)
            elif self.ds_type == "validation" or self.ds_type == "test":
                patches_ss, wm_masks_ss = create_singlePatches_wholebrain(subject['dwi'],
                                                                          subject['bm'],
                                                                          self.ds_type)
            else:
                raise Exception("unknown mode")
            patches.extend(patches_ss)
            wm_masks.extend(wm_masks_ss)
            bvals.extend(subject["gtab"].bvals)
            bvecs.extend(subject["gtab"].bvecs)
            del patches_ss
            del wm_masks_ss
            del subject

        return patches, wm_masks, bvals, bvecs


    def load_subjects(self, subjects=['100307']):
        data = {}
        datadir_hcp = config_experiment.img_path_hcp


        datadir_hcp_seg = config_experiment.img_path_hcp_seg
        datadir_hcp_fiberdir = config_experiment.img_path_hcp_fiberdir

        for subject in subjects:

            subdir = opj(datadir_hcp, subject)
            subdir_seg = opj(datadir_hcp_seg, subject)
            subdir_fiberdir = opj(datadir_hcp_fiberdir, subject)


            brainmask, _ = load_nifti(opj(subdir_seg, "segmentation.nii.gz"))
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
            dwi = edw

            # delete b0 values
            bvecs = np.delete(bvecs, np.where(bvals == 0), axis=0)
            dwi = np.delete(dwi, np.where(bvals == 0), axis=3)
            bvals = np.delete(bvals, np.where(bvals == 0))

            # find your mask
            if sys.argv[2] == "out":
                # round the mask
                brainmask = np.rint(brainmask)

                # 1-hot encoding of the mask
                new_csfmask = np.zeros([76, 91, 76])
                new_csfmask[:, :, :] = (brainmask == 1)

                new_greymask = np.zeros([76, 91, 76])
                new_greymask[:, :, :] = (brainmask == 2)

                new_whitemask = np.zeros([76, 91, 76])
                new_whitemask[:, :, :] = (brainmask == 3)

                new_backgroundmask = np.ones([76, 91, 76])
                new_backgroundmask[:, :, :] = (brainmask == 0)

                brainmask = np.stack((new_backgroundmask, new_whitemask, new_greymask, new_csfmask), axis=0)

            if sys.argv[2] == "out_regression" or sys.argv[2] == "out_classification":
                brainmask = np.load(opj(subdir_fiberdir, "peak_dirs.npy"))


                if sys.argv[2] == "out_classification":
                    # select only the first peak
                    brainmask = brainmask[:, :, :, 0, :] #(76,91,76,3)

                    # safe every peak in one of 8 classes (one-hot vector)
                    # ATTENTION: Not practical at the moment!
                    # Implementation misses all peaks on one axis and also a class/idea for 0-vectors!
                    # Implementation is not fast / good!
                    new_brainmask = np.zeros([76, 91, 76, 8])

                    # First Octant
                    new_brainmask[..., 0] = np.where(brainmask[..., 0] > 0, 1, new_brainmask[..., 0])
                    new_brainmask[..., 0] = np.where(brainmask[..., 1] < 0, 0, new_brainmask[..., 0])
                    new_brainmask[..., 0] = np.where(brainmask[..., 2] < 0, 0, new_brainmask[..., 0])

                    # Second Octant
                    new_brainmask[..., 1] = np.where(brainmask[..., 0] < 0, 1, new_brainmask[..., 1])
                    new_brainmask[..., 1] = np.where(brainmask[..., 1] < 0, 0, new_brainmask[..., 1])
                    new_brainmask[..., 1] = np.where(brainmask[..., 2] < 0, 0, new_brainmask[..., 1])

                    # 3. Octant
                    new_brainmask[..., 2] = np.where(brainmask[..., 0] < 0, 1, new_brainmask[..., 2])
                    new_brainmask[..., 2] = np.where(brainmask[..., 1] > 0, 0, new_brainmask[..., 2])
                    new_brainmask[..., 2] = np.where(brainmask[..., 2] < 0, 0, new_brainmask[..., 2])

                    # 4. Octant
                    new_brainmask[..., 3] = np.where(brainmask[..., 0] > 0, 1, new_brainmask[..., 3])
                    new_brainmask[..., 3] = np.where(brainmask[..., 1] > 0, 0, new_brainmask[..., 3])
                    new_brainmask[..., 3] = np.where(brainmask[..., 2] < 0, 0, new_brainmask[..., 3])

                    # 5. Octant
                    new_brainmask[..., 4] = np.where(brainmask[..., 0] > 0, 1, new_brainmask[..., 4])
                    new_brainmask[..., 4] = np.where(brainmask[..., 1] < 0, 0, new_brainmask[..., 4])
                    new_brainmask[..., 4] = np.where(brainmask[..., 2] > 0, 0, new_brainmask[..., 4])

                    # 6. Octant
                    new_brainmask[..., 5] = np.where(brainmask[..., 0] < 0, 1, new_brainmask[..., 5])
                    new_brainmask[..., 5] = np.where(brainmask[..., 1] < 0, 0, new_brainmask[..., 5])
                    new_brainmask[..., 5] = np.where(brainmask[..., 2] > 0, 0, new_brainmask[..., 5])

                    # 7. Octant
                    new_brainmask[..., 6] = np.where(brainmask[..., 0] < 0, 1, new_brainmask[..., 6])
                    new_brainmask[..., 6] = np.where(brainmask[..., 1] > 0, 0, new_brainmask[..., 6])
                    new_brainmask[..., 6] = np.where(brainmask[..., 2] > 0, 0, new_brainmask[..., 6])

                    # 8. Octant
                    new_brainmask[..., 7] = np.where(brainmask[..., 0] > 0, 1, new_brainmask[..., 7])
                    new_brainmask[..., 7] = np.where(brainmask[..., 1] > 0, 0, new_brainmask[..., 7])
                    new_brainmask[..., 7] = np.where(brainmask[..., 2] > 0, 0, new_brainmask[..., 7])

                    brainmask = np.moveaxis(new_brainmask, 3, 0)
                else:
                    # select only the first peak
                    brainmask = brainmask[:, :, :, 0, :]  # (76,91,76,3)

                    # Very basic implementation, no further/advanced calculations at the moment!
                    new_brainmask = np.zeros([76, 91, 76, 2])
                    sqrtArray = (np.power(brainmask[..., 0], 2) + np.power(brainmask[..., 1], 2) + np.power(brainmask[..., 2], 2))
                    r = np.sqrt(sqrtArray)
                    r[np.logical_and(r >= 0.99, r <= 1.01)] = 1
                    new_brainmask[..., 0] = np.arccos(brainmask[..., 2] / r)  # theta
                    new_brainmask[..., 1] = np.arctan2(brainmask[..., 1], brainmask[..., 0])  # phi

                    brainmask = np.moveaxis(new_brainmask, 3, 0)


            if sys.argv[2] == "out_number":
                brainmask = np.load(opj(subdir_fiberdir, "peak_values.npy"))
                number_of_peaks = np.count_nonzero(brainmask, axis=3)

                # 1-hot encoding of the mask
                mask0 = np.zeros([76, 91, 76])
                mask0[:, :, :] = (number_of_peaks == 0)

                mask1 = np.zeros([76, 91, 76])
                mask1[:, :, :] = (number_of_peaks == 1)

                mask2 = np.zeros([76, 91, 76])
                mask2[:, :, :] = (number_of_peaks == 2)

                mask3 = np.ones([76, 91, 76])
                mask3[:, :, :] = (number_of_peaks == 3)

                brainmask = np.stack((mask0, mask1, mask2, mask2), axis=0)


            gtab = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=150)
            gtab.bvals = np.round(gtab.bvals / 1000) * 1000


            dict = {
                'dwi': dwi.astype(np.float32),
                'gtab': gtab,
                'bm': brainmask.astype(np.float32),
            }

            print(subject)
            data[subject] = dict

        return data