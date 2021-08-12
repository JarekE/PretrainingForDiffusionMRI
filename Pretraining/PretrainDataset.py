from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from os.path import join as opj
import numpy as np
from random import randint
from dipy.core.gradients import gradient_table_from_bvals_bvecs
import sys
from torch.utils.data import Dataset



if sys.argv[1] == "server":
    import config_pretrain
elif sys.argv[1] == "pc_leon":
    from Pretraining import config_pretrain
else:
    raise Exception("unknown first argument")



def create_singlePatches_wholebrain(dwi, mask, type):

    patch = np.moveaxis(dwi, 3, 0)

    #every site of the picture has to be --> site mod 8 = 0
    if type == "validation":

        patch = patch[:,2:-2,1:-2, 2:-2]

    patches = [patch]

    return patches


def create_singlePatches_random(dwi, brainmask, type, n_patches=config_pretrain.n_patches):

    s = brainmask.shape

    patch_size = config_pretrain.patch_size
    ps2 = np.int(patch_size / 2)

    patches_a = []

    np_dwi = np.moveaxis(dwi, 3, 0)

    n = 0
    while n<n_patches:

        x = randint(ps2,s[1]-ps2)
        y = randint(ps2,s[2]-ps2)
        z = randint(ps2,s[3]-ps2)

        # white matter should be central (brain in frame)
        if brainmask[0,x,y,z] != 1:
            continue

        if type == "pretraining":
            patch = np_dwi[:, x - ps2:x + ps2, y - ps2:y + ps2, z - ps2:z + ps2]

        patches_a.append(patch)
        n = patches_a.__len__()

    return patches_a


class PretrainDataset(Dataset):
    def __init__(self, ds_type="train", use_preprocessed=False):

        self.ds_type = ds_type
        self.subject_ids = config_pretrain.subjects[ds_type]

        if use_preprocessed == False:
            data = self.load_subjects(self.subject_ids)
            self.patches, self.bvals, self.bvecs = self.prepare_data(data)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return {
            "input": self.patches[idx],
        }

    def prepare_data(self, data):
        patches = []
        bvals = []
        bvecs = []
        for subject in data.values():

            if self.ds_type == "pretraining":
                patches_ss = create_singlePatches_random(subject['dwi'],
                                                                      subject['bm'],
                                                                      type=self.ds_type)
            elif self.ds_type == "validation":
                patches_ss = create_singlePatches_wholebrain(subject['dwi'],
                                                                          subject['bm'],
                                                                          self.ds_type)
            else:
                raise Exception("unknown mode")
            patches.extend(patches_ss)
            bvals.extend(subject["gtab"].bvals)
            bvecs.extend(subject["gtab"].bvecs)

        return patches, bvals, bvecs


    def load_subjects(self, subjects=['100307']):
        data = {}
        datadir_hcp = config_pretrain.img_path_hcp

        for subject in subjects:

            subdir = opj(datadir_hcp, subject)
            brainmask, _ = load_nifti(opj(subdir, "segmentation.nii.gz"))

            dwi, aff = load_nifti(opj(subdir, "dwi.nii.gz"))
            dwi = dwi * np.expand_dims(brainmask, axis=-1)

            bvals, bvecs = read_bvals_bvecs(opj(subdir, "bvals"), opj(subdir, "bvecs"))
            bvals = np.around(bvals / 1000).astype(np.int) * 1000

            # scale b-values between 0 and 1
            meanb0 = np.expand_dims(np.mean(dwi[..., bvals < 150], axis=-1), axis=-1)
            edw = np.divide(dwi, meanb0)
            edw[edw > 1] = 1
            edw[edw < 0] = 0
            edw[brainmask == 0] = 0
            edw[np.isnan(edw)] = 0
            dwi = edw

            # delete b0-values
            bvecs = np.delete(bvecs, np.where(bvals == 0), axis=0)
            dwi = np.delete(dwi, np.where(bvals == 0), axis=3)
            bvals = np.delete(bvals, np.where(bvals == 0))

            gtab = gradient_table_from_bvals_bvecs(bvals, bvecs, b0_threshold=150)

            brainmask = np.expand_dims(brainmask, axis=0)

            print(subject)

            dict = {
                'dwi': dwi.astype(np.float32),
                'gtab': gtab,
                'bm': brainmask.astype(np.int32),
            }

            data[subject] = dict

        return data