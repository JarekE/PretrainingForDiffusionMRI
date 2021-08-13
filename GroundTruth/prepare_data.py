# This file is used to safe the necessary data in my own datastructure
# Idea: Take data from HCP2019 and change (the dimension) + b-values
from os.path import join as opj
import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
import numpy as np
from dipy.align import reslice


def data():

    dst_uka = "/images/Diffusion_Imaging/pretraining_evaluation/uka"
    src_hcp = "/images/Diffusion_Imaging/HCP2019"
    dst_hcp = "/images/Diffusion_Imaging/PretrainingAutoencoder2021/HCP"
    hcp_subjects = os.listdir(src_hcp)

    _, _, new_zooms = load_nifti(opj(dst_uka, "vp6ctrl/dwi.nii.gz"), return_voxsize=True)

    for s in hcp_subjects:

        src_folder = opj(src_hcp, s)
        if not (os.path.isdir(src_folder) and s.isdigit()):
            continue
        src_folder = opj(src_folder, "T1w/Diffusion")
        dst_folder = opj(dst_hcp, s)
        dst_folder_original = opj(dst_folder, "AllBValues")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        if not os.path.exists(dst_folder_original):
            os.makedirs(dst_folder_original)

        src_bvec = opj(src_folder, "bvecs")
        dst_bvec = opj(dst_folder, "bvecs")
        dst_bvec_original = opj(dst_folder_original, "bvecs")

        src_bval = opj(src_folder, "bvals")
        dst_bval = opj(dst_folder, "bvals")
        dst_bval_original = opj(dst_folder_original, "bvals")

        src_dwi = opj(src_folder, "data.nii.gz")
        dst_dwi = opj(dst_folder, "dwi.nii.gz")
        dst_dwi_original = opj(dst_folder_original, "dwi.nii.gz")

        src_mask = opj(src_folder, "nodif_brain_mask.nii.gz")
        dst_mask = opj(dst_folder, "brainmask.nii.gz")

        bvals, bvecs = read_bvals_bvecs(src_bval, src_bvec)
        bvals = np.around(bvals/1000).astype(int)*1000

        img, aff, zooms = load_nifti(src_dwi, return_voxsize=True)
        mask, aff_mask, zooms_mask = load_nifti(src_mask, return_voxsize=True)

        # Save the zoomed data with all bvalues for further ground truth computation
        new_img_original, new_aff_img_original = reslice.reslice(img, aff, zooms, new_zooms, order=1)
        save_nifti(dst_dwi_original, new_img_original, new_aff_img_original)
        np.savetxt(dst_bval_original, bvals)
        np.savetxt(dst_bvec_original, bvecs)

        # select only b=1000 (and b=0) images
        img = img[...,bvals<1500]
        bvecs = bvecs[bvals<1500,:]
        bvals = bvals[bvals<1500]

        # Save the zoomed data only with the reduced b-values
        new_img, new_aff_img = reslice.reslice(img, aff, zooms, new_zooms, order=1)
        new_mask, new_aff_mask = reslice.reslice(mask, aff_mask, zooms_mask, new_zooms, order=0)
        save_nifti(dst_dwi, new_img, new_aff_img)
        save_nifti(dst_mask, new_mask, new_aff_mask)
        np.savetxt(dst_bval, bvals)
        np.savetxt(dst_bvec, bvecs)
        print(s)


    # some code to check if the hcp scans look as they should
    if 0:
        from dipy.core.gradients import gradient_table
        import dipy.reconst.dti as dti
        import matplotlib.pyplot as plt

        dwi, aff = load_nifti(dst_dwi)
        mask, aff2 = load_nifti(dst_mask)
        bvals, bvecs = read_bvals_bvecs(dst_bval, dst_bvec)
        gtab = gradient_table(bvals, bvecs)

        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(dwi, mask)
        from dipy.reconst.dti import fractional_anisotropy, color_fa

        FA = fractional_anisotropy(tenfit.evals)
        for i in range(0,FA.shape[-1],10):
            plt.imshow(FA[...,i])
            plt.show()

data()