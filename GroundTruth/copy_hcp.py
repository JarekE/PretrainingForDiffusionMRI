# This file is used to safe the necessary data in my own datastructure
# Idea: Take data from HCP2019 and change (the dimension) + b-values
from os.path import join as opj
import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
import numpy as np
import shutil


src_uka = "/images/Diffusion_Imaging/pretraining_evaluation/uka"
dst_uka = "/images/Diffusion_Imaging/PretrainingAutoencoder2021/UKA"
hcp_subjects = os.listdir(src_uka)

for s in hcp_subjects:
    print(s)

    src_folder = opj(src_uka, s)
    if not (os.path.isdir(src_folder) and s.endswith("ctrl")):
        continue
    dst_folder = opj(dst_uka, s)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)


    src_bvec = opj(src_folder, "bvecs")
    dst_bvec = opj(dst_folder, "bvecs")

    src_bval = opj(src_folder,"bvals")
    dst_bval = opj(dst_folder, "bvals")

    src_dwi = opj(src_folder, "dwi.nii.gz")
    dst_dwi = opj(dst_folder, "dwi.nii.gz")

    src_mask = opj(src_folder, "brainmask.nii.gz")
    dst_mask = opj(dst_folder, "brainmask.nii.gz")

    shutil.copy(src_bvec, dst_bvec)
    shutil.copy(src_bval, dst_bval)
    shutil.copy(src_dwi, dst_dwi)
    shutil.copy(src_mask, dst_mask)
    # copy
