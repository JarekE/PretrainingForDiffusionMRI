# Create a semantic segmentation map of the HCP Data as ground truth for experiment 1
# Only the second half. first is for pretraining!

from os.path import join as opj
import os
from shutil import copyfile


def semantic_segmentation_HCP():

    src_hcp = '/work/scratch/ecke/PretrainingForDiffusionMRI/Data'
    dst_hcp = "/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/Segmentation"
    hcp_subjects_all = os.listdir(src_hcp)
    hcp_subjects_all.sort()
    hcp_subjects = hcp_subjects_all[50:100]

    for s in hcp_subjects:

        src_folder = opj(src_hcp, s)
        if not (os.path.isdir(src_folder) and s.isdigit()):
            continue
        dst_folder = opj(dst_hcp, s)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        src_seg = opj(src_folder, "segmentation.nii.gz")
        dst_seg = opj(dst_folder, "segmentation.nii.gz")



        copyfile(src_seg, dst_seg)
        print(s)

    return