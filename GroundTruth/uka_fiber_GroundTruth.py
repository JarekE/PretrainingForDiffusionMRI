# Only the second half of the HCP Data, first is for pretraining!
from os.path import join as opj
import os
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
import matplotlib.pyplot as plt
from dipy.reconst.forecast import ForecastModel
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from dipy.data import default_sphere

def directions():

    src_uka = "/images/Diffusion_Imaging/pretraining_evaluation/uka"
    subjects_all = os.listdir(src_uka)
    subjects_all.sort()

    for s in subjects_all:
        print(s)

        subject_folder = opj(src_uka, s)
        dst_folder = opj(subject_folder, "groundtruth")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Load Data
        bvecs = opj(subject_folder, "bvecs")
        bvals = opj(subject_folder, "bvals")
        dwi = opj(subject_folder, "dwi.nii.gz")
        mask = opj(subject_folder, "brainmask.nii.gz")
        fs_seg = opj(subject_folder, "fs_seg.nii.gz")

        # My Data
        data, affine = load_nifti(dwi)
        mask, affine_mask = load_nifti(mask)
        seg, affine_mask = load_nifti(fs_seg)
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
        bvals = np.around(bvals / 1000).astype(np.int) * 1000
        gtab = gradient_table(bvals, bvecs)

        # cut wm mask for same shape as the data
        mask_wm = np.where(seg == 3, 1, 0)
        #mask_wm = np.delete(mask_wm, -1, 0)
        #mask_wm = np.delete(mask_wm, -1, 2)

        #mask = np.rint(mask)
        #mask_wm = np.where(mask == 3, 1, 0)


        response = recursive_response(gtab, data, mask=mask_wm.astype(bool), sh_order=4,
                                      peak_thr=0.01, init_fa=0.08,
                                      init_trace=0.0021, iter=8, convergence=0.001,
                                      parallel=True)

        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=4)
        #csd_fit = csd_model.fit(data_small)
        csd_peaks = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=default_sphere,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     parallel=True,
                                     npeaks=3)

        n_peaks = np.sum(csd_peaks.peak_values > 0, axis=-1)
        peak_direction = csd_peaks.peak_dirs[...,0,:]
        name_peak_direction = opj(dst_folder, "peak_direction.npy")
        name_n_peaks = opj(dst_folder, "n_peaks.npy")

        # Direction of peaks            peak_dirs (x,y,z, 5 value array, 3D-direction)
        np.save(name_peak_direction, np.float32(peak_direction))
        # Normalized value of peak      peak_values (x,y,z, 5 value array)
        # Number of peaks               to be calculated from peak_values
        np.save(name_n_peaks, np.float32(n_peaks))

    return

directions()