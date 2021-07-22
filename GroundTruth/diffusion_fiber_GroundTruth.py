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


def directions():

    src_hcp = "/work/scratch/ecke/PretrainingForDiffusionMRI/Data"
    dst_hcp = "/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/FiberDirections"
    hcp_subjects_all = os.listdir(src_hcp)
    hcp_subjects_all.sort()
    # For smaller cpus, the data can be split apart
    hcp_subjects = hcp_subjects_all[50:100]

    for s in hcp_subjects:

        src_under = opj(src_hcp, s)
        src_folder = opj(src_under, "AllBValues")
        if not (os.path.isdir(src_folder) and s.isdigit()):
            continue

        dst_folder = opj(dst_hcp, s)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        print("TEST")

        # Load Data
        src_bvec = opj(src_folder, "bvecs")
        src_bval = opj(src_folder, "bvals")
        src_dwi = opj(src_folder, "dwi.nii.gz")
        src_mask = opj(src_under, "segmentation.nii.gz")

        # My Data
        data, affine = load_nifti(src_dwi)
        mask, affine_mask = load_nifti(src_mask)
        bvals, bvecs = read_bvals_bvecs(src_bval, src_bvec)
        bvals = np.around(bvals / 1000).astype(np.int) * 1000
        gtab = gradient_table(bvals, bvecs)

        """
        # cut wm mask for same shape as the data
        mask_wm = np.where(mask == 3, 1, 0)
        mask_wm = np.delete(mask_wm, -1, 0)
        mask_wm = np.delete(mask_wm, -1, 2)
        """

        mask = np.rint(mask)
        mask_wm = np.where(mask == 3, 1, 0)

        # qBall and deleting bvalues 3000
        if 0:
            """
            # TEST: delete all bvalues > 2000
            sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
            data = data[..., sel_b]
            gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
    
            qball_model = shm.QballModel(gtab, 8)
            """
        # Visualisation
        if 0:
            plt.figure("Test")
            plt.subplot(2, 3, 1).set_axis_off()
            plt.title("Data")
            plt.imshow(data[:, :, 90, 10].T, cmap='gray', origin='lower')
            plt.subplot(2, 3, 2).set_axis_off()
            plt.title("Data2")
            plt.imshow(data[:, :, 80, 10].T, cmap='gray', origin='lower')
            plt.subplot(2, 3, 3).set_axis_off()
            plt.title("Data3")
            plt.imshow(data[:, :, 78, 10].T, cmap='gray', origin='lower')
            plt.subplot(2, 3, 4).set_axis_off()
            plt.title("Data4")
            plt.imshow(data[:, :, 76, 10].T, cmap='gray', origin='lower')
            plt.subplot(2, 3, 5).set_axis_off()
            plt.title("Data5")
            plt.imshow(data[:, :, 74, 10].T, cmap='gray', origin='lower')
            plt.subplot(2, 3, 6).set_axis_off()
            plt.title("Data5-Mask")
            plt.imshow(mask[:, :, 74].T, cmap='gray', origin='lower')
            plt.show()
        #mCSD
        if 0:
            """
            # BERECHNUNG
            #sphere = get_sphere('symmetric724')
    
            response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, data,
                                                                             mask_wm,
                                                                             mask_gm,
                                                                             mask_csf)
    
            ubvals = unique_bvals_tolerance(gtab.bvals)
            response_mcsd = multi_shell_fiber_response(sh_order=8,
                                                       bvals=ubvals,
                                                       wm_rf=response_wm,
                                                       gm_rf=response_gm,
                                                       csf_rf=response_csf)
    
           # mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
    
            # ODF
    
          
            # Attention: Very long computation time!
            # first rows should be : for complete fitting
            mcsd_fit = mcsd_model.fit(data[60:70, 60:70,74:75])                 # Rohdaten
    
            # numpy array, easy to use and virtualize on own PC  --> "final" data, sampled on sphere with 724 points
            # after this you have to work with 724 points
            mcsd_odf = mcsd_fit.odf(sphere)
    
            # Save as numpy file
            name = opj(dst_folder, "odf_file.npy")
            np.save(name, mcsd_odf)
            """

        sphere = get_sphere('repulsion724')
        fm = ForecastModel(gtab, sh_order=4, dec_alg='CSD')
        angle = 25
        threshold = 0.25

        # PEAKS FROM MODEL (now 1/145)
        # Good slice at 74
        mcsd_peaks = peaks_from_model(model=fm,
                                    data=data[:, :, :, :],
                                    sphere=sphere,
                                    relative_peak_threshold=threshold,
                                    min_separation_angle=angle,
                                    mask=mask_wm[:, :, :],
                                    return_odf=False,
                                    normalize_peaks=True,
                                    npeaks=3)

        name_dirs = opj(dst_folder, "peak_dirs.npy")
        name_values = opj(dst_folder, "peak_values.npy")

        # Direction of peaks            peak_dirs (x,y,z, 5 value array, 3D-direction)
        np.save(name_dirs, np.float32(mcsd_peaks.peak_dirs))
        # Normalized value of peak      peak_values (x,y,z, 5 value array)
        # Number of peaks               to be calculated from peak_values
        np.save(name_values, np.float32(mcsd_peaks.peak_values))
        print(s)


    return