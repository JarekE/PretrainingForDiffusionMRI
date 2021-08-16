# Only the second half of the HCP Data, first is for pretraining!
from os.path import join as opj
import os
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
import matplotlib.pyplot as plt
from dipy.reconst.forecast import ForecastModel
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.core.gradients import gradient_table_from_bvals_bvecs

from dipy.reconst.shm import sf_to_sh, sh_to_sf
import shutil

from dipy.data import default_sphere


src = "/images/Diffusion_Imaging/pretraining_evaluation/hcp"
subjects_all = os.listdir(src)
subjects_all.sort()

bvec_list = []

src_uka_subj = "/images/Diffusion_Imaging/pretraining_evaluation/uka/vp11ctrl"
uka_bvals, uka_bvecs = read_bvals_bvecs(opj(src_uka_subj, "bvals"), opj(src_uka_subj, "bvecs"))
gtab_uka = gradient_table(uka_bvals, uka_bvecs)

def resample_sphere(dwi, gtab_orig, gtab_dest, brainmask):
    meanb0 = np.mean(dwi[..., gtab_orig.bvals < 150], axis=-1)
    meanb0[brainmask == 0] = 0

    edw = np.divide(dwi, np.expand_dims(meanb0, -1))
    edw[edw > 1] = 1
    edw[edw < 0] = 0
    edw[brainmask == 0] = 0
    edw[np.isnan(edw)] = 0

    sphere_orig = Sphere(xyz=gtab_orig.bvecs[gtab_orig.bvals == 1000])
    sh = sf_to_sh(edw[..., gtab_orig.bvals == 1000], sphere_orig, 8)

    sphere_dest = Sphere(xyz=gtab_dest.bvecs[gtab_dest.bvals == 1000])

    sf = sh_to_sf(sh, sphere_dest, 8)
    sf[sf>1]=1

    res = np.zeros((*dwi.shape[:-1], gtab_dest.bvals.shape[0]))
    res[..., 0] = meanb0
    res[..., 1:] = sf*np.expand_dims(meanb0, axis=-1)

    return res



for s in subjects_all:
    print(s)

    subject_folder = opj(src, s)
    dst_folder = opj(subject_folder, "resampled_to_uka")
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    data, aff = load_nifti(opj(subject_folder, "dwi.nii.gz"))
    brainmask, aff_brainmask = load_nifti(opj(subject_folder, "brainmask.nii.gz"))
    brainmask = brainmask>0.5
    hcp_bvals, hcp_bvecs = read_bvals_bvecs(opj(subject_folder, "bvals"), opj(subject_folder, "bvecs"))
    gtab_hcp = gradient_table(hcp_bvals, hcp_bvecs)

    resampled = resample_sphere(data, gtab_hcp, gtab_uka, brainmask)

    save_nifti(opj(dst_folder, "data_resampled.nii.gz"), resampled, aff)
    shutil.copy(opj(src_uka_subj, "bvals"), opj(dst_folder, "bvals"))
    shutil.copy(opj(src_uka_subj, "bvecs"), opj(dst_folder, "bvecs"))