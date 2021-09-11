import os
import sys

func_test = False

#%% Neural Network Parameters
in_dim = 64
out_dim = 4
out_dim_regression = 2
out_dim_segmentation = 4
out_dim_peaks = 4
num_filter = 16

#%% Training Parameters
batch_size = 3
if sys.argv[2] == "regression":
    lr= 0.001
else:
    lr = 0.001

training_samples_per_epoch = 100
max_epochs = 1000
if func_test == True:
    max_epochs = 10
    training_samples_per_epoch = 4

#%% Logging
username = os.path.expanduser("~").split("/")[-1]
dirpath = os.path.join('/work/scratch', username, 'PretrainingForDiffusionMRI/Pretraining/checkpoints_pretraining')
pre_version = str(sys.argv[1])
version = str(sys.argv[1])+str(sys.argv[2])+str(sys.argv[3])
filename = 'UNET-{epoch:02d}-{val_loss:.2f}'+pre_version
filenameExperiment = 'UNET-{epoch:02d}-{val_loss:.2f}'+version
checkpoint = dirpath + "/" + pre_version

log_dir = os.path.join('/work/scratch', username, 'tensorboard_logger/PretrainingForDiffusionMRI')

#%% Data settings

img_path_hcp = '/images/Diffusion_Imaging/pretraining_evaluation/hcp'
all_hcp_subjects = os.listdir(img_path_hcp)
all_hcp_subjects.sort()

if func_test is False:
    hcp_subjects = {
        "pretraining": all_hcp_subjects[0:46],
        "validation": all_hcp_subjects[46:50],
    }
else:
    hcp_subjects = {
        "pretraining": all_hcp_subjects[0:3],
        "validation": all_hcp_subjects[0:3],
    }

img_path_uka = '/images/Diffusion_Imaging/pretraining_evaluation/uka'
all_uka_subjects = os.listdir(img_path_uka)
all_uka_subjects.sort()

if func_test is False:
    uka_subjects = {
        "training": all_uka_subjects[0:15],
        "validation": all_uka_subjects[15:20],
        "test": all_uka_subjects[20:]
    }
else:
    uka_subjects = {
        "training": all_uka_subjects[0:3],
        "validation": all_uka_subjects[0:3],
        "test": all_uka_subjects[0:3],
    }