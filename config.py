import os
import sys

func_test = True

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

# number of epochs for efficient training with maximum results
if sys.argv[2] == "segmentation":
    max_epochs = 1500
elif sys.argv[2] == "n_peaks":
    max_epochs = 1000
elif sys.argv[2] == "regression":
    max_epochs = 300
else:
    max_epochs = 300

if func_test == True:
    max_epochs = 10
    training_samples_per_epoch = 4

#%% Logging
username = os.path.expanduser("~").split("/")[-1]
dirpath = 'PretrainedModels'
pre_version = str(sys.argv[1])+str(sys.argv[2])
version = str(sys.argv[1])+str(sys.argv[2])+str(sys.argv[3])+str(sys.argv[4])
filenameExperiment = 'UNET-{epoch:02d}-{val_loss:.2f}'+version
checkpoint = str(sys.argv[1])

# Collect data for training and testing
log_path = 'TestData.xlsx'
log_dir = 'TensorBoard'

#%% Data settings
img_path_uka = '/images/Diffusion_Imaging/pretraining_evaluation/uka'
all_uka_subjects = os.listdir(img_path_uka)
all_uka_subjects.sort()

if func_test is False:
    if sys.argv[4] == "1":
        uka_subjects = {
            "training": all_uka_subjects[8:28],
            "validation": all_uka_subjects[4:8],
            "test": all_uka_subjects[0:4]
        }
    if sys.argv[4] == "2":
        uka_subjects = {
            "training": (all_uka_subjects[0:4] + all_uka_subjects[12:28]),
            "validation": all_uka_subjects[8:12],
            "test": all_uka_subjects[4:8]
        }
    if sys.argv[4] == "3":
        uka_subjects = {
            "training": (all_uka_subjects[0:8] + all_uka_subjects[16:28]),
            "validation": all_uka_subjects[12:16],
            "test": all_uka_subjects[8:12]
        }
    if sys.argv[4] == "4":
        uka_subjects = {
            "training": (all_uka_subjects[0:12] + all_uka_subjects[20:28]),
            "validation": all_uka_subjects[16:20],
            "test": all_uka_subjects[12:16]
        }
    if sys.argv[4] == "5":
        uka_subjects = {
            "training": (all_uka_subjects[0:16] + all_uka_subjects[24:28]),
            "validation": all_uka_subjects[20:24],
            "test": all_uka_subjects[16:20]
        }
    if sys.argv[4] == "6":
        uka_subjects = {
            "training": all_uka_subjects[0:20],
            "validation": all_uka_subjects[24:28],
            "test": all_uka_subjects[20:24]
        }
    if sys.argv[4] == "7":
        uka_subjects = {
            "training": all_uka_subjects[4:24],
            "validation": all_uka_subjects[0:4],
            "test": all_uka_subjects[24:28]
        }
else:
    uka_subjects = {
        "training": all_uka_subjects[0:3],
        "validation": all_uka_subjects[0:3],
        "test": all_uka_subjects[0:3],
    }