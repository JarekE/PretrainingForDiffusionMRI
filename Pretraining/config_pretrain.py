from sys import platform
import os

"""
Arguments

1. Argument
pc_leon or server
Where will the program be executed? 

"""

func_test = False
max_epochs = 15

# Changeable
n_batches = 10
patch_size = 48
batch_size = 3
n_patches = n_batches * batch_size

# Parameters of the network. For this data not changeable.
preprocessed = False
in_dim = 90
out_dim = 4
max_channels = in_dim + 1  # B0 layer will be removed afterwards
out_dim_pretraining = in_dim
out_dim_regression = 2
out_dim_classification = 8
num_filter = 16
num_classes = out_dim
lr = 0.001
only_decoder = False
pretraining_on = True


username = os.path.expanduser("~").split("/")[-1]
dirpath = os.path.join('/work/scratch', username, 'PretrainingForDiffusionMRI/Pretraining/checkpoints_pretraining')
filename = 'UNET-{epoch:02d}-{val_loss:.2f}'
checkpoint_name = "checkpoint_pretraining"


def get_image_dir_hcp():
    if platform == "linux":
        data_dir = '/work/scratch/ecke/PretrainingForDiffusionMRI/Data'
    elif platform == "darwin":
        data_dir = '/Volumes/work/scratch/ecke/PretrainingForDiffusionMRI/Data'
    else:
        print("platform not detected")
        raise Exception
    return data_dir


img_path_hcp = get_image_dir_hcp()
hcp_subjects = os.listdir(img_path_hcp)
hcp_subjects.sort()

if func_test is False:
    subjects = {
        "all": [],
        "pretraining": hcp_subjects[0:46],
        "validation": hcp_subjects[46:50],
    }
else:
    # TESTDATA
    subjects = {
        "all": [],
        "pretraining": hcp_subjects[0:4],
        "validation": hcp_subjects[0:4],
    }
