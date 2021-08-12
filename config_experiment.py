from sys import platform
import os
import sys

'''
Argument 1:

pre
Uses the pretrained network for the calculations (best available version)

nopre
Doesn't use the pretraining
 
-------------------------------------------

Argument 2:  

out (Experiment 1)
Use for normal classification of the brain components (WM, GM, CSF, BG)

out_number (Experiment 2)
Number of fiber directions in one voxel (0-3)

out_regression (Experiment 3a)
Use for regression task with two parameters (direction of first fiber, e.g. polar coordinates)

out_classification (Experiment 3b)
Use for classification task of 8 classes (Alternative for out_regression)
Voxel in the middle, each direction-vector can be calculated as one-hot encoding vector with length 8

'''


# Changeable
n_batches = 10
patch_size = 48
batch_size = 3
n_patches = n_batches * batch_size

# Parameters of the network. For this data not changeable.
preprocessed = False
in_dim = 90
out_dim = 4
out_dim_pretraining = in_dim
out_dim_regression = 2
out_dim_classification = 8
num_filter = 16
num_classes = out_dim
lr = 0.001
only_decoder = False

username = os.path.expanduser("~").split("/")[-1]
dirpath = os.path.join('/work/scratch', username, 'PretrainingForDiffusionMRI/Results/')
filename = 'UNET-{epoch:02d}-{val_loss:.2f}'+str(sys.argv[1])+str(sys.argv[2])

pretraining = sys.argv[1]


if sys.argv[1] == "pre":
    #IMPORTANT: This is a test version, the pretraining is not scaled yet and just for showing the possibility
    checkpoint = os.path.join('/work/scratch', username, 'PretrainingForDiffusionMRI/Pretraining/checkpoints_pretraining/UNET-epoch=14-val_loss=602.89.ckpt')
    max_epochs = (200 + 100)
    learning_modus = sys.argv[2]
elif sys.argv[1] == "nopre":
    checkpoint = None
    max_epochs = 10
    learning_modus = sys.argv[2]
else:
    raise Exception("unknown first argument")



func_test = False
if func_test == True:
    max_epochs = 3


def get_image_dir_hcp():
    if platform == "linux":
        data_dir = '/work/scratch/ecke/PretrainingForDiffusionMRI/Data'
        segmentation_dir = '/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/Segmentation'
        fiberdirection_dir = '/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/FiberDirections'
    elif platform == "darwin":
        data_dir = '/Volumes/work/scratch/ecke/PretrainingForDiffusionMRI/Data'
        segmentation_dir = '/Volumes/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/Segmentation'
        fiberdirection_dir = '/Volumes/work/scratch/ecke/PretrainingForDiffusionMRI/GroundTruth/FiberDirections'
    else:
        print("platform not detected")
        raise Exception
    return data_dir, segmentation_dir, fiberdirection_dir


img_path_hcp = get_image_dir_hcp()[0]
img_path_hcp_seg = get_image_dir_hcp()[1]
img_path_hcp_fiberdir = get_image_dir_hcp()[2]

hcp_subjects = os.listdir(img_path_hcp)
hcp_subjects.sort()
hcp_seg = os.listdir(img_path_hcp_seg)
hcp_seg.sort()
hcp_fiberdir = os.listdir(img_path_hcp_fiberdir)
hcp_fiberdir.sort()

if func_test is False:
    subjects = {
        "all": [],
        "training": hcp_subjects[50:90],
        "validation": hcp_subjects[90:95],
        "test": hcp_subjects[95:100]
    }
else:
    # TESTDATA
    subjects = {
        "all": [],
        "training": hcp_subjects[0:3],
        "validation": hcp_subjects[46:47],
        "test": hcp_subjects[40:41]
    }
