# Deep Learning Pretraining for Diffusion MRI

Please note that we do not share any data sets here. Pretraining was calculated with data from the HCP project (http://www.humanconnectomeproject.org/data/).
For the evaluation and further test series, dMRI data with the dimensions (90,90,54) and 64 gradient directions 
are required (the dimensions can be adjusted).

For any questions concerning the code, please feel free to open an issue.

Details on the method and implementations can be found in the paper XXX.

## Overview

Training deep learning networks is very data intensive. Especially in
fields with a very limited number of annotated datasets, such as diffusion MRI, it
is of great importance to develop approaches that can cope with a limited amount
of data. It was previously shown that transfer learning can lead to better results
and more stable training in various medical applications. However, the use of
off-the-shelf transfer learning tools in high angular resolution diffusion MRI is
not straightforward, as such 3D approaches are commonly designed for scalar
data. Here, an extension of self-supervised pretraining to diffusion MRI data
is presented, and enhanced with a modality-specific procedure, where artifacts
encountered in diffusion MRI need to be removed. We pretrained on publicly
available data from the Human Connectome Project and evaluated the success
on data from a local hospital with three modality-related experiments: segmentation
of brain microstructure, detection of fiber crossings, and regression of nerve
fiber spatial orientation. The results were compared against a setting without pretraining,
and against classical autoencoder pretraining. We find that it is possible
to achieve both improved metrics and a more stable training with the proposed
diffusion MRI specific pretraining procedure.

## Prerequisites

Depending on the task you want to check out, the following libraries may be needed:  

- Python 3.7
- Numpy 1.19
- PyTorch 1.7  
- PyTorch-Lightning 1.4 
- Dipy 1.2

The "PretrainedModels" on the HCP data can be used for the given network. Without adjustment, 
these are suitable for any dimensionality of dMRI data, but are limited to 64 gradient directions.

## Citation

If you use our work, please cite the following paper:
```tex
@inproceedings{WEN22a,
title = {Diffusion MRI Specific Pretraining by Self-Supervision on an Auxiliary Dataset},
author = {Leon Weninger and Jarek Ecke and Chuh-Hyoun Na and Kerstin Juetten and Dorit Merhof}}
year = {2022},
journal = {Bildverarbeitung fuer die Medizin (BVM)}}
```