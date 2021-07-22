'''
Idea:
Safe the ground truth in my personal HCP folder
This part-program should be "one use" to create the date

Data:
L:\Diffusion_Imaging\HCP2019
Ground Truth only for second half (first half for pretraining = autoencoding)
Images: 146432 - 899885

ATTENTION:
in main() the to be executed part of the program HAS to be chosen by hand!
For safety reasons non part is selected, because the data already exists!
'''
import sys


if sys.argv[2] == "server":
    import semantic_segmentation
    import diffusion_fiber_GroundTruth
    import prepare_data
elif sys.argv[2] == "pc_leon":
    from GroundTruth import semantic_segmentation
    from GroundTruth import diffusion_fiber_GroundTruth
    from GroundTruth import prepare_data
else:
    raise Exception("unknown second argument")


def main():

    # prepare the data for the experiments and safe this in my folder structure
    if 0:
        prepare_data.data()

    # Experiment 1 - Segmentation
    if 0:
        semantic_segmentation.semantic_segmentation_HCP()

    # Experiment 2 - fiber number and directions
    if 0:
        diffusion_fiber_GroundTruth.directions()

if __name__ == '__main__':
    main()
