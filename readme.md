Directory Structure:

    +-- 01.preprocessing.ipynb
    +-- data
    |   +-- mhd
    |       +-- *.mhd
    |   +-- npy
    |       +-- *.npy
    +-- utils.py
    +-- label_mapping.py
    +-- chestCT_round1_annotation.csv

    # the npy can be obtained by the load_itk function in the former notebook called visualization


Others:

    2d/3d patch --> 2d/3d Neural Network --> loss
    The above is a complete training process, and this notebook clarifies the first step.