


import numpy as np

FILEPATH = "/nitthilan/data/DeepFashion/inshop_cloth_retrival_benchmark/img/part_seg_adgan/\
semantic_merge3/MEN/Denim/id_00000080/01_3_back.npy"

segmentation_input = np.load(FILEPATH)

print(segmentation_input.shape)

print(segmentation_input[100:156, 100:176])