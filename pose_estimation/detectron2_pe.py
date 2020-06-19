import sys, os
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# import pickle

sys.path.append("~/detectron2_repo/")

# print(sys.path)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


im = cv2.imread("./input.jpg")

# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)

# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs['instances'][0])


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# predictor = DefaultPredictor(cfg)
# panoptic_seg, segments_info = predictor(im)["panoptic_seg"]


PATH="../../CPISADGAN/deepfashion_dataset/list_eval_partition.txt"
IMG_BASEPATH = "/nitthilan/data/DeepFashion/inshop_cloth_retrival_benchmark/img/"
OUT_BASEPATH = "/nitthilan/data/DeepFashion/inshop_cloth_retrival_benchmark/img/pose_estimation/"

with open(PATH, "r") as img_list_file:
    img_path_list = img_list_file.readlines()
print(img_path_list[:5])



cntr = 0
num_zero = 0
num_more_than_one = 0
for img_path in img_path_list[2:]:
    img_path = img_path.split(" ")[0]
    split_img_path = img_path.split("/")
    # ------------ load image ------------ #
    im = cv2.imread(os.path.join(IMG_BASEPATH, img_path))

    # --------------- inference --------------- #
    outputs = predictor(im)

    out_filename = split_img_path[1]+"_"+split_img_path[3]+"_"+split_img_path[4][:-4]+".npy"
    out_filename = os.path.join(OUT_BASEPATH, out_filename)

    if cntr%10 == 0:
        print(cntr, out_filename)
    cntr += 1
    # print(len(outputs['instances']), outputs['instances'][0])
    if(len(outputs['instances']) == 0):
    	num_zero += 1
    if(len(outputs['instances']) > 1):
    	num_more_than_one += 1


    np.savez_compressed(out_filename, outputs)
    # pickle.dump( outputs, open( "save.p", "wb" ) )

print("Num images not matching ", num_zero, num_more_than_one)