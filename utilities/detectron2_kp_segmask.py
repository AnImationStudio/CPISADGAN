# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import torch
print(torch.__version__)
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

BASE_FOLDER = "../../data/"
im = cv2.imread(BASE_FOLDER+"bts.png")
# print(dir(im))
# print(im.size())
# cv2_imshow(im)

cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# # print(outputs["instances"].pred_keypoints)
# print(outputs.keys())

# for i,box in enumerate(outputs["instances"].pred_boxes):
# 	# print(box)
# 	# print(int(box[0]), int(box[2]), int(box[1]), int(box[3]))
# 	# print(im.shape)
# 	cropped_img = im[int(box[1]):int(box[3]), int(box[0]):int(box[2]),  :]
# 	# dump_img = np.zeros((cropped_img.shape[0], cropped_img.shape[0], 3))
# 	# offset = int((cropped_img.shape[0] - cropped_img.shape[1])/2)
# 	# dump_img[:,offset:offset+cropped_img.shape[1],:] = cropped_img

# 	print(cropped_img.shape)
# 	cv2.imwrite(BASE_FOLDER+"bts_people_"+str(i)+".png", cropped_img)

cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
print(outputs["instances"].pred_keypoints)



