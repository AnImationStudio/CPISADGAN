from PIL import Image
import sys
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pickle

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



# http://ksimek.github.io/2012/08/22/extrinsic/
# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)/0.57
    temp = param[3]*np.cos(phi)/0.57
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    # cam_mat = np.array([axisX, axisY, axisZ])
    # cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    cam_mat = cam_mat.transpose()

    # https://hackr.io/blog/numpy-matrix-multiplication
    disp = cam_pos 

    # print(axisX, axisY, axisZ, cam_mat)
    # print(cam_pos)
    # print(disp)
    cam_pose = np.hstack((cam_mat, np.array([disp]).T))
    cam_pose = np.vstack((cam_pose, np.array([0.0, 0.0, 0.0, 1.0])))
    # print(cam_pose, cam_mat, disp)

    return cam_mat, cam_pos, cam_pose



src_dir = sys.argv[1] #"../../data/head_input"
dst_dir = sys.argv[2] #"../../data/resize_man_input/"
# cam_pos_pth = sys.argv[3] #"../../data/resize_man_input/cam_pos_info.pkl"
width, height, focal = 320, 240, np.pi/3 #640, 480, np.pi/3

elevation =  45.0 #25.0
distance = 1.0
start_angle = 0.0
step_angle = 10.0 # 7.5




if not os.path.exists(os.path.join(dst_dir,"images/")):
	os.makedirs(os.path.join(dst_dir,"images/"))


files = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]
files.sort()

print(files)

cam_params_store = {"width":width, "height":height, "focal":focal}
for i, file in enumerate(files):
	angle = start_angle+i*step_angle
	param = [angle, elevation, 0, distance, 25]
	cam_mat, cam_pos, cam_pose = camera_info(param)
	cam_params_store[i] = cam_pose

	if file.endswith(".jpg"):
		input_file = os.path.join(src_dir, "turn1.jpgfda2ef38-ac3c-47d7-8ecd-b1daec8e1e50DefaultHQ-"+str(i+1)+".jpg")
		image = Image.open(input_file)
		# print(image.format)
		image = image.convert('RGB')
		#2095 1710 
		# new_image = image.crop((105, 370, 2100, 1720))
		# new_image = image.crop((1078-575, 370,  1078+575, 1720))
		# new_image = image.crop((990-800, 500,  990+800, 1820))
		new_image = image

		height = int(width*new_image.size[1]/new_image.size[0])
		new_image = new_image.resize((width, height))
		print(new_image.size)

		output_file = os.path.join(os.path.join(dst_dir,"images/"), '{0:03}'.format(i)+".jpg")
		print(input_file, output_file, width, height)			
		new_image.save(output_file)
		# print(new_image.size)


a_file = open(os.path.join(dst_dir, "cam_pos_info.pkl"), "wb")
pickle.dump(cam_params_store, a_file)
a_file.close()

