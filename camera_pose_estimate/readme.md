
- [COLMAP]
- [LLFF](https://github.com/Fyusion/LLFF)
- [Colmap Docker](https://hub.docker.com/r/geki/colmap)



sudo nvidia-docker run --gpus all -it --shm-size=1024m  -v /local/data/nitthilan/:/nitthilan  -v ~/myfolder/:/nitthilan_myfolder geki/colmap  /bin/bash

docker run --gpus all -it --shm-size=1024m  -v /local/data/nitthilan/:/nitthilan  -v ~/myfolder/:/nitthilan_myfolder bmild/tf_colmap /bin/bash

docker exec -it 03b1292fc69e /bin/bash


'['colmap', 'feature_extractor', '--database_path', '../data/man_input/database.db', '--image_path', '../data/man_input/images', '--ImageReader.single_camera', '1']'

colmap feature_extractor --database_path ../../data/man_input/database.db --image_path ../../data/man_input_resize/ --ImageReader.single_camera 1
colmap feature_extractor --database_path ../data/man_input_resize/database.db --image_path ../data/man_input_resize/images/ --ImageReader.single_camera 1
colmap feature_extractor --database_path ../data/resize_man_input/database.db --image_path ../data/resize_man_input/images/ --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0


555x535
1600x1720


scp -r  njayakodi_dg@134.121.66.109:/local/data/nitthilan/source_code//virtual_studio/data/resize_man_input/* /Users/kannappanjayakodinitthilan/Desktop/resized_head_input/


scp -r /Users/kannappanjayakodinitthilan/Documents/myfolder/project_devan/aws_workspace/source/virtual_studio/data/oscar njayakodi_dg@134.121.66.110:/local/data/nitthilan/source_code//virtual_studio/data/



references:
- estimating camera matrix from images, estimating camera parameters from turntable multiview images, estimating camera parameters from turntable multiview images, multiview images camera parameters - https://www.di.ens.fr/willow/pdfs/ijcv08a.pdf
- bundle adjustment deep learning, colmap tutorial - https://demuc.de/papers/schoenberger2016sfm.pdf
- https://colmap.github.io/tutorial.html - simple radial camera model - what is bundle adjustment - bundle adjustment using colmap 
- https://github.com/Fyusion/LLFF/issues/8
	- https://github.com/colmap/colmap/issues/851 
	- https://colmap.github.io/faq.html
	- https://github.com/colmap/colmap/issues/373
	- 