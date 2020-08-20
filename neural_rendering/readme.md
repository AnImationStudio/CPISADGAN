Neural rendering:
- CVPR tutorials - [morning](https://www.youtube.com/watch?v=LCTYRqW-ne8), [afternoon](https://www.youtube.com/watch?v=JlyGNvbGKB8 )
- Other tutorials - [2min paper](https://www.youtube.com/watch?v=nCpGStnayHk), [AI#](https://www.youtube.com/watch?v=t06qu-gXrxA)
- NeRF - [paper](https://arxiv.org/pdf/2003.08934.pdf), [project](https://www.matthewtancik.com/nerf), [pytorch](https://github.com/yenchenlin/nerf-pytorch), [pytorch fastest but seems inaccurate](https://github.com/krrish94/nerf-pytorch), [Relatively faster](https://kwea123.github.io/nerf_pl/),[Fourier encoding position](https://people.eecs.berkeley.edu/~bmild/fourfeat/)
- Sculpting in bender - [part1](https://www.youtube.com/watch?v=lfZ8HKUhxak&t=150s), [part2](https://www.youtube.com/watch?v=RTJ0ls88nMM)

3D visualization:
- [Trimesh](https://github.com/mikedh/trimesh) - visualization - reading binvox file - 
- [PyCubes](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb)
- [Pyrenderer](https://github.com/JonathanLehner/Colab-collection/blob/master/pyrender_example.ipynb)
- MeshLab - [Tutorial](http://www.cse.iitd.ac.in/~mcs112609/Meshlab%20Tutorial.pdf), 
- [Pixel2Mesh Visualization](https://github.com/nywang16/Pixel2Mesh/tree/master/GenerateData)
- [Binvox](https://github.com/dimatura/binvox-rw-py)
- [extracting mesh](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb)
- [pytorch 3d renderer](https://github.com/facebookresearch/pytorch3d/blob/master/docs/notes/renderer.md)
- [pytorch_lightining](https://github.com/PyTorchLightning/pytorch-lightning)

Extracting camera parameters:
- COLMAP - [intro1](https://demuc.de/tutorials/cvpr2017/introduction1.pdf), [intro2](https://demuc.de/tutorials/cvpr2017/introduction2.pdf), [thesis](https://www.research-collection.ethz.ch/handle/20.500.11850/295763), [thesis1](file:///Users/kannappanjayakodinitthilan/Downloads/schoenberger_phd_thesis.pdf), [tutorials](https://demuc.de/tutorials/cvpr2017/), [software](https://colmap.github.io/tutorial.html)
	- [code](https://github.com/Fyusion/LLFF/blob/889dece9635db80fc27f39322373dfe2beae9dd0/llff/poses/colmap_wrapper.py)
- Understanding camera parameters - [camera to world frame](http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf)
- How to estimate camera parameters - [RANSAC](https://my.eng.utah.edu/~cs6320/cv_files/Lecture3.pdf), [Algorithm](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/RANSAC/)

Probable animation ideas:
- DensePose and Pose Estimation
- Pose Animator - [Tensorflow.js](https://blog.tensorflow.org/2020/05/pose-animator-open-source-tool-to-bring-svg-characters-to-life.html)
- Animation basics - [3D Deep learner](https://3deeplearner.com/)
- RigNet - [project](https://zhan-xu.github.io/rig-net/), [group page](https://people.cs.umass.edu/~kalo/)
- 3D Character Animation from a Single Photo - [Photo Animation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Weng_Photo_Wake-Up_3D_Character_Animation_From_a_Single_Photo_CVPR_2019_paper.pdf)

Understanding FK and IK:
- [IK and FK basics](https://www.youtube.com/watch?v=p6PYKyxR0aY)
- [Rigging Ideas](https://3deeplearner.com/articles/)
- How bones map to skin?, how does bone movement map to 3d model - [Armature1](https://www.youtube.com/watch?v=ZmiZ6VkSJBE), [Armature](https://www.youtube.com/watch?v=cZ3o5tjO51s)
- rigging a non standard model basics - [Gorrila](https://manual.reallusion.com/3DXchange_6/ENU/Pipeline/03_Pipelines/Converting_Models_to_Non_Standard_Characters.htm)
- Skinning basics - [link1](http://home.metrocast.net/~chipartist/SkinTute/)
- Applying Mocap to any data? mapping motion capture data to rig - [humanIK](https://www.youtube.com/watch?v=eiSHnYYciec), [custom rig](https://www.youtube.com/watch?v=SYv_Z1TdBvU)

Probable markets:
- cgsociety - [forum](https://forums.cgsociety.org/), [site](https://cgsociety.org/)
- [Turbo Squid](https://www.turbosquid.com/)
- [Cube brush](https://cubebrush.co/)
- companies - [polywink](https://www.polywink.com/), [deepmotion](https://deepmotion.com/)

Possible ideas:
- [Siren](https://github.com/bmild/nerf/issues/60)
- Fourier feature let networks : Learn High Frequency Functions in Low Dimensional Domains - [paper](https://github.com/tancik/fourier-feature-networks)
- Local light field fusion - [paper](https://people.eecs.berkeley.edu/~bmild/llff/)
- Nerf Issues and imporvements - [network acceleration and generalization](https://github.com/bmild/nerf/issues/54), [closed issues](https://github.com/bmild/nerf/issues?page=2&q=is%3Aissue+is%3Aclosed), [issues](https://github.com/bmild/nerf/issues)


- Monte Carlo Geometry processing - [papar](https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/paper.pdf), Geometry processing - [reference](https://www.cs.princeton.edu/~rs/AlgsDS07/16Geometric.pdf)
Continous learning:
- Learning 

Things to do:
- Try to make it parallel training to use both the GPU