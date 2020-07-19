Reproduction of Controllable Person Image Synthesis with Attribute-Decomposed GAN

```
@inproceedings{men2020controllable,
  title={Controllable Person Image Synthesis with Attribute-Decomposed GAN},
  author={Men, Yifang and Mao, Yiming and Jiang, Yuning and Ma, Wei-Ying and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

[Repo](https://github.com/AnImationStudio?source=organization_create)

Reusing code for generation of input from the following repositories:
- Controllable Person Image Synthesis with Attribute-Decomposed GAN [paper](https://menyifang.github.io/projects/ADGAN/ADGAN_files/Paper_ADGAN_CVPR2020.pdf), [code - not updated](https://github.com/menyifang/ADGAN)
- Part segmentation is done using Pyramid Scene Parsing Network - [paper](https://arxiv.org/abs/1612.01105), [pytorch](https://github.com/hyk1996/Single-Human-Parsing-LIP), [dataset](http://sysu-hcp.net/lip/)
- Pose estimation using Pose-Transfer - [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.pdf), [pytorch](https://github.com/tengteng95/Pose-Transfer)
<!-- - Pose estimation is done through detectron2 DensePose - [paper](),  -->
- StyleGAN2 - [paper](https://arxiv.org/pdf/1912.04958.pdf),[tensorflow](https://github.com/NVlabs/stylegan2),[pytorch](https://github.com/lucidrains/stylegan2-pytorch)
- StyleGAN - [paper](https://arxiv.org/pdf/1812.04948.pdf)
- Image Harmonization - DoveNet: Deep Image Harmonization via Domain Verification - [paper](https://arxiv.org/pdf/1911.13239.pdf), [pytorch](https://github.com/bcmi/Image_Harmonization_Datasets), Image Colorization - [paper list](https://github.com/MarkMoHR/Awesome-Image-Colorization)
- Dataset: [DeepFashion](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E)
- Intrinsic Image Popularity Assessment - [paper](https://arxiv.org/pdf/1907.01985.pdf)
- SPADE - Semantic Image Synthesis with Spatially-Adaptive Normalization - [pytorch](https://github.com/NVlabs/SPADE),[project page](https://nvlabs.github.io/SPADE/), [paper](https://arxiv.org/pdf/1903.07291.pdf)
- VideoPose Estimation - [code](https://github.com/cbsudux/awesome-human-pose-estimation), [awesome papers](https://github.com/cbsudux/awesome-human-pose-estimation)
- Adversarial Latent Autoencoders - [paper](https://arxiv.org/pdf/2004.04467.pdf), [pytorch](https://github.com/podgorskiy/ALAE), 
- Depth Estimation - Consistent Depth [project](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/)
- Animating using reinforcement learning - [project](https://inventec-ai-center.github.io/projects/CARL/index.html)
- Principles of animation - [youtube](https://www.youtube.com/watch?v=uDqjIdI4bF4)

- Lighting and illumination 
	- DeepLight [youtube](https://www.youtube.com/watch?v=WCuvE97k_HI), [paper](https://augmentedperception.github.io/deeplight/content/DeepLight_Paper.pdf), [project](https://augmentedperception.github.io/deeplight/)
	- HDRI explained [youtube](https://www.youtube.com/watch?v=uzMDHTCEC-k) [details](https://vrender.com/what-is-hdri/)
	- Animation - [Cinematic lighting](https://discover.therookies.co/2018/06/24/learn-how-to-light-and-render-like-a-pixar-artist/)

- Understanding textures maps: albedo - [youtube](https://www.youtube.com/watch?v=ZOHNRlrd1Ak)

- Correcting compositing - [paper](https://s3.amazonaws.com/disney-research-data/wp-content/uploads/2020/06/18013325/High-Resolution-Neural-Face-Swapping-for-Visual-Effects.pdf), [project](http://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/), [youtube](https://www.youtube.com/channel/UCM42XWqRoruK6bNkgbgoJMw)

Other references:
- A Style-Based Generator Architecture for Generative Adversarial Networks - [paper](https://arxiv.org/pdf/1812.04948.pdf)
- First Order Motion Model for Image Animation - [code](https://github.com/AliaksandrSiarohin/first-order-model), [paper](http://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation.pdf)
- Basic GAN pytorch implementation - [code](https://github.com/eriklindernoren/PyTorch-GAN)
- VFX experiments [blog](http://neuralvfx.com/augmented-reality/ar-facial-pose-estimation/)

- Rigging - RigNet [paper](https://arxiv.org/pdf/2005.00559.pdf), [youtube](https://www.youtube.com/watch?v=J90VETgWIDg), [project](https://zhan-xu.github.io/rig-net/), [code-not available](https://github.com/zhan-xu/RigNet), [dattaset](https://umass.app.box.com/s/448zm5iw1ewbq4l2kdll6q99v5y3q4pw) - Predicting Animation Skeletons for 3D Articulated Models via Volumetric Nets [paper](https://arxiv.org/pdf/1908.08506.pdf), [supplement](https://people.cs.umass.edu/~zhanxu/papers/AnimSkelVolNet_supp.pdf), [code](https://github.com/zhan-xu/AnimSkelVolNet)

- what is skin weights in animation - NeuroSkinning - [paper](http://www.youyizheng.net/docs/neuroskinning-final-opt.pdf)
	- autorigging deeplearning
- how is skeleton mapped to required motion
- 3d walk cycle - how is skeleton mapped to required motion

- High resolution 3d human digitization - [code](https://github.com/facebookresearch/pifuhd), [project](https://shunsukesaito.github.io/PIFuHD/)

Companies:  mixamo, https://www.gleechi.com/

catastrophic forgetting deep learning
- making learning rate small for already learnt network
- Having old and new networks. The old weights do not change or get update less
- Measuring Catastrophic Forgetting in Neural Networks - file:///Users/kannappanjayakodinitthilan/Downloads/16410-77483-1-PB.pdf
- Continual Learning and Catastrophic Forgetting https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf
- Continual lifelong learning with neural networks : A review - https://reader.elsevier.com/reader/sd/pii/S0893608019300231?token=0A2F8D43B5B7E04B9C4B16B24769AA0EAC2D14F7C954CF29B6DC4E4A0F6D0851031F84CB2D8B0AC6F5FB68E1C58CE9DD
- CRITICAL LEARNING PERIODS IN DEEP NETWORKS - https://openreview.net/pdf?id=BkeStsCcKQ
- Image Difficulty Curriculum for Generative Adversarial Networks (CuGAN) - https://openaccess.thecvf.com/content_WACV_2020/papers/Soviany_Image_Difficulty_Curriculum_for_Generative_Adversarial_Networks_CuGAN_WACV_2020_paper.pdf

People crowd simulation:
- generation of shadows
- steering people
- interaction between people like not overlapping with each other
- ability to walk jump fall etc..
- ability to choose gender, age, cloth color etc
- [Reference1](https://people.eecs.berkeley.edu/~gberseth/presentations/CrowdSimulation/CrowdSimulation.html)

Animation:
- https://www.nukeygara.com/



List of ideas:
- Can we have a discriminator which learn progressively i.e. initially uses basic input samples to restrict the combinatorial search space. Then gradually feed finer and finer variations to adapt easily
- Can we use GAN based object generattion like small coffettis or lot of people kind of objects and encode them separately such tthat we can store the number and object and recreate them on the other side for decoders?
- Crown simulatitons for car orientation detection deep learning, car videos dataset, trees dataset
- pytorch3d to render outputs with known input and then learning a generator to approximate the output
- SPADE based image painting - bg generation and then add humans

List of ideas 3D sculpting:
- 3d mesh models pixar from the movies extracting the object from differnt views - https://www.renderhub.com/lykomodels/left-shark, search google 3d mesh models 
- 