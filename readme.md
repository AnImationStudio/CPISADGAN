Reproduction of Controllable Person Image Synthesis with Attribute-Decomposed GAN

```
@inproceedings{men2020controllable,
  title={Controllable Person Image Synthesis with Attribute-Decomposed GAN},
  author={Men, Yifang and Mao, Yiming and Jiang, Yuning and Ma, Wei-Ying and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

Reusing code for generation of input from the following repositories:
- Part segmentation is done using Pyramid Scene Parsing Network - [paper](https://arxiv.org/abs/1612.01105), [pytorch](https://github.com/hyk1996/Single-Human-Parsing-LIP), [dataset](http://sysu-hcp.net/lip/)
- Pose estimation using Pose-Transfer - [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.pdf), [pytorch](https://github.com/tengteng95/Pose-Transfer)
- Pose estimation is done through detectron2 DensePose - [paper](), 
- StyleGAN2 - [paper](https://arxiv.org/pdf/1912.04958.pdf),[tensorflow](https://github.com/NVlabs/stylegan2),[pytorch](https://github.com/lucidrains/stylegan2-pytorch)
- StyleGAN - [paper](https://arxiv.org/pdf/1812.04948.pdf)
- Controllable Person Image Synthesis with Attribute-Decomposed GAN [paper](https://menyifang.github.io/projects/ADGAN/ADGAN_files/Paper_ADGAN_CVPR2020.pdf), [code - not updated](https://github.com/menyifang/ADGAN)
- Image Harmonization - DoveNet: Deep Image Harmonization via Domain Verification - [paper](https://arxiv.org/pdf/1911.13239.pdf), [pytorch](https://github.com/bcmi/Image_Harmonization_Datasets)
- Dataset: [DeepFashion](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E)
- Intrinsic Image Popularity Assessment - [paper](https://arxiv.org/pdf/1907.01985.pdf)

Other references:
- A Style-Based Generator Architecture for Generative Adversarial Networks - [paper](https://arxiv.org/pdf/1812.04948.pdf)
- First Order Motion Model for Image Animation - [code](https://github.com/AliaksandrSiarohin/first-order-model), [paper](http://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation.pdf)
- Basic GAN pytorch implementation - [code](https://github.com/eriklindernoren/PyTorch-GAN)


List of ideas:
- Can we have a discriminator which learn progressively i.e. initially uses basic input samples to restrict the combinatorial search space. Then gradually feed finer and finer variations to adapt easily
- Can we use GAN based object generattion like small coffettis or lot of people kind of objects and encode them separately such tthat we can store the number and object and recreate them on the other side for decoders?
- Crown simulatitons for car orientation detection deep learning, car videos dataset, trees dataset