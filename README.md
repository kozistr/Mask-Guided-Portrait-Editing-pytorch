# Mask-Guided-Portrait-Editing-pytorch
Unofficial Mask Guided Portrait Editing w/ cGAN implementation in pytorch.

Original Repo is [here!](https://github.com/cientgu/Mask_Guided_Portrait_Editing)

**WIP** / This implementation includes my version of tunings/tweaks :)

# Introduction
**Mask-Guided Portrait Editing** is a novel technology based on mask-guided conditional GANs, 
which can synthesize diverse, high-quality and controllable facial images from given masks. 
With the changeable input facial mask and source image, this method allows users to do high-level portrait editing.

# Architecture

## Local Components Parsing Network

I use a custom network for parsing face & hair components, which is different w/ the paper.
Left components (eyes, mouth, nose) are parsed w/ fixed coordination.

Performance about my face & hair parsing network, The network achieves ... on Figaro1K test dataset. 

More Details are [here](https://github.com/kozistr/face-hair-segmentation-keras).

## Local Embedding Sub-Network

## Mask-Guided Generative Sub-Network

## Background Fusing Sub-Network

# Usage

# Result

# Citation
```
@inproceedings{gu2019mask,
  title={Mask-Guided Portrait Editing With Conditional GANs},
  author={Gu, Shuyang and Bao, Jianmin and Yang, Hao and Chen, Dong and Wen, Fang and Yuan, Lu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3436--3445},
  year={2019}
}
```

# Author
[HyeongChan Kim](http://kozistr.tech)
