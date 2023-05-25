# UTransformer
The official implementation of the paper "Generalised Image Outpainting with UTransformer",[ArXiv](https://arxiv.org/abs/2201.11403)

While most present image outpainting conducts horizontal extrapolation, we study the generalised image outpainting problem that extrapolates visual context all-side around a given image. To this end, we develop a novel transformer-based generative adversarial network called U-Transformer able to extend image borders with plausible structure and details even for complicated scenery images. Specifically, we design a generator as an encoder-to-decoder structure embedded with the popular Swin Transformer blocks. As such, our novel framework can better cope with image long-range dependencies which are crucially important for generalised image outpainting. We propose additionally a U-shaped structure and multi-view Temporal Spatial Predictor network to reinforce image self-reconstruction as well as unknownpart prediction smoothly and realistically. We experimentally demonstrate that our proposed method could produce visually appealing results for generalized image outpainting against the state-of-the-art image outpainting approaches.

## 1. Requirements
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 10.2;
torchvision;

## 2. Data preparation

### Scenery
Scenery consists of about 6,000 images, and we randomly select 1,000 images for evaluation. The training and test dataset can be down [here](https://github.com/z-x-yang/NS-Outpainting)

### Building
In addition, we prepare a new building facades dataset in different styles consisting of diverse and complicated building architecture. There are about 16,000 images in the training set and 1,500 images in the testing set. All the images are collected on the internet. The data can be downloaded [here](https://pan.baidu.com/s/1DhRn1BFm10GlfkQLRe56AA?pwd=msdf), password:msdf


### Wikiart
Wikiart dataset contains 45,503 training images and 19,492 testing images, which can be downloaded[here](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset).

## Citation
If you find the building data is helpful for your projects, please cite us as follows.
```
@article{gao2023generalized,
  title={Generalized image outpainting with U-transformer},
  author={Gao, Penglei and Yang, Xi and Zhang, Rui and Goulermas, John Y and Geng, Yujie and Yan, Yuyao and Huang, Kaizhu},
  journal={Neural Networks},
  volume={162},
  pages={1--10},
  year={2023},
  publisher={Elsevier}
}
```
