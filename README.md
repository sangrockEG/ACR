# ACR
Official repository for CVPR 2023 paper: [WSSS via Adversarial Learning of Classifier and Reconstructor](https://openaccess.thecvf.com/content/CVPR2023/papers/Kweon_Weakly_Supervised_Semantic_Segmentation_via_Adversarial_Learning_of_Classifier_and_CVPR_2023_paper.pdf)  by [Hyeokjun Kweon](https://github.com/sangrockEG) and [Sung-Hoon Yoon](https://github.com/sunghoonYoon).

# Prerequisite
* Tested on Ubuntu 18.04, with Python 3.8, PyTorch 1.8.2, CUDA 11.4, both on both single and multi gpu.
* You can create conda environment with the provided yaml file.
```
conda env create -f wsss_recon.yaml
```
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
* ImageNet-pretrained weights for resnet38d are from [[resnet_38d.params]](https://drive.google.com/drive/folders/1Ak7eAs8Y8ujjv8TKIp-qCW20fgiIWTc2?usp=sharing) (From the google drive for our another paper, AEFT.)
You need to place the weights as ./pretrained/resnet_38d.params.

## Usage
> With the following code, you can generate pseudo labels to train the segmentation network.
> 
> This code includes  [AffinityNet](https://github.com/jiwoon-ahn/psa)

### Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]
```
python train.py --name [exp_name] --model recon_cvpr23
```
### Evaluation for CAM
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```
## Citation
If our code be useful for you, please consider citing our ECCV paper using the following BibTeX entry.
```
@inproceedings{kweon2023weakly,
  title={Weakly Supervised Semantic Segmentation via Adversarial Learning of Classifier and Reconstructor},
  author={Kweon, Hyeokjun and Yoon, Sung-Hoon and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11329--11339},
  year={2023}
}
```
You can also check our earlier works published on ICCV 2021 ([OC-CSE](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)) and ECCV 2022 ([AEFT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf))!

we heavily borrow the work from [AffinityNet](https://github.com/jiwoon-ahn/psa) repository. Thanks for the excellent codes!
```
## Reference
[1] J. Ahn and S. Kwak. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
