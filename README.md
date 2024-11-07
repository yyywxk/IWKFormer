<h2 align="center">Remote Sensing Image Rectangling with Iterative Warping Kernel Self-correction Transformer</h2>
<p align="center">
    <!-- <a href="https://github.com/yyywxk/IWKFormer/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/yyywxk/IWKFormer/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/yyywxk/IWKFormer">
    </a>
    <a href="https://github.com/yyywxk/IWKFormer/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/yyywxk/IWKFormer">
    </a>
    <a href="https://github.com/yyywxk/IWKFormer/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/yyywxk/IWKFormer?color=pink">
    </a>
    <a href="https://github.com/yyywxk/IWKFormer">
        <img alt="issues" src="https://img.shields.io/github/stars/yyywxk/IWKFormer">
    </a>
    <a href="https://ieeexplore.ieee.org/document/10632108">
        <img alt="IEEE" src="https://img.shields.io/badge/IEEE-10632108-blue">
    </a>
    <a href="mailto: qiulinwei@buaa.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

<p align="center">Linwei Qiu<sup>1,2</sup>, Fengying Xie<sup>1,2</sup>, Chang Liu<sup>1,2</sup>, Ke Wang<sup>2</sup>, Xuedong Song<sup>3</sup>, Zhenwei Shi<sup>2</sup></p>
<p align="center"><sup>1</sup> Tianmushan Laboratory, Hangzhou, China</p>
<p align="center"><sup>2</sup> Department of Aerospace Information Engineering, School of Astronautics, Beihang University, Beijing, China</p>
<p align="center"><sup>3</sup> Shanghai Aerospace Control Technology Institute, Shanghai, China</p>

<details>
<summary>Fig</summary>
<img src=./assets/Fig-IWKFormer.png border=0 width=500>
</details>

## Requirements

- Packages
  
  The code was tested with Anaconda and Python 3.10.13. The Anaconda environment is:
  
  - pytorch = 2.1.1
  - torchvision = 0.16.1
  - cudatoolkit = 11.8
  - tensorboard = 2.17.0
  - tensorboardX = 2.6.2.2
  - opencv-python = 4.9.0.80
  - numpy = 1.26.4
  - pillow = 10.3.0

Install dependencies:

- For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.
- For custom dependencies:
  
  ```bash
  conda install tensorboard tensorboardx
  pip install tqdm opencv-python thop scikit-image lpips scipy
  ```
- We implement this work with Ubuntu 18.04, NVIDIA Tesla V100, and CUDA11.8.

## Datasets (AIRD)

- The details of the dataset AIRD can be found in our paper ([IEEE Xplore](https://ieeexplore.ieee.org/document/10632108)).
- You can download it at [Baidu Cloud](https://pan.baidu.com/s/1oklVqzmjfluqJdwq1R_xlw?pwd=1234) (Extraction code: 1234).
- Put data in `../dataset` folder or  configure your dataset path in the `my_path` function of  `dataloaders/__inint__.py`.
- Our codes also support the DIR-D (Deep Rectangling for Image stitching: A Learning Baseline ([paper](https://arxiv.org/abs/2203.03831))). You can download it at [Google Drive](https://drive.google.com/file/d/1KR5DtekPJin3bmQPlTGP4wbM1zFR80ak/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1aNpHwT8JIAfX_0GtsxsWyQ)(Extraction code: 1234).

## Model Training

Follow steps below to train your model:

1. Input arguments: (see full input arguments via `python train.py --help`):
2. To train IWKFormer using AIRD with one GPU:
   
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py --lr 1e-4 --dataset AIRD --epochs 200 --batch_size 4 --workers 4 --loss-type 8terms --ite_num 2 --GRID_W 16 --GRID_H 12
   ```
3. You can change the dataset from AIRD to DIR-D.

## Model Testing

1. Input arguments: (see full input arguments via `python test.py --help`):
2. Run the testing script.
   
   ```bash
   python test.py --model_path {path/to/your/checkpoint} --save_path {path/to/the/save/result}
   ```
3. Our trained model is available at [Baidu Cloud](https://pan.baidu.com/s/1f68GifSNaGWZT8D94b40oQ?pwd=1234) (Extraction code: 1234).
4. Our test results are available at [Baidu Cloud](https://pan.baidu.com/s/1vs4YdLhm_SoNrExFes3Eew?pwd=1234) (Extraction code: 1234).

## Inference

1. Input arguments: (see full input arguments via `python inference.py --help`):
2. You can use this script to obtain your own results.
   
   ```bash
   python inference.py --model_path {path/to/your/checkpoint} --save_path {path/to/the/save/result} --input_path {path/to/the/input/data}
   ```
3. Make sure to put the data files as the following structure:
   
   ```
   inference
   ├── input
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   │   ├── ...
   |
   ├── mask
   |   ├── 001.png
   │   ├── 002.png
   │   ├── 003.png
   │   ├── 004.png
   |   ├── ...
   ```

## Citation

If our work is useful for your research, please consider citing:

```tex
@article{qiu2024remote,
  title={Remote Sensing Image Rectangling with Iterative Warping Kernel Self-correction Transformer},
  author={Qiu, Linwei and Xie, Fengying and Liu, Chang and Wang, Ke and Song, Xuedong and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## Questions

Please contact [qiulinwei@buaa.edu.cn](mailto:qiulinwei@buaa.edu.cn).

## Acknowledgement

[UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)

[UDIS2](https://github.com/nie-lang/UDIS2)

[DeepRectangling](https://github.com/nie-lang/DeepRectangling)

