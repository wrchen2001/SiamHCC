<div align=center>

# SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters

</div>

Official Pytorch Implementation of "SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters" by Weiran Chen, Guiqian Zhu, Ying Li, Yi Ji, and Chunping Liu*.

Our method is based on [Siamese Network](https://arxiv.org/abs/2011.10566), so we named our evaluation method **SiamHCC**.

## 1. Abstract 

Automatic quality evaluation of handwritten Chinese characters aims to accurately quantify and assess handwritten Chinese characters through computer vision and machine learning technology. It is a topic that gathers significant attention from both handwriting learners and calligraphy enthusiasts. Nevertheless, most existing techniques rely mainly on traditional pre-deep learning methods. As we know, quality evaluation of handwritten Chinese characters based on the siamese network is still in the blank stage. Therefore, in this paper, we propose a novel deep convolutional siamese architecture (SiamHCC) to address this issue, which utilizes DenseNet as the backbone with a similarity-learning function. In order to pay more attention to non-local image features, our model also incorporates several self-attention blocks and Squeeze-and-Excitation (SE) blocks. Additionally, we also present a new collected dataset: Handwritten Chinese Character Evaluation (HCCE), which consists of 3,000 well-handwritten samples. By exploiting it during the model training, we achieve good results in the evaluation of various handwritten Chinese characters. Furthermore, we transfer our model to the evaluation of other eastern handwriting fonts such as Japanese (Kana) and Korean (Hangul) as well. Extensive experimental results demonstrate the effectiveness of our proposed quality evaluation model.

![](/Paper_IMG/mainmodel.png)

The model receives a target character and a corresponding template character to generate the final evaluation result.

## 2. Dependencies
>python >= 3.8  
>torch >= 1.10.0  
>torchvision >= 0.11.0  
>opencv-python >= 4.8.0.76


## 3. HCCE dataset
### 3.1. Introduction
The HCCE dataset consists of 3,000 handwritten images of 200 distinct Chinese characters. The characters, which include both simple and complex ones, were selected from the [kevindkai dataset](https://github.com/kevindkai/paper). The dataset was curated through a quality assessment process conducted by 21 individuals with professional calligraphy training. For each character, 15 high-quality images were selected based on average quality scores. In addition, the images in the dataset were converted to BMP format for consistency and easier processing. 

![](/Paper_IMG/Dataset_example.png)


### 3.2. Download

  | Dataset Name                               | Google Drive                                                              | Baidu Yun                                                             |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | HCCE | [Google Drive](https://drive.google.com/file/d/188NskMGmKBs2fjeg15PeRvn8rqmlqCNH/view?usp=drive_link) | [Baidu Yun](https://pan.baidu.com/s/13oOMwngLhHSlo7TrPHRXcw?pwd=wt5s) |


## 4. Training
Download the HCCE dataset and extract it to the project root directory. Ensure the dataset is organized in the following structure:

    SiamHCC
    ├── HCCE
    │   ├── s0
    │   │   ├── 1.bmp
    │   │   ├── 2.bmp
    │   │   ├── ...
    │   │   └── 15.bmp
    │   ├── s1
    │   ├── ...
    │   └── s199
    ├── train.py
    ├── test.py
    └── ...

Run the training script:

    python train.py

Model checkpoints will be saved to "weights/".

## 5. Test

Before testing, ensure you have: (1) A trained model weights file (e.g., SiamHCC.pkl); (2) Two images to compare (in .png or .jpg format).

To test the model with two images:

    python test.py --img1 path/to/first.png --img2 path/to/second.png --weights path/to/weights.pth


To save the comparison result as an image:

    python test.py --img1 path/to/first.png --img2 path/to/second.png --weights path/to/weights.pth --output result.png

For handwritten Chinese character recognition task, please refer to my other repository [Handwritten Chinese character recognition based on MobileNetV3](https://github.com/wrchen2001/Handwritten_Chinese_character_recognition_based_on_MobileNetV3).

## 6. License
The repository is released under the [MIT license](LICENSE).

## 7. Acknowledgment
We would like to express our sincere gratitude to our collaborators for their valuable supports throughout this project, to the creators of the HCCE dataset for providing high-quality handwritten Chinese character samples, and to the reviewers for their insightful feedback and suggestions, which greatly improved the quality of this work.







