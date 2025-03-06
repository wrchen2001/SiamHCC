<div align=center>

# SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters

</div>

Official Pytorch Implementation of "SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters" by Weiran Chen, Guiqian Zhu, Ying Li, Yi Ji, and Chunping Liu*.

Our method is based on [Siamese Network](https://arxiv.org/abs/2011.10566), so we named our evaluation method **SiamHCC**.

## 1. Abstract 

Automatic quality evaluation of handwritten Chinese characters aims to accurately quantify and assess handwritten Chinese characters through computer vision and machine learning technology. It is a topic that gathers significant attention from both handwriting learners and calligraphy enthusiasts. Nevertheless, most existing techniques rely mainly on traditional pre-deep learning methods. As we know, quality evaluation of handwritten Chinese characters based on the siamese network is still in the blank stage. Therefore, in this paper, we propose a novel deep convolutional siamese architecture (SiamHCC) to address this issue, which utilizes DenseNet as the backbone with a similarity-learning function. In order to pay more attention to non-local image features, our model also incorporates several self-attention blocks and Squeeze-and-Excitation (SE) blocks. Additionally, we also present a new collected dataset: Handwritten Chinese Character Evaluation (HCCE), which consists of 3,000 well-handwritten samples. By exploiting it during the model training, we achieve good results in the evaluation of various handwritten Chinese characters. Furthermore, we transfer our model to the evaluation of other eastern handwriting fonts such as Japanese (Kana) and Korean (Hangul) as well. Extensive experimental results demonstrate the effectiveness of our proposed quality evaluation model.

![](/Paper_IMG/mainmodel.png)

The model receives a target character and a corresponding template character to generate the final evaluation result.

## 2. Usage
### 2.1. Dependencies
>python >= 3.8  
>torch >= 1.10.0  
>torchvision >= 0.11.0  
>opencv-python >= 4.8.0.76


## 3. HCCE dataset
### 3.1. Introduction


### 3.2. Download

  | Dataset Name                               | Google Drive                                                              | Baidu Yun                                                             |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | HCCE | [Google Drive](https://drive.google.com/file/d/188NskMGmKBs2fjeg15PeRvn8rqmlqCNH/view?usp=drive_link) | [Baidu Yun](https://pan.baidu.com/s/13oOMwngLhHSlo7TrPHRXcw?pwd=wt5s) |







