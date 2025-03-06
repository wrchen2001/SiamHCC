# SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters
Official Pytorch Implementation of "SiamHCC: a novel siamese network for quality evaluation of handwritten Chinese characters" by Weiran Chen, Guiqian Zhu, Ying Li, Yi Ji, and Chunping Liu*.

Our method is based on [Siamese Network](https://arxiv.org/abs/2011.10566), so we named our evaluation method **SiamHCC**.

## Abstract 

Automatic quality evaluation of handwritten Chinese characters aims to accurately quantify and assess handwritten Chinese characters through computer vision and machine learning technology. It is a topic that gathers significant attention from both handwriting learners and calligraphy enthusiasts. Nevertheless, most existing techniques rely mainly on traditional pre-deep learning methods. As we know, quality evaluation of handwritten Chinese characters based on the siamese network is still in the blank stage. Therefore, in this paper, we propose a novel deep convolutional siamese architecture (SiamHCC) to address this issue, which utilizes DenseNet as the backbone with a similarity-learning function. In order to pay more attention to non-local image features, our model also incorporates several self-attention blocks and Squeeze-and-Excitation (SE) blocks. Additionally, we also present a new collected dataset: Handwritten Chinese Character Evaluation (HCCE), which consists of 3,000 well-handwritten samples. By exploiting it during the model training, we achieve good results in the evaluation of various handwritten Chinese characters. Furthermore, we transfer our model to the evaluation of other eastern handwriting fonts such as Japanese (Kana) and Korean (Hangul) as well. Extensive experimental results demonstrate the effectiveness of our proposed quality evaluation model.

![](/Paper_IMG/mainmodel.png)

The model receives several style reference characters (from the target style) and content characters (from the source font) to generate style-transformed characters.

# Usage
## Dependencies









