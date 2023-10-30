# AlexNet


### 개요
- paper :  [ImageNet Classification with Deep Convolutional
Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- insight
    1. Overfitting 감소하기 위해 data Augmentation, Dropout 사용
    2. architecture에 ReLU, LRN, Overlapping pooling 사용
    3. train 과정에서 Multiple GPU 사용

---
### code reference 
- [dansuh17/alexnet-pytorch](https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L40)
- [Stanford Dogs AlexNet Paper Implementation Pytorch](https://www.kaggle.com/code/virajbagal/stanford-dogs-alexnet-paper-implementation-pytorch)

---
### Ⅰ.  code 
<details>
<summary><span style="font-size:150%">1. Data Loading and Preparation </span> </summary>
<div markdown="1">

<summary><span style="font-size:150%">1) Dataset </span> </summary>
<div markdown="1">

- datapreprocessing.py
- dataset : [stanford_dog_dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- dog classes: 120,  total images: 20580
- 이 중 data 수가 많은 5 개의 class에 대해 학습 (총 1160개)
- <img src = "./images/1.dataset_df_head.png" width = 250>

</div>
</details>

<summary><span style="font-size:150%">2) data preprocessing </span> </summary>
<div markdown="1">

- dataloader.py
- data Augmentation 
- ① randomResizedCrop + Horizontal Reflection
- ② PCA Color Augmentation


</div>
</details>

</div>
</details>




<details>
<summary><span style="font-size:150%">2. AlexNet</span> </summary>
<div markdown="1">
- pretrained model down: [AlexNet: AlexNet Pre-trainid Model for PyTorch](https://www.kaggle.com/datasets/pytorch/alexnet)
- main.py





---



- [Stanford Dogs AlexNet Paper Implementation Pytorch](https://www.kaggle.com/code/virajbagal/stanford-dogs-alexnet-paper-implementation-pytorch)