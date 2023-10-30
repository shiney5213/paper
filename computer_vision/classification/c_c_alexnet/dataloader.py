# dataset download: https://academictorrents.com/collection/imagenet-2012

import os
import pandas as pd
from torchvision import datasets 
from torchvision import transforms
from torch.utils import data
from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
from albumentations.pytorch.transforms import ToTensor


TRAIN_IMG_DIR = './alexnet_data_in/imagenet'
IMAGE_DIM = 256
# IMAGE_DIM = 224
BATCH_SIZE = 128


def dataloader(TRAIN_IMG_DIR, TRAIN_ANNO_DIR):
    
        
    transform = transforms.Compose([
        # tranforms.RandomResizedCrop(IMAGE_DIM, scale = (0.9, 1.0), ratio = (0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet Normalization method
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageNet(root = TRAIN_IMG_DIR,
                                split = 'train',
                                transforms = transform
                                )                                   

    print('dataset:', dataset.shape)

    dataloader = data.DataLoader(dataset = dataset,
                                shuffle = True,
                                pin_memory = True,
                                num_workers = 8,
                                drop_last = True,      # 남은 data 삭제
                                batch_size= BATCH_SIZE
                                )
    print('Dataloader created')
    
    return dataloader





