# dataset download: https://academictorrents.com/collection/imagenet-2012

import os
import pandas as pd
from torchvision import datasets 
from torchvision import transforms
from torch.utils.data import DataLoader
from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
from albumentations.pytorch.transforms import ToTensor

# from datapreprocessing import  augmentation1, augmentation2
from dataset import prepare_dataset, StanfordDogs




TRAIN_IMG_DIR = './alexnet_data_in/stanford-dog-dataset/images'
IMAGE_TRAIN_DIM = 256
IMAGE_TEST_DIM = 224
BATCH_SIZE = 128

def normalise1():
    return dict(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])



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

def loader(TRAIN_IMG_DIR, data_X, data_y, is_show, idx, batch_size = BATCH_SIZE, obj = 'train'):
    # train_aug = augmentation1()
    # test_aug = augmentation2()
    train_aug = 1
    test_aug = 1
    normalise = normalise1()
    
    if is_show:
        print('is_show is True')
        stanforddogs =  StanfordDogs( train_aug, test_aug, normalise, data_X, data_y, TRAIN_IMG_DIR, obj, True)
        stanforddogs.__getitem__(idx)
        
        
    else:
        print('is_show is False')
        data = StanfordDogs(train_aug, test_aug, normalise, data_X, data_y, TRAIN_IMG_DIR, obj, False)
        print('data', type(data))
        # images, labels  = next(iter(data))
        # print('images', images.shape)
        # cv2.imshow('img_aug', images)
        # cv2.waitKey(0)
        dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle = True)
    
        return dataloader
              


