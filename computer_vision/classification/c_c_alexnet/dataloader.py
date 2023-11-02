# dataset download: https://academictorrents.com/collection/imagenet-2012

import os
import pandas as pd
from torchvision import datasets 
from torchvision import transforms
from torch.utils.data import DataLoader
from datapreprocessing import StanfordDogsDataset





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

def loader(TRAIN_IMG_DIR, data_X, data_y, batch_size = 128,  obj = 'train'):
    

    stanforddogs_dataset = StanfordDogsDataset(TRAIN_IMG_DIR,
                                               data_X, 
                                               data_y,
                                               obj
                                            )
    
    
    # sample = stanforddogs_dataset[65]
        
    dataloader = DataLoader(stanforddogs_dataset, batch_size , shuffle = True)

    
    sample = stanforddogs_dataset[65]
    print('sample', sample['image'].shape)
    
    return dataloader



            


