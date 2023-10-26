# dataset download: https://academictorrents.com/collection/imagenet-2012

import os
import pandas as pd
from torchvision import datasets 
from torchvision import transforms
from torch.utils import data


TRAIN_IMG_DIR = './alexnet_data_in/imagenet'
IMAGE_DIM = 256
BATCH_SIZE = 128



def datapreparation(TRAIN_IMG_DIR, TRAIN_ANNO_DIR):
    length_annotations = len(os.listdir(TRAIN_ANNO_DIR))
    length_image_classes = len(os.listdir(TRAIN_IMG_DIR)) 
    
    print('Length of annotations', length_annotations)
    print('Length of image classes', length_image_classes)
    
    if length_annotations == length_image_classes:
        print('Number of unique annotations matches the number of classes')
    else:
        print("Number of unique annotations doesn't the number of classes")
    
    valid = []
    
    for element in os.listdir(TRAIN_IMG_DIR):
        breed = element.split('-')[1]
        images = len(os.listdir(os.path.join(TRAIN_IMG_DIR, element)))
        valid.append((breed, images))
        
    df = pd.DataFrame(valid, columns = ['Breed', 'Number of images'])
    print('total number of images', df['Number of images'].sum())
    
    print(df.head())
    
    raise ValueError
    return df




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





