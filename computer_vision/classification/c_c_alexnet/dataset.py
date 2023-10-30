import os
import cv2
import pandas as pd
import numpy as np
from torchvision import datasets 
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset, DataLoader  
from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
from albumentations.pytorch.transforms import ToTensor


from datapreprocessing import PCA_color_aug


TRAIN_IMG_DIR = './alexnet_data_in/imagenet'
IMAGE_DIM = 256
BATCH_SIZE = 128



def prepare_dataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR):
    """
    select data info( label, path)
    """
    length_annotations = len(os.listdir(TRAIN_ANNO_DIR))
    length_image_classes = len(os.listdir(TRAIN_IMG_DIR)) 
    
    print('Length of annotations', length_annotations)
    print('Length of image classes', length_image_classes)
    
    if length_annotations == length_image_classes:
        print('Number of unique annotations matches the number of classes')
    else:
        print("Number of unique annotations doesn't the number of classes")
    
    
    label_num_df = create_label_num(TRAIN_IMG_DIR)
    classes = select_classes(label_num_df, 5)
    path_label_df = select_path_label(TRAIN_IMG_DIR, classes )
    
    
    return  classes, path_label_df

def create_label_num(TRAIN_IMG_DIR):
    """
    create df with label and num of classes
    """
    valid = []
    
    for element in os.listdir(TRAIN_IMG_DIR):
        breed = element.split('-')[1]
        images = len(os.listdir(os.path.join(TRAIN_IMG_DIR, element)))
        valid.append((breed, images))
                
    df = pd.DataFrame(valid, columns = ['Breeds', 'Number of images'])
    df = df.sort_values('Number of images', ascending = False)
    print('total number of images', df['Number of images'].sum())
    print(df.head())
    
    return df

def select_classes(df, class_num):
    '''
    select the 5 classes with the largest number: sebset
    '''
    classes = df.sort_values('Number of images', ascending = False)['Breeds'][:class_num].values.tolist()
    print('subset', classes)
    
    return classes


def select_path_label(TRAIN_IMG_DIR, subset ):
    """
    create df with path and label
    """
    valid = []
    for element in os.listdir(TRAIN_IMG_DIR):
        if element.split('-')[1] in subset:
            element_path = os.listdir(os.path.join(TRAIN_IMG_DIR, element))
            for img_id in element_path:
                path = os.path.join(element, img_id)
                label = element.split('_')
                valid.append((path, label))
                
    df = pd.DataFrame(valid, columns = ['Path', 'Label'])
    print('shape of dataframe:', df.shape)
    return df


class StanfordDogs(Dataset):
    def __init__(self, transform1, transform2, X, Y, image_path, objective = 'train'):
        self.X = X
        self.Y = Y
        self.train_trainform = transform1
        self.valid_transform = transform2
        self.objective = objective
        self.image_path = image_path
        self.IMAGE_TRAIN_DIM = 256
        self.IMAGE_TEST_DIM = 244
        

    def __getitem__(self, idx):
        path = self.X['Path'][idx]
        label = self.Y.iloc[idx, :].values
        img = cv2.imread(os.path.join(self.image_path, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.image_resize()
        img = self.is_train(img)
        
        return img, label
        
        
    def __len__(self):
        return len(self.X)
    
    def image_resize(self):
                # Shortest side of image is scaled to 256 pixels and the other side is scaled so as to maintain aspect ratio
        h, w, _ = img.shape
        
        if h <= w:
            aspect_ratio = w/h
            dim = (self.IMAGE_TRAIN_DIM, int(self.IMAGE_TRAIN_DIM * aspect_ratio))
            img = cv2.resize(img, dim)
        else:
            aspect = h/w
            dim = (int(self.IMAGE_TRAIN_DIM * aspect_ratio), self.IMAGE_TRAIN_DIM)
            img = cv2.resize(img, dim)
            
        img = CenterCrop(height = self.IMAGE_TRAIN_DIM, width = self.IMAGE_TRAIN_DIM, p = 1)(image = img)['image']
        
        return img
    
    def is_train(self, img):
        if self.objective == 'train':
            random = np.random.uniform(size = 1)
            if random < 0.5:
                img = PCA_color_aug(img, category = 'numpy')
            augmented = self.train_trainform(image = img)
            
        elif ((self.objective == 'validation') | (self.objective == 'test')):
            img = cv2.resize(img, (self.IMAGE_TEST_DIM, self.IMAGE_TEST_DIM))
            augmented = self.valid_transform(image = img)
            
        return augmented['image']
            
            
        
def augmentation1():
    return Compose([RandomCrop(height = IMAGE_DIM, widht = IMAGE_DIM, p = 1.0),
                    HorizontalFlip(p = 0.5),
                    ToTensor(Normalize = normalise)],
                    p = 1   # the exponent value in the norm formulation.
                    )
    
def augmentation2():
    return Compose([ToTensor(normalize = normalise)],
                   p = 1)
            

def normalise():
    return dict(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

