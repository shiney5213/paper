import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import PIL

from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets 
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset, DataLoader  
from albumentations import CenterCrop
# from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
# from albumentations.pytorch.transforms import ToTensor
from datapreprocessing import PCA_color_aug
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip
# from torchvision.transforms import v2




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
    
    return  classes, path_label_df, label_num_df

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
    
    return df

def select_classes(df, class_num):
    '''
    select the 5 classes with the largest number: sebset
    '''
    classes = df.sort_values('Number of images', ascending = False)['Breeds'][:class_num].values.tolist()
    print('classes', classes)
    
    return classes


def select_path_label(TRAIN_IMG_DIR, classes ):
    """
    create df with path and label
    """
    valid = []
    for element in os.listdir(TRAIN_IMG_DIR):
        if element.split('-')[1] in classes:
            element_path = os.listdir(os.path.join(TRAIN_IMG_DIR, element))
            for img_id in element_path:
                path = os.path.join(element, img_id)
                label = element.split('_')[0].split('-')[1]
                valid.append((path, label))
                
    df = pd.DataFrame(valid, columns = ['Path', 'Label'])
    print('shape of dataframe:', df.shape)
    return df


class StanfordDogs(Dataset):
    def __init__(self, transform1, transform2, normalise1, X, Y, image_path,  objective, is_show = False):
    # def __init__(self, X, Y, image_path,  objective, is_show = False):
        
        self.X = X
        self.Y = Y
        # self.train_transform = transform1
        # self.valid_transform = transform2
        self.normalise1 = normalise1
        self.objective = objective
        self.image_path = image_path
        self.IMAGE_TRAIN_DIM = 256
        self.IMAGE_TEST_DIM = 224
        self.is_show = is_show
        
        

    def __getitem__(self, idx):
        """
        open image by referring to the idx, then perform proprocessing and convert it to tensor data
        """
        path = self.X['Path'][idx]
        label = self.Y.iloc[idx, :].values
        img = cv2.imread(os.path.join(self.image_path, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resize_256 = self.resize_256(img)
        
        img_centercrop = self.centercrop(img_resize_256)
        
        if self.objective == 'train':
            img_pca_aug = self.pca_aug(img_centercrop)
            print('img_pca_aug', img_pca_aug)
        
            img_final = self.transform(img_pca_aug)
        else:
            img_final = self.transform(img_centercrop)
            
            
        
        if self.is_show:
            self.show_img('Original Image', img)
            self.show_img('resize_256', img_resize_256)
            self.show_img('img_centercrop', img_centercrop)
            self.show_img('PCA Color Augmentation', img_pca_aug) if self.objective == 'train' else  print(self.objective)
            self.show_img('transform', img_final )
                
        
            
        return img_final, label
    
    def show_img(self, img_name, img):
        if torch.is_tensor(img):
            img = img.numpy()
            img = self.numpy_shape(img)
            
        cv2.imshow( f'{img_name}_{img.shape}', img)
        cv2.waitKey(0)
        
        
    def __len__(self):
        """
        return number of train dataset
        """
        return len(self.X)
    
    
    
    def resize_256(self, img):
        # Shortest side of image is scaled to 256 pixels and the other side is scaled so as to maintain aspect ratio
        h, w, _ = img.shape
        
        if h <= w:
            aspect_ratio = w/h
            dim = (self.IMAGE_TRAIN_DIM, int(self.IMAGE_TRAIN_DIM * aspect_ratio))
            img_256 = cv2.resize(img, dim)
        else:
            aspect_ratio = h/w
            dim = (int(self.IMAGE_TRAIN_DIM * aspect_ratio), self.IMAGE_TRAIN_DIM)
            img_256 = cv2.resize(img, dim)
        
        return img_256
    
    def centercrop(self, img):
        
        img_centercrop = CenterCrop(height = self.IMAGE_TRAIN_DIM, width = self.IMAGE_TRAIN_DIM, p = 1)(image = img)['image']
        # centercrop = CenterCrop(self.IMAGE_TRAIN_DIM)
        # img_centercrop = centercrop(torch.Tensor(img_256)).numpy()
        
        return img_centercrop
    
    def pca_aug(self, img):
        # PCA Augmentation carried out only 50 percent of time
        random = np.random.uniform(size = 1)
        if random < 0.5:
            # img_pac_aug = PCA_color_aug(img, category = 'numpy').astype(np.float32)
            img_pac_aug = PCA_color_aug(img, category = 'numpy')
            return img_pac_aug
        else:
            return img
                
        
    
    def transform(self, img):
        
        if self.objective == 'train':
            # img, tranforms = self.train_transform_tensor(img)
            img, tranforms = self.train_transform_PIL(img)
        elif ((self.objective == 'validation') | (self.objective == 'test')):
            img = cv2.resize(img, (self.IMAGE_TEST_DIM, self.IMAGE_TEST_DIM))
            transforms = self.valid_transform()
            img_transformed = transforms()
        # img_transformed = tranforms(torch.Tensor(img))
        img_transformed = tranforms(img)
        # img_transformed= img_transformed/255
        print('img_transformed', img_transformed)
        
        return img_transformed
            

    
    def train_transform_tensor(self, img):
        """
        img: numpy -> tensor
        problem: 값이 너무 커서 대체적으로 하얗게 나옴. 255 이상.
        """
        
        img = self.tensor_shape(img).astype(np.float32)
        img = torch.tensor(img)
        transform =  Compose([
                        RandomCrop((self.IMAGE_TEST_DIM, self.IMAGE_TEST_DIM)),                # tensor
                        RandomHorizontalFlip(p = 0.5),   
                        # input: PIL
                        Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),  # tensor
                        ],
                )
        return img, transform
    
    def train_transform_PIL(self, img):
        """
        img: numpy-> PIL
        problem: 값이 정규화(-1~1)되었으나 흑백 이미지, 9등분으로 분할되어 나옴
        """
        # img = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
        img = Image.fromarray(img)
        transform =  Compose([
                        RandomCrop((self.IMAGE_TEST_DIM, self.IMAGE_TEST_DIM)),                # input: PIL, tensor
                        RandomHorizontalFlip(p = 0.5),                                         # input: PIL, tensor
                        ToTensor(),
                        Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),  # input: tensor
                        ],
                )
        return img, transform
    
    
    
    def valid_transform(self):
        return Compose([Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
                        # ToTensor()
                        ]
                        )
    
    def show_shape(self, img_name, img ):
        if isinstance(img, np.ndarray):
            print(img_name, ":", type(img), img.shape)
        elif isinstance(img, torch.Tensor):
            print(img_name, type(img), img.size())
        elif isinstance(img, PIL.Image.Image):
            print(img_name, type(img), img.size)
        else:
            print(img_name, type(img))
            
            
    def tensor_shape(self, img):
        """
        tensor image shape : (3, d, d)
        """
        demention = img.shape[1]
        return img.reshape(3, demention, demention)
    
    def numpy_shape(self, img):
        """
        numpy image shape : (d, d, 3)
        """
        demention = img.shape[1]
        
        return img.reshape(demention, demention, 3)
        
                
        
            
        
    
    

            
            
def get_dataset(df):
   
    labels = pd.get_dummies(df['Label'])
    train_X, test_X, train_y, test_y = train_test_split(df, 
                                                        labels, 
                                                        test_size = 0.25,
                                                        random_state = 5, 
                                                        stratify = df['Label']
                                                        )
    
    train_X, val_X, train_y, val_y = train_test_split(train_X,
                                                      train_y,
                                                      test_size = 0.25,
                                                      random_state = 5,
                                                      stratify = train_X['Label'])
    
    train_X.reset_index(drop = True, inplace = True)
    val_X.reset_index(drop = True, inplace = True)
    test_X.reset_index(drop = True, inplace = True)
    
    train_y.reset_index(drop = True, inplace = True)
    val_y.reset_index(drop = True, inplace = True)
    test_y.reset_index(drop = True, inplace = True)
    
    return train_X, val_X, test_X, train_y, val_y, test_y
    