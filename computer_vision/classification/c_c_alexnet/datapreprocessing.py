import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import PIL

import torch
from torch.utils.data import Dataset  
from albumentations import CenterCrop
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip






class StanfordDogsDataset(Dataset):
    """StanfordDogs datasets """
    def __init__(self, image_path, X, y, is_show, objective = 'train' ):
        self.image_path = image_path
        self.objective = objective
        self.X = X
        self.y = y
        self.IMAGE_TRAIN_DIM = 256
        self.IMAGE_TEST_DIM = 224
        self.is_show = is_show

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        path = self.X['Path'][idx]
        label = self.y.iloc[idx, :].values
        img = Image.open(os.path.join(self.image_path, path)).convert('RGB')
        img = cv2.imread(os.path.join(self.image_path, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resize_256 = self.resize256(img)
        
        img_centercrop = self.centercrop(img_resize_256)
        
        if self.objective == 'train':
            img_pca_aug = self.pca_aug(img_centercrop)
        
            img_final = self.transform(img_pca_aug)
        else:
            img_final = self.transform(img_centercrop)
        
        if self.is_show:
            self.show_img('Original Image', img)
            self.show_img('resize_256', img_resize_256)
            self.show_img('img_centercrop', img_centercrop)
            self.show_img('PCA Color Augmentation', img_pca_aug) if self.objective == 'train' else  print(self.objective)
            self.show_img('transform', img_final )
            
        sample = {'image': img_final, 'label': label}
        return sample
            
    def show_img(self, img_name, img):
        if torch.is_tensor(img):
            img = img.numpy()
            img = self.numpy_shape(img)
        elif isinstance(img, PIL.Image.Image):
            img = np.array(img)
            
        cv2.imshow( f'{img_name}_{img.shape}', img)
        cv2.waitKey(0)
    
    def transform(self, img):
        img = Image.fromarray(img)
        
        if self.objective == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop((224, 224)),      # 중앙 Crop
                transforms.RandomHorizontalFlip(0.5),   # 50% 확률로 Horizontal Flip
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
            ])
            
        img = transform(img)
        img = np.array(img)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((1, 2, 0))

        return img
    
    def resize256(self, img):
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
        
        
        
def PCA_color_aug(image, category = 'Tensor'):
    """augmenation with color info

    Args:
        image (_type_): _description_
        category (str, optional): _description_. Defaults to 'Tensor'.
    """
    
    IMAGE_TRAIN_DIM = 256
    
    if type(image)==torch.Tensor:
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
    
    img_reshaped = image.reshape(-1, 3).astype('float32')
    mean, std = np.mean(img_reshaped, axis = 0), np.std(img_reshaped, axis = 0)
    img_rescaled = (img_reshaped - mean) / std 
    cov_matrix = np.cov(img_rescaled, rowvar = False)  # Covariant matrix of reshaped image, ourput is 3*3 matirx
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix)   # compute eigen Values and Eigen Vectors of the covariant matix
                                                       # eigen_vec is 3*3 matrix with eigen Vectors as column.
    
    alphas = np.random.normal(loc = 0, scale = 0.1, size = 3)
    vec1 = alphas*eigen_val
    valid = np.dot(eigen_vec, vec1)
    pca_aug_norm_image = img_rescaled + valid
    pca_aug_image = pca_aug_norm_image * std + mean
    aug_image = np.maximum(np.minimum(pca_aug_image, 255), 0).astype(np.uint8)
    if category == 'Tensor':
        aug_image =  torch.from_numpy(aug_image.reshape(3, IMAGE_TRAIN_DIM, IMAGE_TRAIN_DIM))
    else:
        aug_image =  aug_image.reshape(IMAGE_TRAIN_DIM,IMAGE_TRAIN_DIM,3)
    
    return aug_image
    
                




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
                
        sample = {'image': img_final, 'label': label}
        return sample
            
               
        
        
        
        

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
                        # RandomHorizontalFlip(p = 0.5),         
                        RandomHorizontalFlip(1),         
                        
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
        
                
        
            
        
    
    

            
            
