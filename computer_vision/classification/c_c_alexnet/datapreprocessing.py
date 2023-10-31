
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets 
from torchvision import transforms
from torch.utils import data
from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
from albumentations.pytorch.transforms import ToTensor
# from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip



TRAIN_IMG_DIR = './alexnet_data_in/imagenet'
IMAGE_TRAIN_DIM = 256
IMAGE_TEST_DIM = 224
BATCH_SIZE = 128


def augmentation1():
    return Compose([RandomCrop(height = IMAGE_TEST_DIM, width = IMAGE_TEST_DIM, p = 1.0),
                    HorizontalFlip(p = 0.5),
                    ToTensor(normalize= normalise)],
                    p = 1   # the exponent value in the norm formulation.
                    )
    
def augmentation2():
    return Compose([ToTensor(normalize = normalise)],
                   p = 1)
            

def normalise():
    return dict(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])



def PCA_color_aug(image, category = 'Tensor'):
    """augmenation with color info

    Args:
        image (_type_): _description_
        category (str, optional): _description_. Defaults to 'Tensor'.
    """
    
    if type(image)==torch.Tensor:
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        
    img_reshaped = image.reshape(-1, 3).astype('float32')
    mean, std = np.mean(img_reshaped, 0), np.std(img_reshaped, 0)
    img_rescaled = (img_reshaped - mean) / std
    cov_matrix = np.cov(img_rescaled, rowvar = False)  # Covariant matrix of reshaped image, ourput is 3*3 matirx
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix)   # compute eigen Values and Eigen Vectors of the covariant matix
                                                       # eigen_vec is 3*3 matrix with eigen Vectors as column.
    alphas = np.random.normal(loc = 0, scale = 0.1, size = 3)
    vec1 = alphas*eigen_val
    valid = np.dot(eigen_vec, vec1)
    pca_aug_norm_image = img_rescaled + valid
    pca_aug_image = pca_aug_norm_image * std + mean
    print('pca_aug_image:', pca_aug_image.shape, pca_aug_image[0,0], type( pca_aug_image[0,0]))
    aug_image = np.maximum(np.minimum(pca_aug_image, 255), 0).astype(np.uint8)
    if category == 'Tensor':
        return torch.from_numpy(aug_image.reshape(3, IMAGE_TRAIN_DIM, IMAGE_TRAIN_DIM))
    else:
        return aug_image.reshape(IMAGE_TRAIN_DIM,IMAGE_TRAIN_DIM,3)
    