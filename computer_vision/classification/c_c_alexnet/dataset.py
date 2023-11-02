import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split


def get_dataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR):
    classes, path_label_df, label_num_df  = prepare_dataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR)

    labels = pd.get_dummies(path_label_df['Label'])
    train_X, test_X, train_y, test_y = train_test_split(path_label_df, 
                                                        labels, 
                                                        test_size = 0.25,
                                                        random_state = 5, 
                                                        stratify = path_label_df['Label']
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



