import os
import random
import numpy as np
import cv2
from PIL import Image

from dataset import get_dataset
from datapreprocessing import StanfordDogsDataset
from dataloader import loader
from utils import get_lr, acculate, model_save
from model import PaperAlexNet
from train import train_loop, val_roop

import torch
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from torchvision.models import alexnet
# from dataloader import dataloader, loader
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision import transforms

# from train import train_one_epoch

# define pytorch devies - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define model parameters
# NUM_EPOCHS = 90  # original paper
NUM_EPOCHS = 30
# NUM_EPOCHS = 1

BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
# LR =0.0001
IMAGE_DIM = 256
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use


print('my location', os.getcwd())
# modify this to point to your data directory
INPUT_ROOT_DIR = './classification/c_c_alexnet/alexnet_data_in/stanford-dog-dataset'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'images')
TRAIN_ANNO_DIR =  os.path.join(INPUT_ROOT_DIR, 'annotations')
TRAIN_MODEL_DIR = os.path.join(INPUT_ROOT_DIR, 'archive/alexnet.pth')

OUTPUT_ROOT_DIR =  './classification/c_c_alexnet/alexnet_data_out/stanford-dog-dataset'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR + '/tblogs' ) # tensorboard logs



if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('seed: {}'.format(seed))
    
    # seed setting : REPRODUCIBILITY
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # deterministic algorithms으로 구성
    torch.backends.cudnn.deterministic = True
    
    
    # log writer
    tbwriter = SummaryWriter(log_dir = LOG_DIR)
    print('TensorboardX summary writer created')
    
    

    # create Dataset 
    train_X, val_X, test_X, train_y, val_y, test_y = get_dataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR)
    
    
    # create dataloader
    
    train_dataloader = loader(TRAIN_IMG_DIR, train_X, train_y, batch_size = BATCH_SIZE, obj = 'train')   # image: [ batch, 224, 224, 3]
    val_dataloader = loader(TRAIN_IMG_DIR, val_X, val_y, batch_size = BATCH_SIZE, obj = 'validation' ) # label : [batch, 5]
    
        
  
     # create Model
    alexnet = alexnet(num_classes = NUM_CLASSES).to(device)
    alexnet.load_state_dict(torch.load(TRAIN_MODEL_DIR))
    alexnet.classifier[6] = nn.Linear(4096, 5)  # 5 classes
    # train in multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids = DEVICE_IDS)
    alexnet = alexnet.to(device)
    print('AlexNet created')

    
    # create optimizer
    # optimizer = optim.Adam(params = alexnet.parameters(),
                        #   lr = LR)
    # 
    
    ##### 1st model #####
    # CHECKPOINT_DIR = OUTPUT_ROOT_DIR + '/models1' 
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # optimizer = optim.SGD(
    #     params = alexnet.parameters(),
    #     lr = LR_INIT,
    #     momentum = MOMENTUM,
    #     weight_decay= LR_DECAY
    # )
    
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma = 0.1)
    
    # ##### 2nd model #####
    # CHECKPOINT_DIR = OUTPUT_ROOT_DIR + '/models2' 
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # optimizer = optim.SGD(
    #     params = alexnet.parameters(),
    #     lr = LR_INIT,
    #     momentum = MOMENTUM,
    #     weight_decay= LR_DECAY
    # )
    
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma = 0.1)
    
        
    # #### 3th model ####
    # CHECKPOINT_DIR = OUTPUT_ROOT_DIR + '/models3' 
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # optimizer = optim.Adam(
    #     params = alexnet.parameters(),
    #     lr = LR_INIT,
    #     weight_decay= LR_DECAY
    # )
    
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma = 0.1)
    
    #### 4th model ####
    CHECKPOINT_DIR = OUTPUT_ROOT_DIR + '/models4' 
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    optimizer = optim.Adam(
        params = alexnet.parameters(),
        lr = LR_INIT,
        weight_decay= LR_DECAY
    )
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma = 0.1)
    

    
    # create loss
    criterion = nn.CrossEntropyLoss()

    # start training
    print('Starting Training')
    losses_df = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': []}
    NUM_BATCHES = len(train_X)//BATCH_SIZE
    VAL_NUM_BATCHES = len(val_X)//BATCH_SIZE
    best_loss = np.inf
    
    
    lr_list = []
    for epoch in range(NUM_EPOCHS):
        print('=== Epoch:', epoch)
        # lr update
        lr_scheduler.step()
        print('Learning rate:', get_lr(optimizer))
        lr_list.append(get_lr(optimizer))
        
        alexnet, optimizer, losses_df =  train_loop(train_dataloader, alexnet, criterion, optimizer, device, losses_df, NUM_BATCHES)       
        losses_df, val_loss =  val_roop(val_dataloader, alexnet, criterion, optimizer, device, losses_df, NUM_BATCHES)       
        
                
        if val_loss <= best_loss:
            best_loss = val_loss
            model_save(CHECKPOINT_DIR, alexnet, optimizer, losses_df,  seed, epoch, lr_list)
   
        if epoch == NUM_EPOCHS -1 :
            model_save(CHECKPOINT_DIR, alexnet, optimizer, losses_df,  seed, epoch, lr_list)
        
    
    
    
    
    
    