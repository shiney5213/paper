import os
import random
import numpy as np
import cv2
from PIL import Image

from dataset import get_dataset
from datapreprocessing import StanfordDogsDataset
from dataloader import loader
from utils import get_lr, acculate
from model import PaperAlexNet

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
# NUM_EPOCHS = 30
NUM_EPOCHS = 1

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
CHECKPOINT_DIR = OUTPUT_ROOT_DIR + '/models'  # model checkpoints

 


# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



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
    print('train_dataloader', len(train_dataloader))    # 128 * 6 = 768
    print('val_dataloader', len(val_dataloader))        # 128 * 2 = 456
    
        
  
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
    #                        lr = LR)
    optimizer = optim.SGD(
        params = alexnet.parameters(),
        lr = LR_INIT,
        momentum = MOMENTUM,
        weight_decay= LR_DECAY
    )
    
    # multiply RL by 1/ 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma = 0.1)
    print('LR Scheduler created')
    
    # create loss
    criterion = nn.CrossEntropyLoss()

    # start training
    print('Starting Training')
    losses_df = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': []}
    NUM_BATCHES = len(train_X)//BATCH_SIZE
    VAL_NUM_BATCHES = len(val_X)//BATCH_SIZE
    best_loss = np.inf
    
    
    train_loss = 0
    for epoch in range(NUM_EPOCHS):
        print('=== Epoch:', epoch)
        # lr update
        lr_scheduler.step()
        print('Learning rate:', get_lr(optimizer))
        loss = 0
        

        
        train_outputs = []
        for i, batch in enumerate(train_dataloader):
            imgs = batch['image']
            labels = batch['label']
           
            imgs, labels = imgs.to(device, dtype = torch.float), labels.to(device, dtype = torch.float)
            
            # foward propagation
            train_output = alexnet(imgs)
            
            # calculate the loss : cross entropy loss
            # batch_loss = F.cross_entropy(train_output, labels)
            batch_loss = criterion(train_output, torch.max(labels, 1)[1])
            
            
            # set gradient 0
            optimizer.zero_grad()
            # backpropagation
            batch_loss.backward()
            # update the parameters
            optimizer.step()
            
            train_output = train_output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            train_outputs.extend(np.argmax(train_output, 1))
            train_loss += batch_loss.item()
            
        # culuate loss
        train_loss = train_loss/NUM_BATCHES
        losses_df['train_loss'].append(train_loss)
        
        # culculate acc
        train_acc = acculate(train_y, train_outputs)
        losses_df['train_acc'].append(train_acc)
        
        
        print('train : loss - {:.4f}'.format(train_loss), 'acc - {:.4f}'.format(train_acc))
        
        val_loss = 0
        val_outputs = []
        # validation, test : not backpropagation
        with torch.no_grad():
            alexnet.eval()
            
            
            for i , batch in enumerate(val_dataloader):
                imgs = batch['image']
                labels = batch['label']
            
                imgs, labels = imgs.to(device, dtype = torch.float), labels.to(device, dtype = torch.float)
                
                # forward propagation
                val_output = alexnet(imgs)
                
                # get loss
                batch_loss = criterion(val_output, torch.max(labels, 1)[1])
                
                
                val_output = val_output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                val_outputs.extend(np.argmax(val_output, 1))
                val_loss += batch_loss.item()
                
        # culuate loss
        val_loss = val_loss/NUM_BATCHES
        losses_df['val_loss'].append(val_loss)
        
        # culuate acc
        val_acc = acculate(val_y, val_outputs)
        losses_df['val_acc'].append(val_acc)
    
        print('val : loss - {:.4f}'.format(val_loss), 'acc - {:.4f}'.format(val_acc))
    
                
        # if val_loss <= best_loss:
        #     best_loss = val_loss
   
        if 1:
            # save checkpoints
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pth'.format(epoch + 1))
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model' : alexnet.state_dict(),
                'seed' : seed,
                'losses_df': losses_df
            }
            
            torch.save(state, checkpoint_path)
            print('model save')
       
        
    
    
    
    
    
    