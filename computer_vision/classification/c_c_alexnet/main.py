import os
import torch
from tensorboardX import SummaryWriter

from model import AlexNet
from dataloader import dataloader

# define pytorch devies - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
# IMAGE_DIM = 227  # pixels
IMAGE_DIM = 256
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use


# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints


# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('seed: {}'.format(seed))
    
    
    
    tbwriter = SummaryWriter(log_dir = LOG_DIR)
    print('TensorboardX summary writer created')
    
    
    # create Model
    alexnet = AlexNet(num_classes = NUM_CLASSES).to(device)
    # train in multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids = DEVICE_IDS)

    
    # create DataLoader
    train_dataloader = dataloader()
    print('dataloader', train_dataloader)
    # create optimizer
    
    
    
    
    
    