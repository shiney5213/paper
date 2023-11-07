import os
import random
import numpy as np
import cv2
from PIL import Image

from dataset import get_dataset
from datapreprocessing import StanfordDogsDataset
from dataloader import loader
from utils import get_lr, acculate, historyplot
from model import PaperAlexNet

from sklearn.metrics import accuracy_score
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
TEST_MODEL_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models/alexnet_states_e1.pth')

 


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
    

    test_dataloader = loader(TRAIN_IMG_DIR, test_X, test_y, batch_size = BATCH_SIZE, obj = 'test')
    
    
    # load model
    
    alexnet = alexnet(num_classes = NUM_CLASSES).to(device)
    alexnet.load_state_dict(torch.load(TRAIN_MODEL_DIR))
    alexnet.classifier[6] = nn.Linear(4096, 5)  # 5 classes
    # train in multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids = DEVICE_IDS)
    alexnet = alexnet.to(device)
    print('AlexNet created')
    
    optimizer = optim.SGD(
        params = alexnet.parameters(),
        lr = LR_INIT,
        momentum = MOMENTUM,
        weight_decay= LR_DECAY
    )
    
    
    
    checkpoint = torch.load(TEST_MODEL_DIR)
    alexnet.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    historyplot(checkpoint['losses_df'])
    
    
    
    # test
    # test_preds = []
    # with torch.no_grad():
    #     alexnet.eval()
        
    #     for i, batch in enumerate(test_dataloader):
    #         img, label = batch
    #         img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
    #         output = alexnet(img)
            
    #         output = output.detach().cpu().numpy()
    #         test_preds.extend(np.argmax(output, 1))
            
    #     test_acc = acculate(test_y, test_preds)
        
    #     print('Accuracy on test dataset : {:.2%}'.format(test_acc))