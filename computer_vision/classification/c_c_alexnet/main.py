import os
import random
import numpy as np
import cv2
from skimage import io
from PIL import Image

import torch
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataset
from datapreprocessing import StanfordDogsDataset



from model import AlexNet
# from dataloader import dataloader, loader
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision import transforms

# from train import train_one_epoch

# define pytorch devies - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
LR =0.0001
# IMAGE_DIM = 227  # pixels
IMAGE_DIM = 256
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use


print('my location', os.getcwd())
# modify this to point to your data directory
INPUT_ROOT_DIR = './classification/c_c_alexnet/'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'alexnet_data_in/stanford-dog-dataset/images')
TRAIN_ANNO_DIR =  os.path.join(INPUT_ROOT_DIR, 'alexnet_data_in/stanford-dog-dataset/annotations')
# .replace('\\', '/')
OUTPUT_DIR =  os.path.join(INPUT_ROOT_DIR, 'alexnet_data_out/standofd-dog-datset')
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints



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
    stanforddogs_dataset = StanfordDogsDataset(TRAIN_IMG_DIR,
                                               train_X, 
                                               train_y,
                                               True,
                                            )
                                           

    # Apply each of the above transforms on sample.
    print('stanforddogs_dataset', len(stanforddogs_dataset))
    
    sample = stanforddogs_dataset[65]
    print('sample', sample['image'].shape)
    
    
    # for i, tsfrm in enumerate([scale, crop, composed1, composed2]):
    #     transformed_sample = tsfrm(sample)
    #     print(i,':', type(transformed_sample['image']), transformed_sample['image'].shape )
    
    # is_show = True
    # # is_show = False
    # idx = 5
    # train_dataloader = loader(TRAIN_IMG_DIR, train_X, train_y, is_show, idx, batch_size = BATCH_SIZE, obj = 'train')
    # print('train_dataloader', type(train_dataloader))
    
    # for batch in train_dataloader:
    #     img, label = batch
    #     print(len(img))
        
    raise ValueError
  
     # create Model
    alexnet = AlexNet(num_classes = NUM_CLASSES).to(device)
    # train in multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids = DEVICE_IDS)
    print('AlexNet created')
    print(alexnet)

    raise ValueError
    
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
    
    
    # start training
    print('Starting Training')
    for epoch in range(NUM_EPOCHS):
        # lr update
        lr_scheduler.step()
        
        total_steps = 1
        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            
            # calculate the loss : cross entropy loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)
            
            # update the parameters
            optimizer.zero_grad()
            loss.backword()
            optimizer.step()
            
            
            # log the information and add to tensorboad
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)
                    loss = loss.item()
                    accuracy = accuracy.item()
                    
                    print('Epoch: {}\tStep: {} \Loss: {:.4f} \tAcc: {}'.format(epoch + 1, total_steps, loss, accuracy))
                    tbwriter.add_scalar('loss', loss, total_steps)
                    tbwriter.add_scalar('accuracy', accuracy, total_steps)
                    
            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name), parameter.grad.cpu().numpy(), total_steps)
                        
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
                            tbwriter.add_histogram('weight/{}'.format(name), parameter.data.cpu().numpy(), total_steps)
                            
        total_steps += 1
        
        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model' : alexnet.state_dict(),
            'seed' : seed
        }
        
        torch.save(state, checkpoint_path)
    
        
    
    
    
    
    
    