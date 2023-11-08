import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score
import torch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def acculate(y, output):
    labels = list(np.argmax(np.array(y), 1))
    acc = accuracy_score(output, labels)
    return acc
    
    
def model_save(CHECKPOINT_DIR, model, optimizer, losses_df,  seed, epoch, lr_list):
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pth'.format(epoch + 1))
        
    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'model' : model.state_dict(),
        'seed' : seed,
        'losses_df': losses_df,
        'lr': lr_list
    }
    
    torch.save(state, checkpoint_path)
    print('model save')
       
        
    