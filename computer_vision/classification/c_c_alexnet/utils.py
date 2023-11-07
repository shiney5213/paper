import numpy as np
import cv2
from sklearn.metrics import accuracy_score



    
    
import matplotlib.pyplot as plt


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def acculate(y, output):
    labels = list(np.argmax(np.array(y), 1))
    acc = accuracy_score(output, labels)
    return acc
    
    
    
def historyplot(losses_df):
    
    val_loss = losses_df['val_loss']
    
    plt.plot(val_loss)
    
