from torchvision import datasets 
from torchvision import transforms
from torch.utils import data


TRAIN_IMG_DIR = './alexnet_data_in/imagenet'
IMAGE_DIM = 256
BATCH_SIZE = 128



def dataloader():
    
    transform = transforms.Compose([
        # tranforms.RandomResizedCrop(IMAGE_DIM, scale = (0.9, 1.0), ratio = (0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet Normalization method
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageNet(root = TRAIN_IMG_DIR,
                                split = 'train',
                                transforms = transform
                                )                                   

    print('dataset:', dataset.shape)

    dataloader = data.DataLoader(dataset = dataset,
                                shuffle = True,
                                pin_memory = True,
                                num_workers = 8,
                                drop_last = True,
                                batch_size= BATCH_SIZE
                                )
    
    return dataloader





