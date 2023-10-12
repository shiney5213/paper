# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py#L36
# https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L40
import os
import torch
from torch import nn


class AlexNet(nn.Module):
    """_summary_
    Neural network model consisting oflayers propsed by AlexNet paper
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        # _log_api_usage_once(self)
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int, optional): _description_. Defaults to 1000.
            dropout (float, optional): _description_. Defaults to 0.5.
        """
        super().__init__()
        self.net = nn.Sequential(
            # inpur size : (N, C, H, W) = (BATCH_SIZE(128), 3, 256, 256)
            # num of kernel : reference paper
            # Conv layer1
            nn.Conv2d(in_channels = 3, out_channels= 96, kernel_size = 11, stride = 4, padding = 2), 
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),  # section 3.3
            nn.MaxPool2d(kernel_size= 3, stride = 2),
            
            # Conv layer2
            nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2),  # section 3.3
            nn.MaxPool2d(kernel_size= 3, stride = 2),
            
            # Conv layer3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv layer4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv layer5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(
            # dropout layer before Linear layer 
            nn.Dropout(p=dropout),
            # FCL 1
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # FCL 2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # FCL 3
            nn.Linear(4096, num_classes)
        )
        self.init_bais()
    
    def init_bais(self):
        """
        initialize weight, bias
        """
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)
                
            # original papaer = 1 for Conv2d layers 2nd, 4th and 5th conv layers
            nn.init.constant_(self.net[4].bias, 1)
            nn.init.constant_(self.net[10].bias, 1)
            nn.init.constant_(self.net[12].bias, 1)
            
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        pass the input through the net.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            output(torch.Tensor): output tensor 
        """
        
        x = self.net(x)
        # x = x.view(-1, 256*6*6 ) # reduce the dimensions for linear layer input
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        
        return x
    
