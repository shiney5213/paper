# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py#L36
import torch
import torch.nn as nn


class AlexNet(nn.module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        # _log_api_usage_once(self)
        
        self.features = nn.Sequential(
            # inpur size : (N, Cin, H, W)
            # Conv layer1
            nn.Conv2d(in_channels = 3, out_channels= 64, kernel_size = 11, stride = 4, padding = 2), # paper: kernel: 96개
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size= 3, stride = 2),
            # Conv layer2
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size= 3, stride = 2),
            # Conv layer3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv layer4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv layer5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(
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
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
        )