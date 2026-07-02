#----------------------
# AUTHORS:
# DAMIAN - 10012545
# BHUVAN - 10001026
#-----------------------

import torch
import torch.nn as nn

from logging import getLogger
logger = getLogger(__name__)

activation_str = "ReLU"  # Placeholder for activation function, can be replaced with "ReLU" or others as needed.


class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.

    C configuration from Simonyan & Zisserman's VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs,activation=nn.ReLU):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            conv_padding = (kernel_size - 1) // 2  # To maintain spatial dimensions
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=conv_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation(inplace=True))
            current_in_channels = out_channels
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If spatial size shrinks (stride > 1) or channels change, adjust the shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    def __init__(self, in_channels,num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        activation_name = kwargs.get("activation_str") or activation_str
        Activation = getattr(nn, activation_name)
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            Activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            Activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Activation(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Activation(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            Activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(3072, 1024),
            Activation(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            Activation(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration of Simonyan & Zisserman, (2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        activation_name = kwargs.get("activation_str") or activation_str
        Activation = getattr(nn, activation_name)
        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2, activation=Activation),
            VGGBlock(64, 128, num_convs=2, activation=Activation),
            VGGBlock(128, 256, num_convs=3, activation=Activation),
            VGGBlock(256, 512, num_convs=3, activation=Activation),
            VGGBlock(512, 512, num_convs=3, activation=Activation)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            Activation(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            Activation(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 (He et al., 2016) adapted for smaller inputs.
    
    activation - flexible activation function to allow experimentation (e.g., ReLU, LeakyReLU, etc.)
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        activation_name = kwargs.get("activation_str") or activation_str
        activation = getattr(nn, activation_name)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        logger.info(f"Using activation function: {self.activation}")

        self.stage1 = nn.Sequential(
            ResBlock(64, 64, activation(inplace=True), stride=1),
            ResBlock(64, 64, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, activation(inplace=True), stride=2),          
            ResBlock(128, 128, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, activation(inplace=True), stride=2),
            ResBlock(256, 256, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, activation(inplace=True), stride=2),
            ResBlock(512, 512, activation(inplace=True), stride=1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
