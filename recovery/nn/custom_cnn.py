# File: recovery/nn/custom_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ==========================================
# 1. MODELS CŨ (ConvNet, MLP...)
# ==========================================
# (Giữ nguyên các class cũ ở phần trước, tôi chỉ thêm phần mới ở dưới)

class BaseDynamicModel(nn.Module):
    def _get_flatten_size(self, features, img_size, num_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, img_size, img_size)
            x = features(dummy_input)
            return x.view(1, -1).shape[1]

class ConvNet(BaseDynamicModel):
    def __init__(self, width=64, num_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        linear_input_size = self._get_flatten_size(self.features, img_size, num_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class MLP(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, img_size=32):
        super().__init__()
        input_dim = num_channels * img_size * img_size
        width = 1024
        self.net = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear0', nn.Linear(input_dim, width)),
            ('relu0', nn.ReLU()),
            ('linear1', nn.Linear(width, width)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(width, width)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(width, num_classes))
        ]))
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. RESNET (CIFAR/MNIST VERSION)
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10, img_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # Thay đổi quan trọng: Kernel size 3x3 thay vì 7x7, stride 1 thay vì 2
        # Loại bỏ MaxPool để giữ kích thước feature map cho ảnh nhỏ
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) # Average Pool cuối cùng
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==========================================
# 3. MOBILENET V2 (CIFAR/MNIST VERSION)
# ==========================================
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # Stride 1 ở đây cho ảnh nhỏ
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_channels=3, num_classes=10, img_size=32):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==========================================
# FACTORY FUNCTION
# ==========================================
def get_custom_model(model_name, num_channels=3, num_classes=10, img_size=32):
    print(f"Initializing {model_name} for input {img_size}x{img_size}, ch={num_channels}")
    
    if model_name == 'ConvNet':
        return ConvNet(width=64, num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    elif model_name == 'MLP':
        return MLP(num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    elif model_name == 'ResNet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    elif model_name == 'ResNet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    elif model_name == 'MobileNetV2':
        return MobileNetV2(num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")