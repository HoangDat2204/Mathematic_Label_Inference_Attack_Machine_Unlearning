# File: recovery/data.py
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Subset
from configs import Config

def get_dataloaders(dataset_name, batch_size=Config.BATCH_SIZE, exclude_num=Config.FORGET_SIZE):
    """
    Load dataset và tách thành 2 phần:
    1. Retain Set: Dùng để Pre-train (Train từ đầu).
    2. Forget Set (exclude_num): Dùng để Finetune sau đó và Unlearn.
    """
    path = os.path.expanduser(Config.DATA_PATH)
    
    # --- 1. Define Transforms & Parameters based on Dataset ---
    if dataset_name == 'mnist':
        # MNIST: 28x28, 1 Channel
        # Chuẩn hóa mean/std chuẩn của MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Dataset gốc
        full_trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        
        num_channels = 1
        img_size = 28
        num_classes = 10
        
    elif dataset_name == 'cifar10':
        # CIFAR10: 32x32, 3 Channels
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        full_trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        
        num_channels = 3
        img_size = 32
        num_classes = 10
        
    elif dataset_name == 'cifar100':
        # CIFAR100: 32x32, 3 Channels
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        full_trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        
        num_channels = 3
        img_size = 32
        num_classes = 100
        
    else:
        raise ValueError(f"Dataset {dataset_name} chưa được hỗ trợ.")

    # --- 2. Split Data (Logic tách 10k) ---
    # Theo yêu cầu: "Chừa khoảng 10000 dữ liệu ra"
    # Forget Set = 0 -> exclude_num (10k ảnh đầu tiên)
    # Retain Set = exclude_num -> End (Phần còn lại để train model)
    
    indices = list(range(len(full_trainset)))
    
    forget_indices = indices[:exclude_num]
    retain_indices = indices[exclude_num:]
    
    trainset_retain = Subset(full_trainset, retain_indices)
    trainset_forget = Subset(full_trainset, forget_indices)

    # --- 3. Create Loaders ---
    retain_loader = torch.utils.data.DataLoader(
        trainset_retain, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    
    forget_loader = torch.utils.data.DataLoader(
        trainset_forget, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    print(f"Dataset: {dataset_name} | Image Size: {img_size}x{img_size}")
    print(f"Retain Size: {len(retain_indices)} | Forget/Finetune Size: {len(forget_indices)}")

    return retain_loader, forget_loader, test_loader, num_channels, img_size, num_classes