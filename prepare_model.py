# File: prepare_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from configs import Config
from recovery.data import get_dataloaders
from recovery.nn.custom_cnn import get_custom_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Config.DEVICE = device
print(f"Device: {device}")
if device.type == 'cuda':
    # Lấy index hiện tại từ device
    current_gpu_index = device.index 
    # Lấy tên card
    gpu_name = torch.cuda.get_device_name(current_gpu_index)
    print(f"GPU Name: {gpu_name}")
else:
    print("Đang chạy trên CPU (không tìm thấy GPU hoặc CUDA không khả dụng)")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / len(loader), 100. * correct / total




def main():
    parser = argparse.ArgumentParser(description='Phase 1: Prepare Target Model for Unlearning Attack')
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10, cifar100, mnist')
    parser.add_argument('--model', default='ResNet18', type=str) 
    parser.add_argument('--pretrain_epochs', default=40, type=int, help='Epochs to train on RETAIN set')
    parser.add_argument('--finetune_epochs', default=20, type=int, help='Epochs to finetune on FORGET set')
    parser.add_argument('--lr', default=Config.LR_PRETRAIN, type=float, help='Learning rate')
    args = parser.parse_args()

    # 1. Prepare Data
    print(f"==> Preparing data {args.dataset}...")
    retain_loader, forget_loader, test_loader, num_channels, img_size, num_classes = get_dataloaders(args.dataset)

    # 2. Build Model
    print(f"==> Building model {args.model}...")
    model = get_custom_model(args.model, num_channels=num_channels, num_classes=num_classes, img_size=img_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # -----------------------------------------------------------
    # GIAI ĐOẠN 1: PRE-TRAIN (Học kiến thức nền trên tập Retain)
    # -----------------------------------------------------------
    print(f"\n[Phase 1A] Pre-training on Retain Set ({len(retain_loader.dataset)} images)...")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs, eta_min=1e-6)
    
    for epoch in range(args.pretrain_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, retain_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{args.pretrain_epochs} | "
              f"Loss: {train_loss:.3f} | Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Time: {time.time()-t0:.1f}s")

    # >>> SAVE MODEL PRETRAINED (Gold Standard) <<<
    pretrain_name = f"{args.model}_{args.dataset}_pretrained.pth"
    pretrain_path = os.path.join(Config.MODEL_SAVE_PATH, pretrain_name)
    torch.save(model.state_dict(), pretrain_path)
    print(f"\n[SAVED] Pretrained Model saved to: {pretrain_path}")

    # -----------------------------------------------------------
    # GIAI ĐOẠN 2: FINETUNE (Học 10k dữ liệu cần tấn công)
    # -----------------------------------------------------------
    print(f"\n[Phase 1B] Fine-tuning on Forget Set ({len(forget_loader.dataset)} images)...")
    
    # Giảm LR để học kỹ các chi tiết của 10k ảnh này
    optimizer = optim.SGD(model.parameters(), lr=Config.LR_FINETUNE, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.finetune_epochs):
        train_loss, train_acc = train_epoch(model, forget_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        if epoch % 5 == 0 or epoch == args.finetune_epochs - 1:
            print(f"Finetune Epoch {epoch+1}/{args.finetune_epochs} | Forget Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # -----------------------------------------------------------
    # ĐÁNH GIÁ CUỐI CÙNG TRƯỚC KHI LƯU (FINAL EVALUATION)
    # -----------------------------------------------------------
    print(f"\n" + "="*70)
    print(" ĐÁNH GIÁ HIỆU NĂNG TOÀN DIỆN TRƯỚC KHI LƯU ".center(70, "="))
    print("-> Đang kiểm tra độ chính xác trên toàn bộ các tập dữ liệu...")
    
    # Tính toán lại độ chính xác trên từng tập
    final_retain_loss, final_retain_acc = test(model, retain_loader, criterion, device)
    final_forget_loss, final_forget_acc = test(model, forget_loader, criterion, device)
    final_test_loss, final_test_acc = test(model, test_loader, criterion, device)
    
    print("-" * 70)
    print(f"• Tập Retain (Dữ liệu huấn luyện nền)  | Acc: {final_retain_acc:.2f}% | Loss: {final_retain_loss:.4f}")
    print(f"• Tập Forget (Dữ liệu unlearn mục tiêu)| Acc: {final_forget_acc:.2f}% | Loss: {final_forget_loss:.4f}")
    print(f"• Tập Test   (Dữ liệu kiểm thử chung)   | Acc: {final_test_acc:.2f}% | Loss: {final_test_loss:.4f}")
    print("="*70 + "\n")

    # >>> SAVE MODEL FINETUNED (Target) <<<
    finetune_name = f"{args.model}_{args.dataset}_finetuned.pth"
    finetune_path = os.path.join(Config.MODEL_SAVE_PATH, finetune_name)
    torch.save(model.state_dict(), finetune_path)
    print(f"[SAVED] Target Model saved to: {finetune_path}")
    print("==> Giai đoạn 1 hoàn tất. Bạn đã có đủ cặp model để thí nghiệm.")
    
if __name__ == '__main__':
    main()