import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import time
import numpy as np
from configs import Config
from recovery.data import get_dataloaders
from recovery.nn.custom_cnn import get_custom_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Config.DEVICE = device
print(f"Device: {device}")
if device.type == 'cuda':
    current_gpu_index = device.index 
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


# =========================================================================
# --- [NEW] HÀM TÍNH TOÁN THỐNG KÊ PHÂN PHỐI VÀ PHƯƠNG SAI CÁC LỚP ---
# =========================================================================
def evaluate_class_stats(model, loader, num_classes, device):
    """
    Tính toán các chỉ số phân phối lớp và phân tích Phương sai trên Vector E đã được chuẩn hóa L2.
    """
    model.eval()
    prob_sums = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)
    correct_counts = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            for c in range(num_classes):
                mask = (targets == c)
                count_c = mask.sum().item()
                if count_c > 0:
                    class_counts[c] += count_c
                    prob_sums[c] += probs[mask].sum(dim=0)
                    correct_counts[c] += (predicted[mask] == targets[mask]).sum().item()
                    
    class_counts_safe = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
    
    avg_probs = prob_sums / class_counts_safe.unsqueeze(1)
    class_accs = (correct_counts / class_counts_safe) * 100.0
    
    # 1. Tính toán Vector sai số E
    E_vector = 100.0 - class_accs
    
    # 2. Tính L2 Norm của Vector E
    l2_norm = torch.norm(E_vector, p=2).item()
    
    # 3. Chuẩn hóa Vector E theo L2 Norm (Độ dài hình học của vector co về bằng 1)
    if l2_norm > 0:
        E_normalized = E_vector / l2_norm
    else:
        E_normalized = torch.zeros_like(E_vector)
        
    # 4. --- LOGIC CHỦ CHỐT: TÍNH PHƯƠNG SAI TRÊN VECTOR E ĐÃ CHUẨN HÓA L2 ---
    var_E_normalized = torch.var(E_normalized, unbiased=False).item()
    # ------------------------------------------------------------------------
    
    mean_acc = torch.mean(class_accs).item()
    std_acc = torch.std(class_accs, unbiased=False).item() 
    cv_ratio = std_acc / mean_acc if mean_acc > 0 else 0.0 
    
    return (avg_probs.cpu().numpy(), 
            class_accs.cpu().numpy(), 
            mean_acc, 
            std_acc, 
            cv_ratio, 
            l2_norm, 
            var_E_normalized,       # Trả về phương sai của vector đã chuẩn hóa
            E_vector.cpu().numpy(),
            E_normalized.cpu().numpy()) # Trả về thêm vector đã chuẩn hóa để quan sát

# =========================================================================


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
        
        # --- [NEW] THỰC THI ĐO ĐẠC THỐNG KÊ LỚP ---
        (avg_probs, class_accs, mean_acc, std_acc, cv_ratio, 
         l2_norm, var_E_normalized, E_vector, E_normalized) = evaluate_class_stats(model, retain_loader, num_classes, device)
        
        scheduler.step()
        
        # In ra các thông số huấn luyện tiêu chuẩn
        print(f"Epoch {epoch+1:02d}/{args.pretrain_epochs} | "
              f"Loss: {train_loss:.3f} | Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Time: {time.time()-t0:.1f}s")
              
        # In các thông số phân tán cơ bản
        print(f"   [Thống kê phân tán] Mean Acc: {mean_acc:.2f}% | Độ lệch chuẩn (Std): {std_acc:.4f} | Hệ số biến thiên (CV): {cv_ratio:.6f}")
        
        # --- CẬP NHẬT PHẦN HIỂN THỊ PHƯƠNG SAI TRÊN VECTOR ĐÃ NORM ---
        print(f"   [Phân tích Vector Sai Số E (100 - Acc)]:")
        print(f"     • L2 Norm của Vector E            : {l2_norm:.4f}")
        print(f"     • Phương sai trên Vector E đã Norm: {var_E_normalized:.8f}")  # In 8 chữ số thập phân
        
        # In chi tiết các giá trị của Vector E đã chuẩn hóa (5 lớp đầu)
        limit_print = min(num_classes, 5)
        E_norm_str = ", ".join([f"{val:.4f}" for val in E_normalized[:limit_print]])
        print(f"     • Chi tiết Vector E đã Norm (5 lớp): [{E_norm_str}]" + (" ..." if num_classes > 5 else ""))
        
        print("   [Xác suất dự đoán trung bình của từng lớp (Softmax Vector)]:")
        for c in range(limit_print):
            prob_vec_str = ", ".join([f"{p:.4f}" for p in avg_probs[c]])
            print(f"     Lớp {c:02d}: [{prob_vec_str}] (Accuracy riêng lớp: {class_accs[c]:.2f}%)")
        if num_classes > 5:
            print(f"     ... (Đã ẩn {num_classes - 5} lớp còn lại để tránh tràn màn hình log console) ...")
        print("-" * 80)

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