# File: recovery/unlearn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.data import Subset, DataLoader, ConcatDataset
from configs import Config
import math # Thêm thư viện math để tính logarit
from itertools import cycle
from recovery.nn.custom_cnn import get_custom_model
import time


class DistillKL(nn.Module):
    """Kullback-Leibler Divergence với Temperature Scaling"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # Sử dụng reduction='sum' kết hợp chia cho batch_size tương đương với code gốc
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss


class Unlearner:
    def __init__(self, target_model, base_model, device='cuda'):
        self.target_model = target_model
        self.base_model = base_model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    

    def _train_distill_epoch(self, loader, model_s, model_t, optimizer, criterion_div, split, alpha, gamma):
        """Hàm lõi chạy 1 epoch Distillation"""
        model_s.train()
        total_loss = 0.0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            
            # Student suy nghĩ
            logit_s = model_s(inputs)
            
            # Teacher đưa đáp án mẫu (không tính gradient)
            with torch.no_grad():
                logit_t = model_t(inputs)

            loss_cls = self.criterion(logit_s, targets)
            loss_div = criterion_div(logit_s, logit_t)

            # Lựa chọn chế độ theo tham số 'split'
            if split == "minimize":
                # Ép Student giống Teacher và đoán đúng nhãn
                loss = gamma * loss_cls + alpha * loss_div
            elif split == "maximize":
                # Ép Student dự đoán khác Teacher càng nhiều càng tốt
                loss = -loss_div

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)


    def approximate_unlearn(self, list_of_batches, lr=0.01):
        """
        Thực hiện Unlearning tuần tự trên nhiều batch (User gọi là nhiều epoch).
        Mỗi bước unlearn một batch ảnh khác nhau.
        
        Input:
            list_of_batches: List chứa các tuple (images, labels). 
                             Ví dụ: [(imgs1, lbls1), (imgs2, lbls2), ...]
                             Độ dài list chính là số 'epoch' bạn muốn.
        Output:
            model: Model sau khi đã unlearn xong toàn bộ chuỗi batch.
        """
        # Luôn bắt đầu từ Target Model (M_finetuned)
        model = copy.deepcopy(self.target_model)
        model.train()
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Unlearn tuần tự: M0 -> M1 -> M2 ...
        for i, (images, labels) in enumerate(list_of_batches):

            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            
            loss = -loss 
            loss.backward()
            optimizer.step()
        
        
        grads = []
        for param in model.parameters():
            grads.append(param.grad.detach().cpu().clone())
        w_grad, b_grad = grads[-2], grads[-1]
        gradients_for_prediction = torch.sum(w_grad, dim=-1)
        print(gradients_for_prediction)
            
            
        return model

    def fine_tune_unlearn(self, forget_dataset_base, indices_to_remove, unlr =0.001 ):
        """
        Exact Unlearn: Loại bỏ hoàn toàn tất cả các ảnh trong indices_to_remove
        (Tổng hợp của tất cả các batch trong chuỗi)
        """
        epochs=5
        lr=unlr

        # 1. Tạo dataset mới: D_new = D_full - {All 80 images}
        all_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)
        keep_indices = list(all_indices - remove_indices)
        
        sub_dataset = Subset(forget_dataset_base, keep_indices)
        loader = DataLoader(sub_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        # 2. Load Base Model (Pretrained on Retain Set)
        model = copy.deepcopy(self.base_model)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # 3. Finetune lại
        print(f"   [Exact Sim] Retraining on {len(keep_indices)} samples for {epochs} epochs...")
        for _ in range(epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        return model

    

    def scrub_unlearn(self, retain_dataset_base, forget_dataset_base, indices_to_remove, unlr = 0.001):
        """
        Thuật toán SCRUB SOTA: Alternating Min-Max Knowledge Distillation
        """
        # ==========================================
        # 1. GÁN CỨNG HYPERPARAMETERS CHUẨN SOTA
        # ==========================================
        T = 2           # Nhiệt độ làm mềm xác suất (Softmax Temperature)
        alpha = 1.0       # Trọng số cho KL Divergence
        gamma = 1       # Trọng số cho Cross-Entropy (thường để rất nhỏ hoặc 0 để tin tưởng hoàn toàn vào Teacher)
        msteps = 1        # Số lượng epoch cho phép phá hủy (Maximize). Nếu để quá cao, model sẽ hỏng hoàn toàn.
        sgda_lr=unlr
        sgda_momentum = 0.9
        sgda_weight_decay = 0.1
        epochs=4
        # ==========================================
        # 2. CHUẨN BỊ DATA LOADERS
        # ==========================================
        all_forget_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)
        
        keep_in_forget_indices = list(all_forget_indices - remove_indices)
        
        actual_forget_dataset = Subset(forget_dataset_base, list(remove_indices))        
        remaining_forget_dataset = Subset(forget_dataset_base, keep_in_forget_indices)
        
        # C. Tập Retain THỰC SỰ = Retain gốc + Phần còn lại của Forget
        actual_retain_dataset = ConcatDataset([retain_dataset_base, remaining_forget_dataset])

        # ==========================================
        # 2. CHUẨN BỊ DATA LOADERS
        # ==========================================
        retain_loader = DataLoader(actual_retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        forget_loader = DataLoader(actual_forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        # ==========================================
        # 3. KHỞI TẠO MÔ HÌNH (STUDENT & TEACHER)
        # ==========================================
        # Student: Bắt đầu từ mô hình đã nhiễm dữ liệu
        model_s = copy.deepcopy(self.target_model)
        
        # Teacher: Mô hình chuẩn mực để tham chiếu. 
        # Trong SCRUB gốc, Teacher chính là target_model đóng băng. 
        # (Nếu thực nghiệm nâng cao, bạn có thể truyền self.base_model vào đây)
        model_t = copy.deepcopy(self.target_model)
        model_t.eval() # Bắt buộc đóng băng Teacher
        
        # optimizer = optim.SGD(model_s.parameters(), lr=sgda_lr, momentum=sgda_momentum, weight_decay=sgda_weight_decay)
        optimizer = optim.Adam(model_s.parameters(), lr=sgda_lr, weight_decay=sgda_weight_decay)
        
        criterion_div = DistillKL(T)

        # ==========================================
        # 4. VÒNG LẶP MIN-MAX (THE ORCHESTRATOR)
        # ==========================================
        print(f"   [SCRUB Unlearn] Khởi động với {len(actual_retain_dataset)} Retain | {len(indices_to_remove)} Forget...")
        
        for epoch in range(1, epochs + 1):
            
            # Pha 1: Tẩy não (Maximize trên tập Forget) - Chỉ chạy trong vài bước đầu
            if epoch <= msteps:
                max_loss = self._train_distill_epoch(
                    forget_loader, model_s, model_t, optimizer, criterion_div, 
                    split="maximize", alpha=alpha, gamma=gamma
                )
            else:
                max_loss = 0.0
                
            # Pha 2: Củng cố (Minimize trên tập Retain) - Luôn chạy để bảo vệ trí nhớ
            min_loss = self._train_distill_epoch(
                retain_loader, model_s, model_t, optimizer, criterion_div, 
                split="minimize", alpha=alpha, gamma=gamma
            )
            
            print(f"   Epoch {epoch}/{epochs} | Maximize Loss: {max_loss:.4f} | Minimize Loss: {min_loss:.4f}")

        return model_s


    def neggrad_unlearn(self, retain_dataset_base, forget_dataset_base, indices_to_remove, unlr =  0.001 ,num_classes=10):
        """
        NegGrad+ Unlearn với Chance Level Clamping.
        Đã cập nhật logic gộp (Merge) Dataset chuẩn xác.
        """
        # ==========================================
        # 1. GÁN CỨNG HYPERPARAMETERS 
        # ==========================================
        epochs=5
        lr=unlr
        alpha=0.8
        
        
        chance_level = -math.log(1.0 / num_classes)
        
         # ==========================================
        # 2. CHUẨN BỊ DATA LOADERS
        # ==========================================

        all_forget_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)
        
        keep_in_forget_indices = list(all_forget_indices - remove_indices)
        
        actual_forget_dataset = Subset(forget_dataset_base, list(remove_indices))        
        remaining_forget_dataset = Subset(forget_dataset_base, keep_in_forget_indices)
        
        # C. Tập Retain THỰC SỰ = Retain gốc + Phần còn lại của Forget
        actual_retain_dataset = ConcatDataset([retain_dataset_base, remaining_forget_dataset])

        retain_loader = DataLoader(actual_retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        forget_loader = DataLoader(actual_forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

        # ==========================================
        # 3. KHỞI TẠO MÔ HÌNH VÀ VÒNG LẶP (Giữ nguyên như cũ)
        # ==========================================
        model = copy.deepcopy(self.target_model)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        print(f"   [NegGrad+] Khởi động với alpha={alpha} | Chance Level: {chance_level:.4f}")
        print(f"   [Data] Retain thực sự: {len(actual_retain_dataset)} mẫu | Forget thực sự: {len(actual_forget_dataset)} mẫu")
        
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            r_loss_sum = 0.0
            f_loss_sum = 0.0
            
            for (r_inputs, r_targets), (f_inputs, f_targets) in zip(retain_loader, cycle(forget_loader)):
                
                r_inputs, r_targets = r_inputs.to(self.device), r_targets.to(self.device)
                f_inputs, f_targets = f_inputs.to(self.device), f_targets.to(self.device)

                optimizer.zero_grad()

                r_outputs = model(r_inputs)
                f_outputs = model(f_inputs)

                r_loss = self.criterion(r_outputs, r_targets)
                f_loss = self.criterion(f_outputs, f_targets)

                # Clamping ở ngưỡng Chance Level
                f_loss_clamped = torch.clamp(f_loss, max=chance_level)

                # Công thức lõi
                loss = alpha * r_loss - (1 - alpha) * f_loss_clamped

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                r_loss_sum += r_loss.item()
                f_loss_sum += f_loss.item()

            batches = len(retain_loader)
            print(f"   Epoch {epoch}/{epochs} | Tổng Loss: {total_loss/batches:.4f} "
                  f"(Retain: {r_loss_sum/batches:.4f}, Forget (Unclamped): {f_loss_sum/batches:.4f})")
        return model

    def retrain_from_scratch(self, retain_dataset_base, forget_dataset_base, indices_to_remove, 
                         model_name, dataset_name, epochs, lr=0.1, 
                         num_channels=3, img_size=32, num_classes=10, device='cuda'):
        """
        Retrain from Scratch (Gold Standard Model):
        Xây dựng một mô hình hoàn toàn mới từ đầu (Random Initialization) 
        chỉ sử dụng phần dữ liệu SẠCH (Retain gốc + phần không bị xóa trong Forget).
        """
        # ==========================================
        # 1. TẠO TẬP DỮ LIỆU HUẤN LUYỆN SẠCH (CLEAN DATASET)
        # ==========================================
        # Lọc ra những index không nằm trong danh sách cần xóa
        all_forget_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)
        keep_in_forget_indices = list(all_forget_indices - remove_indices)
        
        # Lấy phần dữ liệu còn lại của tập Forget
        remaining_forget_dataset = Subset(forget_dataset_base, keep_in_forget_indices)
        
        # Gộp (Merge) Retain gốc và phần Forget an toàn
        clean_train_dataset = ConcatDataset([retain_dataset_base, remaining_forget_dataset])
        
        # Khởi tạo DataLoader
        clean_train_loader = DataLoader(clean_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        print(f"\n[Retrain from Scratch] Bắt đầu xây dựng Gold Standard Model...")
        print(f" - Dữ liệu Retain gốc: {len(retain_dataset_base)} mẫu")
        print(f" - Dữ liệu giữ lại từ Forget: {len(remaining_forget_dataset)} mẫu (Đã xóa vĩnh viễn {len(remove_indices)} mẫu)")
        print(f" - TỔNG DATA HUẤN LUYỆN MỚI: {len(clean_train_dataset)} mẫu")

        # ==========================================
        # 2. KHỞI TẠO MÔ HÌNH MỚI TINH (RANDOM WEIGHTS)
        # ==========================================
        model = get_custom_model(model_name, num_channels=num_channels, 
                                num_classes=num_classes, img_size=img_size)
        model = model.to(device)

        # Khởi tạo Optimizer và LR Scheduler giống với Phase 1A của bạn
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[int(epochs * 0.5), int(epochs * 0.75)], 
            gamma=0.1
        )

        # ==========================================
        # 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
        # ==========================================
        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in clean_train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs,_ = model(inputs)
                loss =  self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            scheduler.step()
            
            train_loss = running_loss / len(clean_train_loader)
            train_acc = 100. * correct / total
            
            print(f"   Epoch {epoch+1:02d}/{epochs} | Loss: {train_loss:.3f} | Acc: {train_acc:.2f}% | Time: {time.time()-t0:.1f}s")

        print(f"[Retrain from Scratch] Hoàn tất! Đã có Gold Standard Model.")
        return model



    

def get_weight_difference(model_orig, model_new):
    diff_dict = {}
    state_orig = model_orig.state_dict()
    state_new = model_new.state_dict()
    for k in state_orig.keys():
        if 'weight' in k or 'bias' in k:
            diff_dict[k] = (state_new[k] - state_orig[k]).cpu().detach()
   
    return diff_dict