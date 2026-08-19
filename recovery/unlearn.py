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


def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss


class DistillKL(nn.Module):
    """Kullback-Leibler Divergence với Temperature Scaling"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # Sử dụng reduction='sum' kết hợp chia cho batch_size tương đương với code gốc
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class Unlearner:
    def __init__(self, target_model, base_model, device='cuda'):
        self.target_model = copy.deepcopy(target_model)
        self.base_model = base_model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    

    def _train_distill_epoch(self, loader, model_s, model_t, optimizer, criterion_div, split, alpha, gamma, T):
        model_s.train()
        model_t.eval()
        total_loss = 0.0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            logit_s = model_s(inputs) 
            with torch.no_grad():
                logit_t = model_t(inputs)
                
            loss_cls = self.criterion(logit_s, targets)
            
            # CHÌA KHÓA TOÁN HỌC: Nhân KL Divergence loss với T^2
            loss_div = criterion_div(logit_s, logit_t) * (T ** 2)

            if split == "minimize":
                # Kết hợp Cross-Entropy gốc và KL Distillation để bảo toàn tri thức tập Retain
                loss = gamma * loss_cls + alpha * loss_div
            elif split == "maximize":
                # Pha phá hủy tri thức tập Forget: đẩy khoảng cách phân phối ra xa
                loss = -loss_div
            
            optimizer.zero_grad()
            loss.backward()
            
            # Khống chế độ lớn gradient cực kỳ nghiêm ngặt
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0


    
    def approximate_unlearn(self, list_of_batches, retain_loader, test_loader, forget_dataset, target_indices, 
                           lr=0.01, batch_size=256, local_epochs=1):
        """
        Thực hiện Unlearning và đánh giá hiệu năng Trước/Sau trên các tập dữ liệu.
        Thuật toán unlearn gốc được giữ nguyên bản tuyệt đối.
        """
        # --- [ĐÁNH GIÁ] CHUẨN BỊ TẬP FORGET CÒN LẠI (FORGET_SET \ {FORGET_BATCH}) ---
        all_forget_indices = set(range(len(forget_dataset)))
        remaining_indices = list(all_forget_indices - set(target_indices))
        remaining_forget_subset = Subset(forget_dataset, remaining_indices)
        remaining_forget_loader = DataLoader(
            remaining_forget_subset, batch_size=256, shuffle=False, num_workers=2
        )
        
        # Hàm hỗ trợ đánh giá accuracy của một Loader
        def evaluate_acc(eval_model, loader):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0

        # Hàm hỗ trợ đánh giá accuracy của batch ảnh đang unlearn (list_of_batches)
        def evaluate_batch_input_acc(eval_model, batches):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in batches:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0
        
        acc_batch_before = evaluate_batch_input_acc(self.target_model, list_of_batches)


        # =========================================================================
        # --- THUẬT TOÁN UNLEARN GỐC CỦA BẠN (GIỮ NGUYÊN BẢN TUYỆT ĐỐI) ---
        # =========================================================================
        # 1. Trích xuất toàn bộ dữ liệu gộp từ batch_input
        all_images, all_labels = list_of_batches[0]
        num_samples = all_images.size(0)
        
        # Đảm bảo batch_size không vượt quá tổng số lượng mẫu có sẵn
        actual_batch_size = min(batch_size, num_samples)
        
        # Luôn khởi đầu từ Target Model (M_finetuned)
        model = copy.deepcopy(self.target_model)
        model.train()
        
        # Khởi tạo Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Xác định tham số bias lớp cuối cùng để theo dõi sự biến động
        last_bias_param = None
        last_bias_name = None
        for name, param in reversed(list(model.named_parameters())):
            if 'bias' in name:
                last_bias_param = param
                last_bias_name = name
                break

        print(f"\n[BẮT ĐẦU UNLEARN] Tổng số mẫu: {num_samples} | Batch Size: {actual_batch_size} | Epochs: {local_epochs}")

        # --- VÒNG LẶP CHÍNH: LOCAL EPOCHS ---
        for epoch in range(local_epochs):
            # Lưu lại trạng thái bias ở đầu mỗi epoch để tính delta epoch
            if last_bias_param is not None:
                bias_start_epoch = last_bias_param.detach().clone()
                
            # Xáo trộn ngẫu nhiên các chỉ số (indices) ở đầu mỗi epoch để tăng tính ổn định
            shuffled_indices = torch.randperm(num_samples)
            epoch_images = all_images[shuffled_indices]
            epoch_labels = all_labels[shuffled_indices]
            
            # Tính toán tổng số lượng batch nhỏ trong epoch này
            num_batches = (num_samples + actual_batch_size - 1) // actual_batch_size
            epoch_loss = 0.0
            
            # --- VÒNG LẶP PHỤ: DUYỆT TỪNG MINI-BATCH ---
            for b in range(num_batches):
                start_idx = b * actual_batch_size
                end_idx = min((b + 1) * actual_batch_size, num_samples)
                
                # Cắt lát (slice) lấy mini-batch tương ứng
                batch_imgs = epoch_images[start_idx:end_idx].to(self.device)
                batch_lbls = epoch_labels[start_idx:end_idx].to(self.device)

                outputs = model(batch_imgs)
                probabilities = F.softmax(outputs, dim=1) 
                
                # Cập nhật trọng số bằng Gradient Ascent
                loss = -self.criterion(outputs, batch_lbls) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # --- ĐÁNH GIÁ VÀ IN RA BIẾN ĐỘNG SAU MỖI EPOCH ---
            print(f"\n" + "="*70)
            print(f" EPOCH CỤC BỘ {epoch+1:02d} / {local_epochs:02d} hoàn tất ".center(70, "-"))
            print(f"• Loss trung bình của Epoch: {epoch_loss / num_batches:.4f}")
            
            if last_bias_param is not None:
                bias_end_epoch = last_bias_param.detach().clone()
                delta_bias = bias_end_epoch - bias_start_epoch
                delta_norm = torch.norm(delta_bias).item()
                
                print(f"• Delta Bias lớp '{last_bias_name}' sau Epoch này:\n  {delta_bias.cpu().numpy()}")
                print(f"• Khoảng cách dịch chuyển bias (L2 Norm): {delta_norm:.8f}")
            print("="*70)
        # =========================================================================

        # --- [ĐÁNH GIÁ] ĐO ĐẠC SAU KHI CHẠY UNLEARN (AFTER) ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG SAU KHI UNLEARN (AFTER) ".center(80, "="))
        
        acc_retain_after = evaluate_acc(model, retain_loader)
        acc_test_after = evaluate_acc(model, test_loader)
        acc_rem_forget_after = evaluate_acc(model, remaining_forget_loader)
        acc_batch_after = evaluate_batch_input_acc(model, list_of_batches)
        
        print(f"• Retain Accuracy (Cần giữ lại)       : {acc_retain_after:.2f}% ")
        print(f"• Test Accuracy (Tổng quát chung)       : {acc_test_after:.2f}%  ")
        print(f"• Remaining Forget Acc (Giữ lại bổ trợ) : {acc_rem_forget_after:.2f}% ")
        print(f"• Target Forget Batch Acc (ĐÃ QUÊN)     : {acc_batch_after:.2f}%  ")
        print("="*80 + "\n")
        
        return model, acc_retain_after, acc_test_after, acc_rem_forget_after, acc_batch_after, acc_batch_before

    def approximate_unlearn_noise(self, list_of_batches, retain_loader, test_loader, forget_dataset, target_indices, 
                           lr=0.01, batch_size=256, local_epochs=1, noise_var=0.001):
        """
        Thực hiện Unlearning bằng Gradient Ascent + Gaussian Noise.
        Giống approximate_unlearn nhưng cộng thêm nhiễu Gaussian N(0, noise_var) 
        vào gradient tại mỗi bước cập nhật ascent.
        
        Args:
            list_of_batches: List chứa tuple [(images, labels)] của batch cần quên.
            retain_loader: DataLoader của tập Retain.
            test_loader: DataLoader của tập Test.
            forget_dataset: Dataset gốc của tập Forget.
            target_indices: Các chỉ số trong forget_dataset cần quên.
            lr: Learning rate cho Gradient Ascent.
            batch_size: Kích thước mini-batch.
            local_epochs: Số epoch cục bộ.
            noise_var: Phương sai (σ²) của nhiễu Gaussian. Độ lệch chuẩn σ = √noise_var.
        """
        # --- [ĐÁNH GIÁ] CHUẨN BỊ TẬP FORGET CÒN LẠI (FORGET_SET \ {FORGET_BATCH}) ---
        all_forget_indices = set(range(len(forget_dataset)))
        remaining_indices = list(all_forget_indices - set(target_indices))
        remaining_forget_subset = Subset(forget_dataset, remaining_indices)
        remaining_forget_loader = DataLoader(
            remaining_forget_subset, batch_size=256, shuffle=False, num_workers=2
        )
        
        # Hàm hỗ trợ đánh giá accuracy của một Loader
        def evaluate_acc(eval_model, loader):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0

        # Hàm hỗ trợ đánh giá accuracy của batch ảnh đang unlearn (list_of_batches)
        def evaluate_batch_input_acc(eval_model, batches):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in batches:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0
        
        acc_batch_before = evaluate_batch_input_acc(self.target_model, list_of_batches)

        # =========================================================================
        # --- THUẬT TOÁN UNLEARN: GRADIENT ASCENT + GAUSSIAN NOISE ---
        # =========================================================================
        # 1. Trích xuất toàn bộ dữ liệu gộp từ batch_input
        all_images, all_labels = list_of_batches[0]
        num_samples = all_images.size(0)
        
        # Đảm bảo batch_size không vượt quá tổng số lượng mẫu có sẵn
        actual_batch_size = min(batch_size, num_samples)
        
        # Luôn khởi đầu từ Target Model (M_finetuned)
        model = copy.deepcopy(self.target_model)
        model.train()
        
        # Khởi tạo Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Tính toán độ lệch chuẩn từ phương sai: σ = √(σ²)
        noise_std = noise_var ** 0.5
        
        # Xác định tham số bias lớp cuối cùng để theo dõi sự biến động
        last_bias_param = None
        last_bias_name = None
        for name, param in reversed(list(model.named_parameters())):
            if 'bias' in name:
                last_bias_param = param
                last_bias_name = name
                break

        print(f"\n[BẮT ĐẦU UNLEARN + NOISE] Tổng số mẫu: {num_samples} | Batch Size: {actual_batch_size} | Epochs: {local_epochs} | Noise Var: {noise_var} (σ={noise_std:.6f})")

        # --- VÒNG LẶP CHÍNH: LOCAL EPOCHS ---
        for epoch in range(local_epochs):
            # Lưu lại trạng thái bias ở đầu mỗi epoch để tính delta epoch
            if last_bias_param is not None:
                bias_start_epoch = last_bias_param.detach().clone()
                
            # Xáo trộn ngẫu nhiên các chỉ số (indices) ở đầu mỗi epoch để tăng tính ổn định
            shuffled_indices = torch.randperm(num_samples)
            epoch_images = all_images[shuffled_indices]
            epoch_labels = all_labels[shuffled_indices]
            
            # Tính toán tổng số lượng batch nhỏ trong epoch này
            num_batches = (num_samples + actual_batch_size - 1) // actual_batch_size
            epoch_loss = 0.0
            
            # --- VÒNG LẶP PHỤ: DUYỆT TỪNG MINI-BATCH ---
            for b in range(num_batches):
                start_idx = b * actual_batch_size
                end_idx = min((b + 1) * actual_batch_size, num_samples)
                
                # Cắt lát (slice) lấy mini-batch tương ứng
                batch_imgs = epoch_images[start_idx:end_idx].to(self.device)
                batch_lbls = epoch_labels[start_idx:end_idx].to(self.device)

                outputs = model(batch_imgs)
                probabilities = F.softmax(outputs, dim=1) 
                
                # Cập nhật trọng số bằng Gradient Ascent
                loss = -self.criterion(outputs, batch_lbls) 
                
                optimizer.zero_grad()
                loss.backward()
                
                # ============================================================
                # === GAUSSIAN NOISE INJECTION ===
                # Cộng nhiễu N(0, σ²) vào gradient trước khi cập nhật trọng số
                # Mục đích: Tăng tính riêng tư (differential privacy style)
                # hoặc thay đổi tín hiệu gradient để nghiên cứu ảnh hưởng 
                # lên khả năng tấn công suy diễn nhãn.
                # ============================================================
                if noise_var > 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * noise_std
                            param.grad.add_(noise)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            # --- ĐÁNH GIÁ VÀ IN RA BIẾN ĐỘNG SAU MỖI EPOCH ---
            print(f"\n" + "="*70)
            print(f" EPOCH CỤC BỘ {epoch+1:02d} / {local_epochs:02d} hoàn tất (Noise σ²={noise_var}) ".center(70, "-"))
            print(f"• Loss trung bình của Epoch: {epoch_loss / num_batches:.4f}")
            
            if last_bias_param is not None:
                bias_end_epoch = last_bias_param.detach().clone()
                delta_bias = bias_end_epoch - bias_start_epoch
                delta_norm = torch.norm(delta_bias).item()
                
                print(f"• Delta Bias lớp '{last_bias_name}' sau Epoch này:\n  {delta_bias.cpu().numpy()}")
                print(f"• Khoảng cách dịch chuyển bias (L2 Norm): {delta_norm:.8f}")
            print("="*70)
        # =========================================================================

        # --- [ĐÁNH GIÁ] ĐO ĐẠC SAU KHI CHẠY UNLEARN (AFTER) ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG SAU KHI UNLEARN + NOISE (AFTER) ".center(80, "="))
        
        acc_retain_after = evaluate_acc(model, retain_loader)
        acc_test_after = evaluate_acc(model, test_loader)
        acc_rem_forget_after = evaluate_acc(model, remaining_forget_loader)
        acc_batch_after = evaluate_batch_input_acc(model, list_of_batches)
        
        print(f"• Retain Accuracy (Cần giữ lại)       : {acc_retain_after:.2f}% ")
        print(f"• Test Accuracy (Tổng quát chung)       : {acc_test_after:.2f}%  ")
        print(f"• Remaining Forget Acc (Giữ lại bổ trợ) : {acc_rem_forget_after:.2f}% ")
        print(f"• Target Forget Batch Acc (ĐÃ QUÊN)     : {acc_batch_after:.2f}%  ")
        print("="*80 + "\n")
        
        return model, acc_retain_after, acc_test_after, acc_rem_forget_after, acc_batch_after, acc_batch_before

    # def approximate_unlearn(self, list_of_batches, lr=0.01, batch_size=64, local_epochs=1):
    #     """
    #     Thực hiện Unlearning bằng cách chia nhỏ cục dữ liệu lớn thành các mini-batches 
    #     và lặp lại qua nhiều epochs cục bộ.
        
    #     Args:
    #         list_of_batches: List chứa 1 tuple duy nhất [(all_images, all_labels)]
    #         lr: Học suất (Learning rate) cho Gradient Ascent.
    #         batch_size: Kích thước batch nhỏ mong muốn để thực hiện từng bước nhảy.
    #         local_epochs: Số lần lặp lại (epochs) trên toàn bộ lượng dữ liệu này.
    #     """
    #     # 1. Trích xuất toàn bộ dữ liệu gộp từ batch_input
    #     all_images, all_labels = list_of_batches[0]
    #     num_samples = all_images.size(0)
        
    #     # Đảm bảo batch_size không vượt quá tổng số lượng mẫu có sẵn
    #     actual_batch_size = min(batch_size, num_samples)
        
    #     # Luôn khởi đầu từ Target Model (M_finetuned)
    #     model = copy.deepcopy(self.target_model)
    #     model.train()
        
    #     # Khởi tạo Optimizer
    #     optimizer = optim.SGD(model.parameters(), lr=lr)
        
    #     # Xác định tham số bias lớp cuối cùng để theo dõi sự biến động
    #     last_bias_param = None
    #     last_bias_name = None
    #     for name, param in reversed(list(model.named_parameters())):
    #         if 'bias' in name:
    #             last_bias_param = param
    #             last_bias_name = name
    #             break

    #     print(f"\n[BẮT ĐẦU UNLEARN] Tổng số mẫu: {num_samples} | Batch Size: {actual_batch_size} | Epochs: {local_epochs}")
    #     # if last_bias_param is not None:
    #     #     print(f"-> Đang theo dõi lớp bias cuối cùng: '{last_bias_name}'")

    #     # --- VÒNG LẶP CHÍNH: LOCAL EPOCHS ---
    #     for epoch in range(local_epochs):
    #         # Lưu lại trạng thái bias ở đầu mỗi epoch để tính delta epoch
    #         if last_bias_param is not None:
    #             bias_start_epoch = last_bias_param.detach().clone()
                
    #         # Xáo trộn ngẫu nhiên các chỉ số (indices) ở đầu mỗi epoch để tăng tính ổn định
    #         shuffled_indices = torch.randperm(num_samples)
    #         epoch_images = all_images[shuffled_indices]
    #         epoch_labels = all_labels[shuffled_indices]
            
    #         # Tính toán tổng số lượng batch nhỏ trong epoch này
    #         num_batches = (num_samples + actual_batch_size - 1) // actual_batch_size
    #         epoch_loss = 0.0
            
    #         # --- VÒNG LẶP PHỤ: DUYỆT TỪNG MINI-BATCH ---
    #         for b in range(num_batches):
    #             start_idx = b * actual_batch_size
    #             end_idx = min((b + 1) * actual_batch_size, num_samples)
                
    #             # Cắt lát (slice) lấy mini-batch tương ứng
    #             batch_imgs = epoch_images[start_idx:end_idx].to(self.device)
    #             batch_lbls = epoch_labels[start_idx:end_idx].to(self.device)

               
    #             outputs = model(batch_imgs)
    #             probabilities = F.softmax(outputs, dim=1) 
    #             # print(probabilities)
                
    #             # Cập nhật trọng số bằng Gradient Ascent
    #             loss = -self.criterion(outputs, batch_lbls) 
                
                
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             epoch_loss += loss.item()
                
    #         # --- ĐÁNH GIÁ VÀ IN RA BIẾN ĐỘNG SAU MỖI EPOCH ---
    #         # print(f"\n" + "="*70)
    #         # print(f" EPOCH CỤC BỘ {epoch+1:02d} / {local_epochs:02d} hoàn tất ".center(70, "-"))
    #         # print(f"• Loss trung bình của Epoch: {epoch_loss / num_batches:.4f}")
            
    #         # if last_bias_param is not None:
    #         #     bias_end_epoch = last_bias_param.detach().clone()
    #         #     delta_bias = bias_end_epoch - bias_start_epoch
    #         #     delta_norm = torch.norm(delta_bias).item()
                
    #         #     print(f"• Delta Bias lớp '{last_bias_name}' sau Epoch này:\n  {delta_bias.cpu().numpy()}")
    #         #     print(f"• Khoảng cách dịch chuyển bias (L2 Norm): {delta_norm:.8f}")
    #         # print("="*70)
            
    #     return model

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

    

    def scrub_unlearn(self, retain_dataset_base, forget_dataset_base, indices_to_remove, test_loader):
        """
        Thuật toán SCRUB SOTA chuẩn hóa toán học.
        """
        T = 5             # Softmax Temperature
        alpha = 5         # Trọng số cho KL Divergence
        gamma = 1         # Trọng số cho Cross-Entropy
        msteps = 3        # Số epoch chạy pha Maximize
        epochs = 6
        
        # 1. PHÂN CHIA DỮ LIỆU
        all_forget_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)
        keep_in_forget_indices = list(all_forget_indices - remove_indices)
        
        actual_forget_dataset = Subset(forget_dataset_base, list(remove_indices))        
        remaining_forget_dataset = Subset(forget_dataset_base, keep_in_forget_indices)
        actual_retain_dataset = ConcatDataset([retain_dataset_base, remaining_forget_dataset])
    
        # Tăng batch_size của forget_loader lên một chút để làm mịn gradient, tránh nhiễu gây sụp đổ
        retain_loader = DataLoader(actual_retain_dataset, batch_size=64, shuffle=True)
        forget_loader = DataLoader(actual_forget_dataset, batch_size=16, shuffle=True) 
    
        eval_retain_loader = DataLoader(actual_retain_dataset, batch_size=256, shuffle=False)
        eval_keep_loader = DataLoader(remaining_forget_dataset, batch_size=256, shuffle=False)
        eval_forget_loader = DataLoader(actual_forget_dataset, batch_size=256, shuffle=False)
    
        def evaluate_acc(eval_model, loader):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0
    
        # --- ĐO BEFORE ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG TRƯỚC KHI UNLEARN (BEFORE - SCRUB) ".center(80, "="))
        acc_retain_before = evaluate_acc(self.target_model, eval_retain_loader)
        acc_keep_before = evaluate_acc(self.target_model, eval_keep_loader)
        acc_forget_before = evaluate_acc(self.target_model, eval_forget_loader)
        acc_test_before = evaluate_acc(self.target_model, test_loader)
        print(f"• Retain Accuracy: {acc_retain_before:.2f}%")
        print(f"• Keep Forget Acc: {acc_keep_before:.2f}%")
        print(f"• Deleted Forget Acc: {acc_forget_before:.2f}%")
        print(f"• Test Accuracy: {acc_test_before:.2f}%")
        print("="*80)
    
        model_s = copy.deepcopy(self.target_model)
        model_t = copy.deepcopy(self.target_model)
        
        # CHÌA KHÓA: Sử dụng 2 Optimizers với LR khác nhau
        # Pha Maximize (phá hủy) bắt buộc phải dùng LR cực kỳ nhỏ để tránh nổ mô hình
        optimizer_max = optim.SGD(model_s.parameters(), lr=5e-6, momentum=0.9, weight_decay=5e-4)
        # Pha Minimize (tái hấp thu tri thức) dùng LR thông thường
        optimizer_min = optim.SGD(model_s.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
        
        criterion_div = DistillKL(T)
    
        print(f"   [SCRUB Unlearn] Khởi động với {len(actual_retain_dataset)} Retain | {len(indices_to_remove)} Forget...")
        
        for epoch in range(1, epochs + 1):
            # Pha 1: Tẩy não (Maximize)
            if epoch <= msteps:
                max_loss = self._train_distill_epoch(
                    forget_loader, model_s, model_t, optimizer_max, criterion_div, 
                    split="maximize", alpha=alpha, gamma=gamma, T=T
                )
            else:
                max_loss = 0.0
                
            # Pha 2: Củng cố (Minimize)
            min_loss = self._train_distill_epoch(
                retain_loader, model_s, model_t, optimizer_min, criterion_div, 
                split="minimize", alpha=alpha, gamma=gamma, T=T
            )
            
            print(f"   Epoch {epoch}/{epochs} | Maximize Loss: {max_loss:.4f} | Minimize Loss: {min_loss:.4f}")
    
        # --- ĐO AFTER ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG SAU KHI UNLEARN (AFTER - SCRUB) ".center(80, "="))
        acc_retain_after = evaluate_acc(model_s, eval_retain_loader)
        acc_keep_after = evaluate_acc(model_s, eval_keep_loader)
        acc_forget_after = evaluate_acc(model_s, eval_forget_loader)
        acc_test_after = evaluate_acc(model_s, test_loader)
        
        print(f"• Retain Accuracy: {acc_retain_after:.2f}%  (Thay đổi: {acc_retain_after - acc_retain_before:+.2f}%)")
        print(f"• Keep Forget Acc: {acc_keep_after:.2f}%  (Thay đổi: {acc_keep_after - acc_keep_before:+.2f}%)")
        print(f"• Deleted Forget Acc: {acc_forget_after:.2f}%  (Thay đổi: {acc_forget_after - acc_forget_before:+.2f}%)")
        print(f"• Test Accuracy: {acc_test_after:.2f}%  (Thay đổi: {acc_test_after - acc_test_before:+.2f}%)")
        print("="*80 + "\n")
    
        return model_s


    def neggrad_unlearn(self, retain_dataset_base, forget_dataset_base, indices_to_remove, test_loader, num_classes=10):
        """
        NegGrad+ Unlearn với Chance Level Clamping và Đánh giá hiệu năng Trước & Sau Unlearn.
        Thuật toán huấn luyện ngược được giữ nguyên bản 100%.
        """
        # ==========================================
        # 1. GÁN CỨNG HYPERPARAMETERS (Giữ nguyên)
        # ==========================================
        epochs = 10
        lr = 0.01
        alpha = 0.8
        chance_level = -math.log(1.0 / num_classes)
        weight_decay = 0.0
    
        # ==========================================
        # 2. CHUẨN BỊ DATA LOADERS (Giữ nguyên)
        # ==========================================
        all_forget_indices = set(range(len(forget_dataset_base)))
        remove_indices = set(indices_to_remove)  
        keep_in_forget_indices = list(all_forget_indices - remove_indices)
        actual_forget_dataset = Subset(forget_dataset_base, list(remove_indices))        
        remaining_forget_dataset = Subset(forget_dataset_base, keep_in_forget_indices)
    
        # Tập Retain THỰC SỰ = Retain gốc + Phần còn lại của Forget
        actual_retain_dataset = ConcatDataset([retain_dataset_base, remaining_forget_dataset])
        retain_loader = DataLoader(actual_retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        forget_loader = DataLoader(actual_forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
        # --- [ĐÁNH GIÁ] CHUẨN BỊ CÁC LOADER DÙNG RIÊNG CHO ĐO ĐẠC ---
        eval_retain_loader = DataLoader(actual_retain_dataset, batch_size=256, shuffle=False, num_workers=2)
        eval_forget_loader = DataLoader(actual_forget_dataset, batch_size=256, shuffle=False, num_workers=2)
    
        # Hàm hỗ trợ đo Accuracy nhanh của một Loader
        def evaluate_acc(eval_model, loader):
            eval_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = eval_model(imgs)
                    _, predicted = outputs.max(1)
                    total += lbls.size(0)
                    correct += predicted.eq(lbls).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0
    
        # --- [ĐÁNH GIÁ] ĐO ĐẠC TRƯỚC KHI CHẠY UNLEARN (BEFORE) ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG TRƯỚC KHI UNLEARN (BEFORE - NEGGRAD+) ".center(80, "="))
        
        acc_retain_before = evaluate_acc(self.target_model, eval_retain_loader)
        acc_forget_before = evaluate_acc(self.target_model, eval_forget_loader)
        acc_test_before = evaluate_acc(self.target_model, test_loader)
        
        print(f"• Retain Accuracy (Huấn luyện thực tế giữ lại): {acc_retain_before:.2f}%")
        print(f"• Forget Accuracy (Thực tế cần xóa hoàn toàn) : {acc_forget_before:.2f}%")
        print(f"• Test Accuracy (Kiểm thử tổng quát chung)     : {acc_test_before:.2f}%")
        print("="*80)
    
        # ==========================================
        # 3. KHỞI TẠO MÔ HÌNH VÀ VÒNG LẶP (Giữ nguyên bản 100%)
        # ==========================================
        model = copy.deepcopy(self.target_model)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
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
                loss = alpha * (r_loss + l2_penalty(self.target_model, model, weight_decay)) - (1 - alpha) * f_loss_clamped
    
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                r_loss_sum += r_loss.item()
                f_loss_sum += f_loss.item()
    
            batches = len(retain_loader)
            print(f"   Epoch {epoch}/{epochs} | Tổng Loss: {total_loss/batches:.4f} "
                  f"(Retain: {r_loss_sum/batches:.4f}, Forget (Unclamped): {f_loss_sum/batches:.4f})")
    
        # --- [ĐÁNH GIÁ] ĐO ĐẠC SAU KHI CHẠY UNLEARN (AFTER) ---
        print("\n" + "="*80)
        print(" ĐÁNH GIÁ HIỆU NĂNG SAU KHI UNLEARN (AFTER - NEGGRAD+) ".center(80, "="))
        
        acc_retain_after = evaluate_acc(model, eval_retain_loader)
        acc_forget_after = evaluate_acc(model, eval_forget_loader)
        acc_test_after = evaluate_acc(model, test_loader)
        
        print(f"• Retain Accuracy (Huấn luyện thực tế giữ lại): {acc_retain_after:.2f}%  (Thay đổi: {acc_retain_after - acc_retain_before:+.2f}%)")
        print(f"• Forget Accuracy (Thực tế đã xóa hoàn toàn) : {acc_forget_after:.2f}%  (Thay đổi: {acc_forget_after - acc_forget_before:+.2f}%)")
        print(f"• Test Accuracy (Kiểm thử tổng quát chung)     : {acc_test_after:.2f}%  (Thay đổi: {acc_test_after - acc_test_before:+.2f}%)")
        print("="*80 + "\n")
        
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
                outputs = model(inputs)
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