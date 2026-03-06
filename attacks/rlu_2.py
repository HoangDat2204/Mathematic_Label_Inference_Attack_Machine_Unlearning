import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions.multivariate_normal import MultivariateNormal

def compute_expected_confidence(model, aux_loader, num_classes, device, M=10000):
    model.eval()
    logits_dict = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in aux_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            for i in range(num_classes):
                # Cách lấy index chuẩn hơn
                mask = (labels == i)
                if mask.any(): # Chỉ thêm nếu lớp i có tồn tại trong batch này
                    logits_dict[i].append(outputs[mask])

    S = torch.zeros(num_classes, num_classes).to(device)

    for n in range(num_classes):
        # KIỂM TRA: Chỉ xử lý nếu lớp n có ít nhất 1 mẫu
        if len(logits_dict[n]) > 0:
            logits_n = torch.cat(logits_dict[n], dim=0)
            
            # Tính mean
            mu_n = torch.mean(logits_n, dim=0)
            
            # Kiểm tra nếu mu_n chứa NaN (do lỗi data đầu vào)
            if torch.isnan(mu_n).any():
                print(f"Cảnh báo: Lớp {n} chứa NaN trong logits. Bỏ qua.")
                continue

            num_samples = logits_n.shape[0]
            
            # Tính Sigma an toàn
            if num_samples > 1:
                centered_logits = logits_n - mu_n
                sigma_n = (centered_logits.t() @ centered_logits) / (num_samples - 1)
            else:
                # Nếu chỉ có 1 mẫu, không thể tính hiệp phương sai, dùng ma trận 0
                sigma_n = torch.zeros(num_classes, num_classes).to(device)
            
            # Ổn định số học: Epsilon đủ lớn để tránh lỗi ma trận suy biến
            sigma_n += torch.eye(num_classes).to(device) * 1e-4

            try:
                dist = torch.distributions.MultivariateNormal(mu_n, sigma_n)
                sampled_logits = dist.rsample((M,))
                sampled_probs = torch.nn.functional.softmax(sampled_logits, dim=1)
                S[n] = torch.mean(sampled_probs, dim=0)
            except Exception as e:
                print(f"Lỗi tại lớp {n}: {e}. Đang dùng giá trị mặc định.")
                # Nếu lỗi (ví dụ sigma vẫn không khả nghịch), lấy softmax trực tiếp từ mu_n
                S[n] = torch.nn.functional.softmax(mu_n, dim=0)
        else:
            # Nếu lớp n hoàn toàn không có dữ liệu trong aux_loader
            # Gán xác suất đồng đều hoặc một vector mặc định để tránh S[n] toàn 0
            S[n] = torch.ones(num_classes).to(device) / num_classes
            
    return S

def construct_matrix_A(S, num_classes):
    """
    Xây dựng ma trận hệ số A từ S.
    Nguồn: Section 3.2.1 [1] và Algorithm 1 [2].
    """
    A = torch.zeros(num_classes, num_classes).to(S.device)
    for j in range(num_classes):
        # Đường chéo: tổng các S_{j,n} (với n != j)
        row_j = S[j].clone()
        row_j[j] = 0
        A[j, j] = row_j.sum()
        
        for n in range(num_classes):
            if n != j:
                # Ngoài đường chéo: -S_{n,j}
                A[n, j] = -S[n, j]
    return A

def solve_constrained_ls(A, u, num_classes, batch_size, device):
    """
    Giải bài toán tối ưu hóa: min ||Az - u||^2 với ràng buộc.
    Nguồn: Eq. 10 [1].
    """
    # FIX LỖI: Khởi tạo z trực tiếp như một leaf tensor
    # Tạo tensor giá trị trước, sau đó detach và set requires_grad
    z_init = torch.full((num_classes,), 1.0/num_classes, device=device)
    z = z_init.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([z], lr=0.01)
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).to(A.device, dtype=A.dtype)
    elif isinstance(u, torch.Tensor) and u.device != A.device:
        u = u.to(A.device)

    # 2. Do the same for 'z' just in case it is also a NumPy array 
    # (e.g., if it was output by a SciPy solver)
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).to(A.device, dtype=A.dtype)
    elif isinstance(z, torch.Tensor) and z.device != A.device:
        z = z.to(A.device)
    # Projected Gradient Descent để giải bài toán Least Squares có ràng buộc
    for _ in range(2000):
        optimizer.zero_grad()
        # Hàm mục tiêu: ||Az - u||^2
        loss = torch.norm(torch.matmul(A, z) - u)**2
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Ràng buộc 1: 0 <= z <= 1
            z.clamp_(min=0, max=1)
            # Ràng buộc 2: Tổng z = 1 (Simplex projection đơn giản hóa)
            if z.sum() > 0:
                z /= z.sum()
                
    return z.detach()

def attack_rlu_full(model_original, proxy_update, aux_loader, 
                    batch_size, lr, num_epochs=1, num_classes=10, device='cpu'):
    """
    Triển khai tấn công RLU để khôi phục nhãn từ cập nhật cục bộ.
    
    Args:
        proxy_update: lr * gradient.
    
    Nguồn: Algorithm 1 [2].
    """
    model_original = model_original.to(device)
    
    target_update = None
    # Tìm bias layer cuối
    for name in reversed(list(proxy_update.keys())):
        if 'bias' in name and proxy_update[name].shape[0] == num_classes:
            target_update = proxy_update[name].detach().cpu().numpy()
            break
            
    if target_update is None:
        print("[RLU Error] Không tìm thấy Bias lớp cuối.")
        return []

    # 2. Tính vector u
    # Bài báo định nghĩa u = Delta_b / eta [1].
    # Bài báo giả định Gradient Descent: Delta_b = -eta * gradient.
    # Đề bài cho: proxy_update = eta * gradient.
    # => Delta_b = -proxy_update.
    # => u = (-proxy_update) / eta = -grad_b / lr.
    u = -target_update / (lr*batch_size)
    # 3. Tính toán ma trận S và A tại trạng thái t (model gốc)
    S_t = compute_expected_confidence(model_original, aux_loader, num_classes, device)
    A_t = construct_matrix_A(S_t, num_classes)
    
    final_A = A_t
    # 4. Xử lý Multi-epoch (Algorithm 1 dòng 12-16 [2])
    if num_epochs > 1:
        # Tạo model tại trạng thái t+1: new_model = model + proxy_update
        model_new = copy.deepcopy(model_original)
        with torch.no_grad():
            if isinstance(proxy_update, dict):
                for name, param in model_new.named_parameters():
                    if name in proxy_update:
                        param.add_(proxy_update[name].to(device))
            elif isinstance(proxy_update, (list, tuple)):
                for param, update in zip(model_new.parameters(), proxy_update):
                    param.add_(update.to(device))
        
        # Tính S và A tại t+1
        S_next = compute_expected_confidence(model_new, aux_loader, num_classes, device)
        A_next = construct_matrix_A(S_next, num_classes)
        
        # Trung bình cộng ma trận A (Algorithm 1 dòng 13 [2])
        final_A = (A_t + A_next) / 2.0

    # 5. Giải hệ phương trình tối ưu để tìm tỉ lệ nhãn z
    # Gọi hàm đã sửa lỗi "can't optimize a non-leaf Tensor"
    z = solve_constrained_ls(final_A, u, num_classes, batch_size, device)
    
    # 6. Khôi phục số lượng nhãn (Algorithm 1 dòng 11/14 [2])
    # N(t) = round(|B| * z)
    predicted_counts = torch.round(z * batch_size).int()
    
    # Điều chỉnh sai số làm tròn để tổng bằng batch_size
    diff = batch_size - predicted_counts.sum().item()
    if diff != 0:
        # Cộng phần dư vào lớp có xác suất cao nhất
        idx_max = torch.argmax(z)
        predicted_counts[idx_max] += diff

    # 7. Tạo danh sách nhãn
    predicted_labels = []
    for cls_idx in range(num_classes):
        count = predicted_counts[cls_idx].item()
        if count > 0:
            predicted_labels.extend([cls_idx] * count)
            
    return sorted(predicted_labels)