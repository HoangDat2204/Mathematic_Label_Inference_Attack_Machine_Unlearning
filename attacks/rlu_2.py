import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def compute_expected_confidence(model, aux_loader, num_classes, device):
    """
    Tính toán ma trận S (expected erroneous confidence) dựa trên tập dữ liệu phụ.
    Nguồn: Eq. 6 và Eq. 8 [3][4].
    """
    model.eval()
    confidence_sums = torch.zeros(num_classes, num_classes).to(device)
    class_counts = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for images, labels in aux_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)

            for i in range(num_classes):
                idxs = (labels == i).nonzero(as_tuple=True)
                if len(idxs) > 0:
                    probs_i = probabilities[idxs]
                    confidence_sums[i] += probs_i.sum(dim=0)
                    class_counts[i] += len(idxs)

    S = torch.zeros(num_classes, num_classes).to(device)
    for n in range(num_classes):
        if class_counts[n] > 0:
            S[n] = confidence_sums[n] / class_counts[n]
    
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
        proxy_update: lr * gradient (theo yêu cầu đề bài).
    
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