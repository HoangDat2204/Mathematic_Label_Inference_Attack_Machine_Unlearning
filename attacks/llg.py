# File: attacks/llg.py
import torch
import numpy as np

def attack_llg(proxy_gradients, num_classes=10, batch_size=1):
    """
    Standard LLG Attack (Algorithm 1 from Paper 2110.09074).
    Tuân thủ Property 1: Gradient của target class là số ÂM.
    
    Quy trình:
    1. Lấy Gradient Vector (Bias hoặc Weight sum).
    2. Ước lượng impact 'm' từ các giá trị âm.
    3. Lặp batch_size lần: Tìm min, thêm label, trừ impact.
    """
    
    # --- BƯỚC 1: TRÍCH XUẤT GRADIENT VECTOR ---
    target_grad = None
    
    # Ưu tiên 1: Bias Gradient (Property 1 thể hiện rõ nhất ở Bias: p - y)
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            target_grad = proxy_gradients[name].detach().clone()
            break
            
    # Ưu tiên 2: Weight Gradient (Sum over features)
    if target_grad is None:
        for name in reversed(list(proxy_gradients.keys())):
            if 'weight' in name and len(proxy_gradients[name].shape) == 2:
                if proxy_gradients[name].shape[0] == num_classes:
                    # Sum theo chiều feature để ra vector [num_classes]
                    # Lý do: Property 1 phát biểu trên dot product w*h, sum lại bảo toàn dấu
                    target_grad = torch.sum(proxy_gradients[name], dim=1).detach().clone()
                    break
    
    if target_grad is None:
        return []

    # --- BƯỚC 2: XỬ LÝ DẤU (SIGN CORRECTION) ---
    # Unlearning Proxy = W_old - W_new = - lr * Gradient
    # Bài báo yêu cầu Gradient thật.
    # => Gradient_Paper = - Proxy
    gradients = target_grad 

    # --- BƯỚC 3: THUẬT TOÁN 1 (ITERATIVE REMOVAL) ---
    predicted_labels = []
    
    # 3a. Ước lượng tham số impact 'm' (mean impact parameter)
    # Theo bài báo: m được ước lượng bằng trung bình của các gradient âm (confirmed targets)
    # Tìm các phần tử âm
    negative_indices = torch.where(gradients < 0)[0]
    
    if len(negative_indices) > 0:
        # Sửa đoạn tính m trong llg.py
        sum_neg = torch.sum(gradients[negative_indices])
        # Giả sử num_classes = 10, batch_size là input
        m = (sum_neg / batch_size) * (1 + 1/num_classes)
    else:
        # Fallback: Nếu nhiễu quá lớn khiến không có số âm nào (hiếm gặp trong Unlearning chuẩn)
        # Ta lấy giá trị nhỏ nhất làm m
        m = torch.min(gradients)

    # Đảm bảo m luôn âm để phép trừ impact hoạt động đúng logic (trừ số âm = cộng dương)
    if m > 0: m = -m 
    # 3b. Vòng lặp trích xuất (Iterative Extraction)
    # Copy để không ảnh hưởng dữ liệu gốc
    g_iter = gradients.clone()
    
    for _ in range(batch_size):
        # 1. Tìm nhãn i có gradient nhỏ nhất (Min - Âm nhất)
        # Property 1: Target labels have negative gradients
        min_val, min_idx = torch.min(g_iter, dim=0)
        idx = min_idx.item()
        
        # 2. Ghi nhận nhãn
        predicted_labels.append(idx)
        
        # 3. Cập nhật gradient: g_i = g_i - m
        # Vì m là số âm (đại diện cho impact của 1 sample),
        # Trừ đi m tức là cộng một lượng dương -> Làm g_i tiến về 0 hoặc dương
        # Điều này mô phỏng việc "gỡ bỏ" 1 sample khỏi gradient tổng hợp
        g_iter[idx] = g_iter[idx] - m
        
    return sorted(predicted_labels)