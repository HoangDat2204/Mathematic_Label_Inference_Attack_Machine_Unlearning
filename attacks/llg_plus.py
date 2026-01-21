# File: attacks/llg_plus.py
import torch
import torch.nn as nn
import numpy as np

def compute_impact_stats(model, aux_loader, num_classes, device):
    """
    Tính tham số m (Mean Impact) và s (Mean Offset) từ Aux Data.
    Theo định nghĩa bài báo:
    - Gradient g_i = Sum(Row_i của dL/dW).
    - m: Giá trị trung bình của g_i khi class i là nhãn ĐÚNG (Target).
         Theo lý thuyết m sẽ là số ÂM (ví dụ -1.5).
    - s: Giá trị trung bình của g_j khi class j là nhãn SAI (Non-target).
         Theo lý thuyết s sẽ là số DƯƠNG nhỏ hoặc xấp xỉ 0.
    """
    model.eval()
    
    impacts = [] # List các giá trị g_target
    offsets = [] # List các giá trị g_non_target
    
    print("   [LLG+] Estimating 'm' (Impact) and 's' (Offset) from Aux Data...")
    
    # Hook để lấy feature (h) phục vụ tính gradient chính xác
    # dL/dW = (p-y) * h^T => Sum(dL/dW) = (p-y) * Sum(h)
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features['feat'] = input[0].detach()
        return hook

    # Tìm lớp Linear cuối
    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            final_layer = module
    
    if final_layer is None: return 0.0, 0.0 # Fail safe

    handle = final_layer.register_forward_hook(get_features('feat'))

    for images, labels in aux_loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        
        outputs = model(images)
        feat = features['feat'] # [Batch, FeatDim]
        
        probs = torch.softmax(outputs, dim=1) # [Batch, NumClasses]
        
        # One-hot
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Error term: (p - y) [Batch, NumClasses]
        error = probs - targets_one_hot
        
        # Tính g_scalar cho từng sample trong batch
        # g_i = Sum(dL/dW_i) = Sum_over_feat((p_i - y_i) * h) 
        #     = (p_i - y_i) * Sum(h)
        
        sum_h = torch.sum(feat, dim=1) # [Batch] - Tổng giá trị feature
        
        # Ma trận Gradient vô hướng [Batch, NumClasses]
        # Mỗi hàng là vector gradient (đã sum) của 1 sample
        grads_scalar = error * sum_h.unsqueeze(1) 
        
        # Tách Impact (đúng nhãn) và Offset (sai nhãn)
        for i in range(len(labels)):
            lbl = labels[i].item()
            
            # Impact: giá trị tại cột label đúng
            val_impact = grads_scalar[i, lbl].item()
            impacts.append(val_impact)
            
            # Offset: giá trị tại các cột label sai
            # Mask chọn các cột khác lbl
            row = grads_scalar[i].clone()
            # Xóa phần tử đúng để tính trung bình các phần tử sai
            # Cách đơn giản: sum(row) - val_impact chia cho (C-1)
            # Hoặc lấy mẫu random. Ở đây ta lấy trung bình của các non-target
            val_offset = (torch.sum(row) - val_impact) / (num_classes - 1)
            offsets.append(val_offset.item())

    handle.remove()
    
    # Tính trung bình
    m = np.mean(impacts)
    s = np.mean(offsets)
    
    print(f"   [LLG+] Estimated m={m:.4f} (Expect < 0), s={s:.4f}")
    return m, s

def attack_llg_plus(proxy_gradients, m_impact, s_offset, batch_size, num_classes=10):
    """
    LLG+ Attack (Algorithm 1 Improved).
    Input:
        proxy_gradients: Dict chứa gradients từ Unlearning.
        m_impact: Giá trị impact trung bình (được tính từ Aux).
        s_offset: Giá trị offset trung bình (được tính từ Aux).
    """
    
    # 1. Trích xuất Gradient Weights và tính Sum (Scalar Gradient)
    grad_vector = None
    
    # Tìm weight layer cuối
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                w_grad = proxy_gradients[name]
                # Tính tổng theo chiều feature: g_i = Sum(W_i)
                grad_vector = torch.sum(w_grad, dim=1).detach().clone()
                break
    
    if grad_vector is None: return []

    # 2. Xử lý Dấu (Quan trọng!)
    # Unlearning Proxy = - Gradient_Paper
    # Vì Unlearn là Gradient Ascent, còn bài báo dùng Gradient Descent.
    gradients = grad_vector 

    
    # Hiệu chỉnh cơ bản: Trừ đi offset tích lũy (Optional, giúp làm sạch nhiễu nền)
    # G <- G - BatchSize * s (Giả định mỗi sample đóng góp s vào các class khác)
    # Tuy nhiên Algorithm 1 thường xử lý trực tiếp.
    # Ta giữ nguyên gradient gốc để chạy Algorithm 1 cho chuẩn.

    predicted_labels = []
    
    # Đảm bảo m là số âm (theo định nghĩa Property 1)
    # Nếu Aux Data tính ra m dương (do nhiễu), ta ép về âm hoặc cảnh báo
    # if m_impact > 0: m_impact = -m_impact 

    # Copy để lặp
    g_curr = gradients.clone()
    g_curr = g_curr - s_offset 

    # 3. Algorithm 1 (Iterative Extraction)
    while len(predicted_labels) < batch_size:
        
        # Bước A: Ưu tiên Property 1 (Số Âm)
        # Tìm các index có giá trị < 0 (hoặc ngưỡng an toàn < m/2)
        # Trong bài báo gốc, họ ưu tiên lấy min(G).
        
        # Tìm phần tử nhỏ nhất (Most Negative)
        min_val, min_idx = torch.min(g_curr, dim=0)
        idx = min_idx.item()
        
        # Thêm vào danh sách dự đoán
        predicted_labels.append(idx)
        
        # Bước B: Cập nhật Gradient (Trừ đi Impact)
        # g_new = g_old - m
        # Vì m là số âm, trừ m tương đương cộng một lượng dương -> g tiến về 0 hoặc dương
        g_curr[idx] = g_curr[idx] - m_impact
        
        
    return sorted(predicted_labels)