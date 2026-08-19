# File: attacks/llg.py
import torch
import numpy as np

def attack_llg(proxy_gradients, num_classes=10, batch_size=1):
    """
    Standard LLG Attack (Algorithm 1 from Paper 2110.09074).
    Tuân thủ Property 1: Gradient của target class là số ÂM.
    
    Quy trình:
    1. Lấy Gradient Vector (Ưu tiên Bias hoặc Weight sum).
    2. Ước lượng impact 'new_impact' từ các giá trị âm.
    3. Trích xuất nhãn qua 2 giai đoạn (Giai đoạn âm và Giai đoạn argmin).
    """
    
    # --- BƯỚC 1: TRÍCH XUẤT GRADIENT VECTOR ---
    target_grad = None
            
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                target_grad = torch.sum(proxy_gradients[name], dim=-1).detach().clone()
                break

    if target_grad is None:
        return []

    # --- BƯỚC 2: SAO CHÉP GRADIENT ĐỂ XỬ LÝ ---
    gradients = target_grad.clone()

    # --- BƯỚC 3: THUẬT TOÁN LLG_ATTACK ---
    h1_extraction = []
    negative_gradient = 0.0
    
    for i_cg, class_gradient in enumerate(gradients):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient.item()))
            negative_gradient += class_gradient.item()
    
    # Tính toán giá trị impact dựa trên tổng gradient âm thu được
    if batch_size > 0:
        new_impact = (1 + 1 / num_classes) * (negative_gradient / batch_size)
    else:
        new_impact = 0.0
        
    predicted_labels = []

    # Giai đoạn 1: Thêm các class có gradient âm vào danh sách dự đoán
    for (i_c, _) in h1_extraction:
        predicted_labels.append(i_c)
        gradients[i_c] = gradients[i_c] - new_impact

    # Giai đoạn 2: Điền thêm các nhãn còn thiếu cho đủ batch_size bằng phương pháp tìm argmin lặp lại
    remaining_slots = batch_size - len(predicted_labels)
    for _ in range(remaining_slots):
        min_id = torch.argmin(gradients).item()
        predicted_labels.append(min_id)
        # Cập nhật lại gradient sau khi chọn nhãn bằng cách trừ đi new_impact
        gradients[min_id] = gradients[min_id] - new_impact

    return sorted(predicted_labels)