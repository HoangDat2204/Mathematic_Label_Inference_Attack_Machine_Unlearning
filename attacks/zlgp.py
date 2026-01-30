import torch
import numpy as np

def attack_zlgp(proxy_gradients,  batch_size=1, num_classes=10):
    """
    ZLG (Zero-shot Label Gradient) Attack.
    Nguyên lý: Dựa trên dấu của Gradient.
    - Trong Unlearning (Gradient Ascent):
        Weight Update ~ Gradient L = (p - y) * h
        Class đúng (y=1) -> (p - 1) * h -> Giá trị Âm.
        Class sai (y=0) -> p * h -> Giá trị Dương.
    - Chiến thuật: Tìm các class có tổng giá trị gradient (Algebraic Sum) NHỎ NHẤT (Most Negative).
    """
    
    # 1. Tìm layer cuối cùng (Ưu tiên Weight, nếu không thì dùng Bias)
    target_layer = None
    # Duyệt ngược tìm weight layer
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                target_layer = name
                break
    
    if target_layer is None:
        return []

    # 2. Lấy Gradient Tensor
    # Shape: [num_classes, feature_dim]
    grad_tensor = proxy_gradients[target_layer]
    
    # LƯU Ý QUAN TRỌNG VỀ DẤU:
    # Proxy Gradient = W_old - W_new
    # Gradient Ascent: W_new = W_old + lr * Grad
    # => Proxy = - lr * Grad
    # => Proxy mang dấu NGƯỢC với Gradient thật.
    # Gradient thật: Class đúng là ÂM.
    # => Proxy: Class đúng sẽ là DƯƠNG.
    # DO ĐÓ: Ta cần tìm class có tổng giá trị LỚN NHẤT (Most Positive) trên Proxy Gradient.
    
    # Tính tổng giá trị (Sum) của từng hàng (từng class)
    # Không dùng abs() hay norm(), giữ nguyên dấu
    class_scores = torch.sum(grad_tensor, dim=1) # [num_classes]
    
    # 3. Phân phối số lượng (Ranking based on Score)
    # Class có Score càng lớn (dương) càng có khả năng là Target (do Proxy bị đảo dấu)
    
    # Để kết hợp với Largest Remainder Method, ta cần chuyển Score thành Probability dương
    # Dùng Softmax hoặc Min-Max scaling
    
    # Cách đơn giản: Chỉ lấy Top-K (vì ZLG thuần túy thường là ranking)
    # Tuy nhiên để công bằng với batch_size > 1, ta dùng phân phối tỷ lệ.
    
    # Shift về dương để tính tỷ lệ
    min_val = class_scores.min()
    shifted_scores = class_scores - min_val # Tất cả >= 0
    
    # ZLG giả định sự chênh lệch rất rõ ràng giữa Âm và Dương
    # Ta có thể dùng lũy thừa để làm rõ sự chênh lệch (Temperature Scaling)
    probs = shifted_scores ** 2 # Hoặc torch.exp(shifted_scores)
    probs = probs / (probs.sum() + 1e-9)
    
    estimated_counts = probs * batch_size
    floor_counts = torch.floor(estimated_counts).int()
    
    remainders = estimated_counts - floor_counts
    diff = int(batch_size - floor_counts.sum())
    
    if diff > 0:
        _, top_indices = torch.topk(remainders, diff)
        for idx in top_indices:
            floor_counts[idx] += 1
            
    # 4. Output
    predicted_labels = []
    for cls_idx in range(num_classes):
        count = floor_counts[cls_idx].item()
        if count > 0:
            predicted_labels.extend([cls_idx] * count)
    return sorted(predicted_labels)