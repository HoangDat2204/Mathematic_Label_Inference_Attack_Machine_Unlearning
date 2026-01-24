# File: attacks/llg_plus.py
import torch
import torch.nn as nn
import numpy as np

def compute_impact_stats(model, aux_loader, num_classes, device):
    """
    Tính tham số m (Mean Impact) và S (Vector Offset) từ Aux Data.
    
    Ref: 
    - Impact m: Label-agnostic (Scalar) [3]
    - Offset S: Label-specific (Vector) [2]
    """
    model.eval()
    
    impacts = [] 
    # Offset cần lưu riêng cho từng class
    offset_sums = np.zeros(num_classes)
    offset_counts = np.zeros(num_classes)

    print(" [LLG+] Estimating 'm' (Impact) and 'S' (Offset Vector) from Aux Data...")

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features['feat'] = input[0].detach()
        return hook

    # Hook vào layer cuối
    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            final_layer = module
    
    if final_layer is None: return 0.0, np.zeros(num_classes)

    handle = final_layer.register_forward_hook(get_features('feat'))

    for images, labels in aux_loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        
        outputs = model(images)
        feat = features['feat']             
        probs = torch.softmax(outputs, dim=1) 

        # One-hot
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Error term: (p - y)
        error = probs - targets_one_hot

        # Tính g_scalar (gradient vô hướng)
        # g_i = Sum(h) * (p_i - y_i)
        sum_h = torch.sum(feat, dim=1) 
        grads_scalar = error * sum_h.unsqueeze(1) # [Batch, NumClasses]

        # Tách Impact và Offset
        for i in range(len(labels)):
            lbl = labels[i].item() # Nhãn đúng (Ground Truth)
            
            # 1. Thu thập Impact (tại nhãn đúng)
            val_impact = grads_scalar[i, lbl].item()
            impacts.append(val_impact)

            # 2. Thu thập Offset (tại các nhãn sai)
            # Với mọi class j != lbl, giá trị gradient đó là offset của class j
            for class_idx in range(num_classes):
                if class_idx != lbl:
                    val_offset = grads_scalar[i, class_idx].item()
                    offset_sums[class_idx] += val_offset
                    offset_counts[class_idx] += 1

    handle.remove()

    # Tính trung bình
    m_impact = np.mean(impacts)
    
    # Tính Vector S (Offset cho từng class)
    s_offset_vector = np.zeros(num_classes)
    for c in range(num_classes):
        if offset_counts[c] > 0:
            s_offset_vector[c] = offset_sums[c] / offset_counts[c]
        else:
            s_offset_vector[c] = 0.0 # Fallback nếu không có mẫu

    print(f" [LLG+] Estimated m={m_impact}")
    print(f" [LLG+] Estimated S (first 5): {s_offset_vector[:5]}")

    return m_impact, s_offset_vector



def attack_llg_plus(proxy_gradients, m_impact, s_offset_vector, batch_size, num_classes=10):
    """
    LLG+ Attack (Correct implementation of Algorithm 1).
    
    Args:
        proxy_gradients: Dict gradients.
        m_impact: Scalar float (Impact).
        s_offset_vector: Numpy array hoặc Tensor shape [num_classes] (Offset).
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

    # Unlearning Proxy thường ngược dấu với Gradient Descent chuẩn.
    # LLG gốc giả định g_i < 0 là nhãn đúng.
    # Nếu proxy_gradients của bạn là gradient ascent, bạn cần đổi dấu.
    # Giả sử input đã đúng hướng gradient descent:
    G = grad_vector.clone() # Vector G trong Algorithm 1
    
    # Chuyển offset về tensor nếu cần
    if not isinstance(s_offset_vector, torch.Tensor):
        S = torch.tensor(s_offset_vector, device=G.device, dtype=G.dtype)
    else:
        S = s_offset_vector

    predicted_labels = []
    
    # --- ALGORITHM 1 STEP 1: Property 1 (Extract Negative Gradients) ---
    # Duyệt qua G, nếu g_i < 0 -> chắc chắn là label [1, 4]
    # Lưu ý: Property 1 đúng tuyệt đối trước khi trừ Offset.
    
    for i in range(num_classes):
        # Bài báo: "for g_i in G do if g_i < 0 then append i to E..."
        # Nếu g_i rất âm, nó có thể chứa nhiều instance, nhưng Algo 1 
        # chỉ mô tả vòng lặp đơn giản check < 0.
        if G[i] < 0:
            predicted_labels.append(i)
            G[i] = G[i] - m_impact # Cập nhật gradient (trừ số âm = cộng)
            
            # Kiểm tra an toàn: không bóc quá số lượng batch_size
            if len(predicted_labels) >= batch_size:
                return sorted(predicted_labels)

    # --- ALGORITHM 1 STEP 2: Calibration (Subtract Offset) ---
    # G <- G - S [1]
    G = G - S

    # --- ALGORITHM 1 STEP 3: Heuristic Extraction (Find Min) ---
    # "After calibration, the minimum gradient value... corresponding to a label" [5]
    
    while len(predicted_labels) < batch_size:
        # Tìm phần tử nhỏ nhất (Most Negative)
        min_val, idx_tensor = torch.min(G, dim=0)
        idx = idx_tensor.item()

        predicted_labels.append(idx)
        
        # Cập nhật Gradient: g_i <- g_i - m
        G[idx] = G[idx] - m_impact

    return sorted(predicted_labels)