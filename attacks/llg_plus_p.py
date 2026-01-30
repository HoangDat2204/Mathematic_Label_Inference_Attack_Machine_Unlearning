import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import nnls

def compute_impact_and_offsetp(model, aux_loader, num_classes, device):
    """
    Tính Impact Matrix (A) từ Aux Data.
    Mỗi cột của A là Gradient trung bình (Mean Gradient) của một class.
    
    Trong bài báo, họ xấp xỉ: Gradient_Batch ~ Sum(N_c * Impact_c) + Offset
    Ở đây ta tính Impact_c = Mean(Gradient_c) trên tập Aux.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Lưu tổng gradient cho từng class để tính trung bình
    sum_grads = {c: None for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}
    
    print("   [LLG+] Computing Impact Matrix from Aux Data...")
    
    for images, labels in aux_loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Để lấy gradient chuẩn xác cho từng class, ta dùng công thức Bias Gradient:
        # g_bias = prob - one_hot
        probs = torch.softmax(outputs, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Gradient của Bias lớp cuối (cực kỳ sạch và hiệu quả cho LLG+)
        bias_grads = probs - targets_one_hot
        
        # Cộng dồn
        for i in range(len(labels)):
            lbl = labels[i].item()
            grad = bias_grads[i].detach().cpu().numpy()
            
            if sum_grads[lbl] is None:
                sum_grads[lbl] = grad
            else:
                sum_grads[lbl] += grad
            counts[lbl] += 1
            
    # Xây dựng Matrix A [Dimension, Num_Classes]
    impact_matrix = []
    for c in range(num_classes):
        if counts[c] > 0:
            mean_grad = sum_grads[c] / counts[c]
        else:
            mean_grad = np.zeros(num_classes) # Fallback nếu Aux thiếu class
        impact_matrix.append(mean_grad)
    
    # Transpose để có dạng [Dim, Classes] cho phép nhân ma trận
    A = np.array(impact_matrix).T 
    return A

def attack_llg_plusp(proxy_gradients, impact_matrix, batch_size, num_classes=10):
    """
    LLG+ Attack: Giải hệ phương trình A * x = b
    A: Impact Matrix (đã tính từ Aux)
    b: Proxy Gradient (từ Unlearning)
    x: Số lượng nhãn cần tìm (Counts)
    """
    # 1. Lấy Gradient Bias lớp cuối từ Proxy Gradients
    target_layer = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            target_layer = name
            break
            
    if target_layer is None: return []

    # Vector b (Proxy Gradient)
    b = proxy_gradients[target_layer].cpu().numpy()
    
    # Lưu ý dấu: Unlearning là Gradient Ascent (cộng), Impact tính từ Gradient Descent (trừ).
    # Proxy = W_old - W_new = - (W_new - W_old) ~ - (Impact)
    # Tuy nhiên Impact = p - y.
    # Nên thử nghiệm thực tế thường cần đảo dấu b hoặc A. 
    # Ta sẽ đảo dấu b để khớp với hệ phương trình dương.
    b = -b 

    # 2. Giải NNLS: argmin ||Ax - b||
    counts_float, _ = nnls(impact_matrix, b)
    
    # 3. Scale và Rounding (Largest Remainder Method)
    current_sum = counts_float.sum()
    if current_sum > 1e-6:
        counts_float = counts_float * (batch_size / current_sum)
    
    floor_counts = np.floor(counts_float).astype(int)
    remainders = counts_float - floor_counts
    
    diff = int(batch_size - floor_counts.sum())
    if diff > 0:
        top_indices = np.argsort(remainders)[-diff:]
        for idx in top_indices:
            floor_counts[idx] += 1
            
    # 4. Convert sang list labels
    predicted_labels = []
    for cls_idx in range(num_classes):
        count = floor_counts[cls_idx]
        if count > 0:
            predicted_labels.extend([cls_idx] * count)
            
    return sorted(predicted_labels)