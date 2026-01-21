# File: attacks/zlg.py
import torch
import torch.nn as nn
import numpy as np

def estimate_model_params(model, aux_loader, num_classes, device):
    """
    Ước lượng tham số p (Mean Probability) và O_bar (Mean Feature Sum).
    """
    model.eval()
    
    sum_p = torch.zeros(num_classes).to(device)
    sum_O = 0.0 
    total_samples = 0
    
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features['feat'] = input[0].detach()
        return hook

    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            final_layer = module
    
    if final_layer is None: return None, None

    handle = final_layer.register_forward_hook(get_features('feat'))
    
    # print("   [ZLG] Estimating model parameters...")
    
    with torch.no_grad():
        for images, _ in aux_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            feat = features['feat']
            O_val = torch.sum(feat, dim=1)
            
            sum_p += torch.sum(probs, dim=0)
            sum_O += torch.sum(O_val).item()
            total_samples += images.size(0)
            
    handle.remove()
    
    if total_samples == 0: return None, 1.0

    mean_p = sum_p / total_samples
    mean_O = sum_O / total_samples
    
    return mean_p, mean_O

def attack_zlg(proxy_gradients, mean_p, mean_O, batch_size, num_classes=10):
    """
    ZLG Attack (Corrected Device Mismatch).
    """
    proxy_gradients = {k: -v for k, v in proxy_gradients.items()}    
    # 1. Trích xuất Gradient Scalar (Sum Weights)
    grad_vector = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                w_grad = proxy_gradients[name]
                grad_vector = torch.sum(w_grad, dim=1).detach().clone()
                break
    
    if grad_vector is None: return []

     # 2. Xử lý Dấu & Device
    # KHÔNG đảo dấu gradient (trừ khi proxy_gradients là update vector nghịch đảo)
    g = grad_vector 
    
    if isinstance(mean_p, torch.Tensor):
        g = g.to(mean_p.device)
        
    # 3. Giải phương trình tìm y (Theo Eq. 16 trong bài báo)
    if abs(mean_O) < 1e-9: mean_O = 1.0
    
    # Formula: Sum_y = Sum_p - (Batch_Size * Gradient) / Mean_O
    # Lưu ý: Gradient ở đây là gradient trung bình của batch (theo Eq. 12 FedSGD)
    y_raw = (batch_size * mean_p) - ((batch_size * g) / mean_O)
    
    # 4. Làm tròn số lượng
    y_pos = torch.relu(y_raw)
    
    current_sum = y_pos.sum()
    if current_sum > 1e-6:
        y_scaled = y_pos * (batch_size / current_sum)
    else:
        y_scaled = y_pos
        
    floor_counts = torch.floor(y_scaled).int()
    remainders = y_scaled - floor_counts
    
    diff = int(batch_size - floor_counts.sum())
    
    if diff > 0:
        _, top_indices = torch.topk(remainders, diff)
        for idx in top_indices:
            floor_counts[idx] += 1
            
    # 5. Convert sang list labels
    predicted_labels = []
    for cls_idx in range(num_classes):
        count = floor_counts[cls_idx].item()
        if count > 0:
            predicted_labels.extend([cls_idx] * count)
            
    return sorted(predicted_labels)