# File: attacks/zlgp.py
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import Dataset


class LocalDataset(Dataset):
    """
    Wrapper Dataset để ánh xạ index cục bộ vào dataset gốc.
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y


def estimate_static_ZLGp(model, aux_data, batch_size, n_classes):
    """
    ZLG+ Estimation: Ước lượng O_bar và pj bằng DỮ LIỆU THẬT từ auxiliary set.
    
    Khác với ZLG gốc (dùng dummy random inputs), ZLG+ sử dụng 
    dữ liệu auxiliary thực tế theo từng class để có ước lượng chính xác hơn.
    
    Args:
        model: Mô hình cần ước lượng.
        aux_data: DataLoader chứa dữ liệu auxiliary (batch_size=1).
        batch_size: Kích thước batch cho forward pass.
        n_classes: Số lượng lớp phân loại.
    
    Returns:
        O_bar: Mean embedding scalar (trung bình tổng embedding).
        pj: Vector xác suất trung bình theo class [n_classes].
    """
    O_bar = 0
    device = next(model.parameters()).device
    pj = torch.zeros(n_classes, device=device)
    label_dict = {}

    # Phân loại index theo nhãn từ auxiliary data
    y_aux = np.array([target.item() if isinstance(target, torch.Tensor) else target for _, target in aux_data])
    K = n_classes
    for k in range(K):
        idx_k = np.where(y_aux == k)[0]
        label_dict[k] = list(idx_k)

    model.train()
    criterion = nn.CrossEntropyLoss()
    prop = 1
    total_count = 0

    for k in range(K):
        dict_k = label_dict[k]
        if len(dict_k) == 0:
            continue
        aux_num = int(prop * len(dict_k))
        aux_dict = np.random.choice(dict_k, aux_num)
        aux_dataset = LocalDataset(aux_data.dataset, aux_dict)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)

        count = 0
        for batch_idx, (inputs, targets) in enumerate(aux_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # Forward pass lấy cả logits và embedding
            outputs, embedding = model.forward_with_features(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=-1)
            mean_probs = torch.mean(probs, dim=0)
            embedding_sum = torch.sum(embedding, dim=1)
            mean_embedding = torch.mean(embedding_sum, dim=0)

            O_bar += mean_embedding
            pj[k] += mean_probs[k]
            count += 1

        total_count += count

    if total_count > 0:
        O_bar = O_bar / (n_classes * (total_count // n_classes if total_count >= n_classes else 1))
        pj = pj / (total_count // n_classes if total_count >= n_classes else 1)
    return O_bar, pj


def attack_zlgp(original_model, unlearned_model, proxy_gradients, lr, aux_loader, batch_size, num_classes=10):
    """
    ZLG+ Attack: Sử dụng dữ liệu auxiliary THẬT để ước lượng O_bar và pj.
    
    So với ZLG gốc (dùng dummy random inputs), ZLG+ cho ước lượng 
    chính xác hơn nhờ sử dụng dữ liệu thực từ auxiliary set.
    
    Args:
        original_model: Mô hình gốc (trước unlearn).
        unlearned_model: Mô hình sau unlearn.
        proxy_gradients: Dict chênh lệch trọng số.
        lr: Learning rate dùng khi unlearn.
        aux_loader: DataLoader chứa dữ liệu auxiliary.
        batch_size: Kích thước batch tấn công.
        num_classes: Số lượng lớp phân loại.
    
    Returns:
        List nhãn dự đoán (sorted).
    """
    # 1. Ước lượng thống kê với dữ liệu thật từ cả 2 model
    O_bar, pj = estimate_static_ZLGp(copy.deepcopy(original_model), aux_loader, batch_size, num_classes)
    new_O_bar, new_pj = estimate_static_ZLGp(copy.deepcopy(unlearned_model), aux_loader, batch_size, num_classes)
    
    # Lấy trung bình 2 ước lượng
    new_O_bar = (new_O_bar + O_bar) / 2
    new_pj = (new_pj + pj) / 2

    # 2. Trích xuất Gradient Vector từ weight layer cuối
    grad_vector = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                w_grad = proxy_gradients[name]
                grad_vector = torch.sum(w_grad, dim=-1).detach().clone()
                break

    gradients_for_prediction = grad_vector / lr
    
    # 3. Tính số lượng mẫu ước lượng cho mỗi class
    raw_n = []
    for i in range(num_classes):
        nj = batch_size * (new_pj[i].detach().cpu() - gradients_for_prediction[i] / new_O_bar.detach().cpu())
        raw_n.append(max(nj.item(), 0))

    # 4. Chuẩn hóa tỷ lệ để tổng = batch_size
    total_raw = sum(raw_n)
    if total_raw == 0:
        n = [batch_size // num_classes] * num_classes
    else:
        scaled_n = [(val * batch_size / total_raw) for val in raw_n]
        n = [int(val) for val in scaled_n]
        remainder = batch_size - sum(n)
        
        # Bù phần dư do làm tròn (ưu tiên class có phần thập phân lớn nhất)
        diffs = [(i, scaled_n[i] - n[i]) for i in range(num_classes)]
        diffs.sort(key=lambda x: x[1], reverse=True)
        for i in range(int(remainder)):
            idx = diffs[i][0]
            n[idx] += 1

    # 5. Tạo danh sách nhãn dự đoán
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = n[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)

    return sorted(predicted_labels)
