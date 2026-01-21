# File: attacks/rlu.py
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import nnls

import torch
import numpy as np
from scipy.stats import multivariate_normal

def construct_A_matrix(S_matrix, num_classes):
    """
    Xây dựng ma trận hệ số A cho hệ phương trình Ax = u.
    Dựa trên Gradient của Bias: g = p - y.
    
    Với input thuộc class k:
        E[g] = E[p|k] - E[y|k]
             = S_k    - one_hot_k
             
    Vậy Cột k của ma trận A sẽ là (S_k - one_hot_k).
    """
    S_numpy = S_matrix.cpu().numpy() # [Num_Classes, Num_Classes]
    
    # Ma trận A: Mỗi CỘT tương ứng với Impact của một Class k lên Bias Gradient
    A = np.zeros_like(S_numpy)
    
    # A[:, k] = S[k, :]^T - OneHot[k]
    # Lưu ý: S_matrix của ta: Hàng là Class thật (k), Cột là Class dự đoán (j)
    # Vector prob trung bình của class k là S_numpy[k, :]
    
    identity = np.eye(num_classes)
    
    for k in range(num_classes):
        # Impact của việc thêm 1 mẫu class k vào batch:
        # Nó làm tăng xác suất các class j (theo S_{k,j})
        # Và nó trừ đi 1 tại vị trí đúng k (do -y)
        
        # Vector cột k
        A[:, k] = S_numpy[k, :] - identity[k, :]
        
    return A

def compute_S_matrix(model, aux_loader, num_classes, device):
    """
    [Canonical RLU] Tính Ma trận S (Mean Softmax Probabilities).
    Paper Eq. 6 & 8: S_{k, j} là xác suất trung bình model đoán class j 
    khi input thực tế là class k.
    
    Output:
        S_matrix: [Num_Classes, Num_Classes]
        Hàng k, Cột j chứa giá trị S_{k,j}
    """
    model.eval()
    
    # Khởi tạo ma trận tích lũy xác suất
    S_sum = torch.zeros(num_classes, num_classes).to(device)
    counts = torch.zeros(num_classes).to(device)
    
    print(f"   [RLU] Computing S Matrix (Softmax stats) from Aux Data...")
    
    with torch.no_grad():
        for images, labels in aux_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Forward pass lấy Logits
            outputs = model(images)
            
            # 2. Tính Softmax Probabilities (p)
            probs = torch.softmax(outputs, dim=1) # [Batch, Num_Classes]
            
            # 3. Cộng dồn vào S theo từng class
            for i in range(len(labels)):
                lbl = labels[i].item()
                S_sum[lbl] += probs[i]
                counts[lbl] += 1
                
    # 4. Tính trung bình (Mean)
    S_matrix = torch.zeros_like(S_sum)
    for c in range(num_classes):
        if counts[c] > 0:
            S_matrix[c] = S_sum[c] / counts[c]
        else:
            # Fallback: Nếu không có aux data cho class này, giả định uniform hoặc one-hot
            # Ở đây để uniform noise để tránh singular matrix
            S_matrix[c] = 1.0 / num_classes
            
    return S_matrix

def compute_distribution_params(model, aux_loader, num_classes, device):
    """
    Tính Mean (mu) và Covariance (sigma) của logits cho từng class.
    Dựa trên Eq. 7 và Appendix B.3.
    Output:
        mus: List[np.array] - Mean vector cho mỗi class (khi input là class n)
        sigmas: List[np.array] - Covariance matrix cho mỗi class
    """
    model.eval()
    logits_per_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels in aux_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # Logits
            
            for i in range(len(labels)):
                lbl = labels[i].item()
                logits_per_class[lbl].append(outputs[i].cpu().numpy())

    mus = []
    sigmas = []
    
    for c in range(num_classes):
        data = np.array(logits_per_class[c])
        if len(data) > 1:
            mu = np.mean(data, axis=0)
            # Thêm regularization nhỏ để tránh ma trận kỳ dị
            sigma = np.cov(data, rowvar=False) + np.eye(num_classes) * 1e-5
        else:
            # Fallback nếu không đủ dữ liệu
            mu = np.zeros(num_classes)
            sigma = np.eye(num_classes)
            
        mus.append(mu)
        sigmas.append(sigma)
        
    return mus, sigmas

def monte_carlo_S(mus, sigmas, num_classes, num_samples=1000):
    """
    Ước tính ma trận S dựa trên phân phối Normal(mu, sigma) - Eq. 8.
    """
    S_matrix = np.zeros((num_classes, num_classes))
    
    for n in range(num_classes): # True label n
        # Sample M logits từ phân phối N(mu_n, sigma_n)
        if sigmas[n].ndim == 0: sigmas[n] = np.eye(num_classes) # Handle scalar case
        
        sampled_logits = np.random.multivariate_normal(mus[n], sigmas[n], num_samples)
        
        # Tính Softmax
        # Exp trick để ổn định số học
        exp_logits = np.exp(sampled_logits - np.max(sampled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # S_{n,j} = Mean probability của class j khi input là n
        S_matrix[n, :] = np.mean(probs, axis=0)
        
    return S_matrix

def posterior_search(initial_N, proxy_update_W, proxy_update_b, 
                     mu_t, sigma_t, mu_final, sigma_final, 
                     lr, num_epochs, batch_size, num_classes=10, T_iters=5):
    """
    Algorithm 2: Posterior Search [Paper Appendix C.2]
    Hiệu chỉnh số lượng nhãn (N) dựa trên mô phỏng đa epoch.
    """
    # Chuyển đổi input sang numpy
    dW = proxy_update_W.cpu().numpy() # [Out_features, In_features]
    db = proxy_update_b.cpu().numpy() # [Out_features]
    
    # 1. Tính Average Embedding Signal (e_bar) - Eq. 14, 112
    # e_l = dW_{j,l} / db_j. Ta lấy trung bình qua các class j hoặc tổng bình phương.
    # Trong bài báo (Eq 13), ta cần tổng bình phương embedding: sum(e_l^2)
    # Ước lượng sum(e^2) xấp xỉ tỉ lệ norm giữa dW và db
    
    # Tránh chia cho 0
    valid_idx = np.abs(db) > 1e-8
    if np.sum(valid_idx) == 0:
        return initial_N # Không thể update nếu db quá nhỏ
        
    # Tính hệ số tỉ lệ embedding_power = sum(e_l^2)
    # Dựa trên Eq 13: E[Delta q] = E[Delta b] * sum(e^2)
    # Ta ước lượng sum(e^2) trung bình từ update
    embedding_power = np.mean(np.sum(dW[valid_idx]**2, axis=1) / (db[valid_idx]**2))

    current_N = np.array(initial_N, dtype=int)
    
    # Target thực tế: Tổng Mean logits của model cuối cùng (mu_final)
    # Server biết model cuối, nên tính được mu_final từ aux data.
    target_sum_mu = np.sum([np.sum(m) for m in mu_final]) 

    # Bắt đầu vòng lặp hiệu chỉnh (Lines 25-43 in Alg 2)
    for _ in range(T_iters):
        # Guess per epoch (Line 26)
        g = np.round(current_N / num_epochs).astype(int)
        
        # Init simulation params (Line 23)
        mu_sim = [m.copy() for m in mu_t] # Bắt đầu từ model t
        
        # Simulate local epochs (Line 27)
        for tau in range(num_epochs):
            # Update S estimates (Line 28) - dùng Sigma cũ (giả sử sigma ít đổi)
            S_sim = monte_carlo_S(mu_sim, sigma_t, num_classes)
            
            # Estimate Delta b (Line 30) - Eq. 12
            # E[db_j] = (eta / |B|) * (N_j * sum(S_{j,n}) - sum(N_n * S_{n,j}))
            # Lưu ý: Code trước của bạn dùng update = -lr * grad, bài báo dùng grad
            # Ta cần tính Expectation của Update.
            
            expected_db = np.zeros(num_classes)
            # Tạo ma trận A tạm thời từ S_sim (giống hàm construct_A_matrix)
            # db_pred = A * g (vector g là số mẫu mỗi class trong 1 epoch)
            # A_{j,k}: tác động của class k lên bias j
            
            # Tính nhanh expected update bias cho epoch này
            for j in range(num_classes):
                term1 = g[j] * (1 - S_sim[j, j]) # Gradient contribution từ đúng class (dương/âm tuỳ định nghĩa)
                # Trong bài báo: u = delta_b/eta. Eq 9: E[u] = A*z. 
                # A[j,j] = sum_{n!=j} S_{j,n} = 1 - S_{j,j}.
                # Code cũ của bạn: A[j,j] = S[j,j] - 1. (Ngược dấu).
                # Ta giữ logic code cũ: Update = A * counts
                
                # Cột j của A cũ: A[:, j] = S[j, :] - Identity[j, :]
                # Update_pred = sum_k (A[:, k] * g[k])
                pass 

            # Cách tính vector hóa cho nhanh:
            # A_sim[:, k] = S_sim[k, :] - Identity[k, :]
            # expected_update = A_sim @ g
            A_sim = np.zeros((num_classes, num_classes))
            identity = np.eye(num_classes)
            for k in range(num_classes):
                A_sim[:, k] = S_sim[k, :] - identity[k, :]
            
            # Scale update theo learning rate và batch size
            # Code cũ: u = target_update / (-lr). A*N ~ u.
            # => target_update ~ (-lr) * A * N
            # Ở đây tính cho 1 epoch với g mẫu
            expected_update_tau = (-lr) * (A_sim @ g) 
            
            # Update intermediate means (Line 33, 34) - Eq. 13
            # Delta mu = Delta b * embedding_power
            # Delta b ở đây là expected_update_tau
            for n in range(num_classes):
                # Mỗi class n có mean vector mu_n (size 10)
                # Bias update ảnh hưởng lên tất cả logits
                mu_sim[n] += expected_update_tau * embedding_power

        # So sánh kết quả mô phỏng với thực tế (Line 37-42)
        # Bài báo so sánh: sum(mu_sim) vs sum(mu_real)
        # Nếu sum(mu_sim) > sum(mu_real) tại class j -> overestimated -> giảm N
        
        sim_sum_vector = np.zeros(num_classes)
        real_sum_vector = np.zeros(num_classes)
        
        # Bài báo: Sum over n of mu_{n,j} (Tổng giá trị logit j qua tất cả các class n)
        # Line 37: Sum_{n} mu_{n,j}
        for j in range(num_classes):
            sim_val = sum(mu_sim[n][j] for n in range(num_classes))
            real_val = sum(mu_final[n][j] for n in range(num_classes))
            sim_sum_vector[j] = sim_val
            real_sum_vector[j] = real_val
            
        diff = sim_sum_vector - real_sum_vector
        
        # Tìm class sai lệch nhiều nhất
        # Nếu diff > 0: Mô phỏng ra logit lớn hơn thực tế -> Guesstimate g đang quá lớn cho class này (hoặc các class kích hoạt nó)
        # Logic đơn giản hóa của RLU:
        j_max = np.argmax(diff) # Overestimated
        j_min = np.argmin(diff) # Underestimated
        
        if diff[j_max] > 0:
            current_N[j_max] = max(0, current_N[j_max] - num_epochs)
        if diff[j_min] < 0:
            current_N[j_min] = current_N[j_min] + num_epochs
            
    return current_N


def attack_rlu_full(model_original, proxy_update, aux_loader, 
                    batch_size, lr, num_epochs=1, num_classes=10, device='cpu'):

    proxy_update = {k: -v for k, v in proxy_update.items()}    
    # 1. Tính S_matrix tĩnh (Dùng cho Single Epoch hoặc Init guess)
    S_matrix = compute_S_matrix(model_original, aux_loader, num_classes, device)
    
    # 2. Chạy Algorithm 1 (Canonical RLU) để lấy N_initial
    # Lưu ý: Cần sửa hàm attack_rlu cũ để trả về 'floor_counts' (mảng số lượng)
    # thay vì predicted_labels (list nhãn)
    N_initial = attack_rlu_counts_only(proxy_update, S_matrix, batch_size, lr, num_classes)
    
    if len(N_initial) == 0: return [] # Failed
    
    final_N = N_initial

    # 3. Nếu Multi-epoch -> Chạy Algorithm 2
    if num_epochs > 1:
        print(" [RLU] Running Posterior Search (Alg 2) for Multi-epoch...")
        
        # Tính Stats cho model t (ban đầu)
        mu_t, sigma_t = compute_distribution_params(model_original, aux_loader, num_classes, device)
        
        # Tạo model t+1 (Model đã update) để tính target stats
        # Clone model và apply update
        import copy
        model_final = copy.deepcopy(model_original)
        with torch.no_grad():
            for name, param in model_final.named_parameters():
                if name in proxy_update:
                    # proxy_update là (old - new) -> new = old - proxy_update
                    # Tùy định nghĩa proxy_update của bạn. 
                    # Nếu proxy_update = update vector (gửi về server) = new - old
                    # param.data += proxy_update[name]
                    # Code cũ bạn: target_update / (-lr) => proxy_update đang là bias change
                    # Giả sử proxy_update chứa Tensor update weights chuẩn
                    param.data += proxy_update[name].to(device)
        
        mu_final, sigma_final = compute_distribution_params(model_final, aux_loader, num_classes, device)
        
        # Lấy Weight/Bias update của lớp cuối cho Posterior Search
        last_weight_name = list(proxy_update.keys())[-2] # Thường là weight lớp cuối
        last_bias_name = list(proxy_update.keys())[-1]   # Thường là bias lớp cuối
        
        dW = proxy_update[last_weight_name]
        db = proxy_update[last_bias_name]
        
        final_N = posterior_search(N_initial, dW, db, 
                                   mu_t, sigma_t, mu_final, sigma_final,
                                   lr, num_epochs, batch_size, num_classes)

    # 4. Convert counts to labels list
    predicted_labels = []
    for cls_idx in range(num_classes):
        count = int(final_N[cls_idx])
        if count > 0:
            predicted_labels.extend([cls_idx] * count)
            
    return sorted(predicted_labels)

def attack_rlu_counts_only(proxy_update, S_matrix, batch_size, lr, num_classes):
    """
    Phiên bản sửa đổi nhỏ của hàm attack_rlu cũ để trả về mảng counts (N)
    thay vì list labels. Logic giữ nguyên.
    """
    target_update = None
    # Tìm bias layer cuối
    for name in reversed(list(proxy_update.keys())):
        if 'bias' in name and proxy_update[name].shape[0] == num_classes:
            target_update = proxy_update[name].detach().cpu().numpy()
            break
            
    if target_update is None:
        print("[RLU Error] Không tìm thấy Bias lớp cuối.")
        return []

    # 2. Khôi phục Gradient tổng hợp (u) từ Update
    # diff = - lr * g  =>  g = diff / (-lr)
    # u (Target Gradient) = g
    u = target_update / (-lr)

    # 3. Xây dựng ma trận A
    A = construct_A_matrix(S_matrix, num_classes)
    
    # 4. Giải NNLS: argmin ||A * x - u||
    # A shape: [10, 10], u shape: [10]
    # Rất nhanh và ổn định
    try:
        counts_float, residual = nnls(A, u)
    except Exception as e:
        print(f"[RLU Error] NNLS Failed: {e}")
        return []
    
    # 5. Scale và Rounding (Largest Remainder Method)
    # RLU gốc dùng thuật toán tham lam (Greedy) hoặc làm tròn đơn giản
    # Ở đây ta dùng Largest Remainder để đảm bảo tổng = batch_size
    
    # Nếu tổng quá bé (do nhiễu), scale lên
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
    return floor_counts # np.array [N_class_0, N_class_1, ...]






# def attack_rlu(proxy_update, S_matrix, batch_size, lr, num_classes=10):
#     """
#     Canonical RLU Attack (Algorithm 1 - Single Epoch).
#     Giải hệ: A * N = u
    
#     Input:
#         proxy_update: Dict chứa Update (W_old - W_new). 
#                       Lưu ý: Update = -lr * Gradient.
#         S_matrix: Ma trận Softmax trung bình (đã tính từ Aux).
#         lr: Learning Rate (dùng để khôi phục Gradient từ Update).
#     """
#     # 1. Trích xuất Bias Update của lớp cuối
#     # Proxy Update ở đây là \Delta b = b_old - b_new
#     # Trong code main, ta tính diff = target - unlearned
#     # Nếu Unlearn là Gradient Ascent: b_unlearn = b_target + lr * g
#     # => diff = b_target - (b_target + lr * g) = - lr * g
    
#     target_update = None
#     # Tìm bias layer cuối
#     for name in reversed(list(proxy_update.keys())):
#         if 'bias' in name and proxy_update[name].shape[0] == num_classes:
#             target_update = proxy_update[name].detach().cpu().numpy()
#             break
            
#     if target_update is None:
#         print("[RLU Error] Không tìm thấy Bias lớp cuối.")
#         return []

#     # 2. Khôi phục Gradient tổng hợp (u) từ Update
#     # diff = - lr * g  =>  g = diff / (-lr)
#     # u (Target Gradient) = g
#     u = target_update / (-lr)

#     # 3. Xây dựng ma trận A
#     A = construct_A_matrix(S_matrix, num_classes)
    
#     # 4. Giải NNLS: argmin ||A * x - u||
#     # A shape: [10, 10], u shape: [10]
#     # Rất nhanh và ổn định
#     try:
#         counts_float, residual = nnls(A, u)
#     except Exception as e:
#         print(f"[RLU Error] NNLS Failed: {e}")
#         return []
    
#     # 5. Scale và Rounding (Largest Remainder Method)
#     # RLU gốc dùng thuật toán tham lam (Greedy) hoặc làm tròn đơn giản
#     # Ở đây ta dùng Largest Remainder để đảm bảo tổng = batch_size
    
#     # Nếu tổng quá bé (do nhiễu), scale lên
#     current_sum = counts_float.sum()
#     if current_sum > 1e-6:
#         counts_float = counts_float * (batch_size / current_sum)
    
#     floor_counts = np.floor(counts_float).astype(int)
#     remainders = counts_float - floor_counts
    
#     diff = int(batch_size - floor_counts.sum())
    
#     if diff > 0:
#         top_indices = np.argsort(remainders)[-diff:]
#         for idx in top_indices:
#             floor_counts[idx] += 1
            
#     # 6. Convert sang labels
#     predicted_labels = []
#     for cls_idx in range(num_classes):
#         count = floor_counts[cls_idx]
#         if count > 0:
#             predicted_labels.extend([cls_idx] * count)
            
#     return sorted(predicted_labels)