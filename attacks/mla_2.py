import torch
import numpy as np
import gc

def normalize_to_unit(vector):
    """Chuẩn hóa vector về độ dài đơn vị (Unit Vector)"""
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def calculate_distribution_ratios(sum_vector, basis_vectors):
    """
    Tính toán tỉ lệ phân phối của sum_vector dựa trên hình chiếu lên các basis_vectors.
    """
    s_vec = np.array(sum_vector).flatten()
    b_mat = np.array(basis_vectors)
    
    if b_mat.ndim == 2:
        if b_mat.shape[1] == s_vec.shape[0]:
            projections = np.dot(b_mat, s_vec)
        elif b_mat.shape[0] == s_vec.shape[0]:
            projections = np.dot(s_vec, b_mat)
        else:
            raise ValueError(f"Lỗi kích thước: sum_vector {s_vec.shape} không khớp với basis {b_mat.shape}")
    else:
        raise ValueError("basis_vectors phải là mảng 2 chiều hoặc danh sách các vector.")

    clean_projections = np.maximum(projections, 0)
    total_score = np.sum(clean_projections)
    
    if total_score > 1e-9:
        ratios = clean_projections / total_score
    else:
        ratios = np.zeros_like(clean_projections)
        
    return ratios


def create_synthetic_basis_matrix(num_classes, diagonal_value):
    """
    Tạo ma trận cơ sở (Basis Matrix) động dựa trên thống kê của Target Bias.
    
    Quy luật mới:
    - Đường chéo (Target Class): diagonal_value (được tính từ Proxy Gradient).
    - Ngoài đường chéo (Non-Target): Tự động tính sao cho tổng 1 hàng = 0.
      => diagonal + (num_classes - 1) * off_diagonal = 0
      => off_diagonal = -diagonal / (num_classes - 1)
    """
    # Tính giá trị ngoài đường chéo để tổng hàng = 0
    if num_classes > 1:
        off_diagonal_value = -diagonal_value / (num_classes - 1)
    else:
        off_diagonal_value = 0.0

    # Khởi tạo ma trận với giá trị off-diagonal
    basis_matrix = np.full((num_classes, num_classes), off_diagonal_value)
    
    # Điền đường chéo
    np.fill_diagonal(basis_matrix, diagonal_value)
    
    # Stack lại thành ma trận [Num_Classes, Num_Classes]
    # Cột i là Basis Vector đại diện cho Class i (Vì ma trận đối xứng nên cột hay hàng như nhau)
    normalized_basis = []
    for i in range(num_classes):
        col_vec = basis_matrix[:, i]
        # Giữ nguyên độ lớn (magnitude) để thực hiện phép trừ (peeling)
        normalized_basis.append(col_vec)

    final_basis = np.stack(normalized_basis, axis=1)
    
    return final_basis

def attack_mla(proxy_gradients, batch_size, confident=None, num_classes=10,  weights=None, biases=None):
    """
    MLA Attack: Bias Peeling dựa trên căn chỉnh hình học ETF-NC (Neural Collapse Calibration).
    
    Args:
        proxy_gradients (dict): Gradient thu được từ mô hình.
        batch_size (int): Kích thước batch của dữ liệu mục tiêu.
        confident (any): Không dùng nữa (giữ lại để tương thích ngược).
        num_classes (int): Số lượng lớp phân loại.
        approx (bool): Không dùng nữa (giữ lại để tương thích ngược).
        weights (torch.Tensor hoặc np.ndarray, optional): Trọng số của lớp tuyến tính cuối cùng.
                                                          Kích thước: (num_classes, feature_dim).
        biases (torch.Tensor hoặc np.ndarray, optional): Bias của lớp tuyến tính cuối cùng.
                                                         Kích thước: (num_classes,).
    """
    
    # 1. TRÍCH XUẤT TARGET VECTOR (Gradient của bias lớp cuối)
    target_bias = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            target_bias = proxy_gradients[name].detach().cpu().numpy().flatten()
            break
    
    if target_bias is None:
        print("[MLA Error] Không tìm thấy Bias lớp cuối phù hợp.")
        return []

    # 2. CHUẨN BỊ THÔNG TIN HÌNH HỌC (NC GEOMETRY)
    # Chuyển đổi trọng số sang NumPy
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            W = weights.detach().cpu().numpy()
        else:
            W = np.array(weights)
    else:
        # Trường hợp không có thông tin trọng số (Fallback sang ma trận đơn vị - giả định trực giao hoàn hảo)
        W = np.eye(num_classes)
        
    # Chuyển đổi bias sang NumPy
    if biases is not None:
        if isinstance(biases, torch.Tensor):
            B_vec = biases.detach().cpu().numpy().flatten()
        else:
            B_vec = np.array(biases).flatten()
    else:
        B_vec = np.zeros(num_classes)

    # Tính toán ma trận tích vô hướng giữa các véc-tơ trọng số (Cos Sim / Dot Product)
    dot_products = np.dot(W, W.T)

    # 3. ƯỚC LƯỢNG GIÁ TRỊ TOÀN CỤC g_base TỪ BATCH
    negative_elements = target_bias[target_bias < 0]
    if len(negative_elements) > 0:
        g_base = np.sum(negative_elements) / batch_size
    else:
        g_base = np.sum(target_bias) / batch_size
    
    Basis = create_synthetic_basis_matrix(num_classes, g_base)
    probs = calculate_distribution_ratios(target_bias, Basis)
    max_p = np.max(probs)

    alpha =1.0 #1.0 #1.0 #1.5
    beta = 30.0 #1.0 #2.0 #3.0

    boost_factor = alpha  + beta * (1.0 - max_p) *( 1/num_classes)
    print(boost_factor)
    # Tính Diagonal cuối cùng
    g_base = g_base * boost_factor
    
    # Xác định mục tiêu điều chỉnh trung bình (H_target = 1 + g_base)
    h_target = 1 + g_base
    min_h = 1.0 / num_classes + 1e-5
    max_h = 1.0 - 1e-5
    h_target = np.clip(h_target, min_h, max_h)

    # Định nghĩa hàm tính xác suất Softmax dựa trên hình học trọng số và nhiệt độ tau
    def compute_probabilities(tau):
        # Logit: z_j(n) = ( <w_j, w_n> / tau ) + b_j
        Z = dot_products / tau + B_vec[:, np.newaxis]
        Z_max = np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def get_h(tau):
        probs = compute_probabilities(tau)
        return np.mean(np.diagonal(probs))

    # 4. TÌM KIẾM NHỊ PHÂN (BISECTION SEARCH) ĐỂ GIẢI NGHIỆM tau*
    tau_min = 1e-5
    tau_max = 1e5
    for _ in range(100):
        tau_mid = (tau_min + tau_max) / 2.0
        h_mid = get_h(tau_mid)
        if h_mid > h_target:
            # Nếu xác suất trung bình trên đường chéo quá cao, tăng nhiệt độ tau để giảm độ tự tin
            tau_min = tau_mid
        else:
            tau_max = tau_mid
            
    tau_star = (tau_min + tau_max) / 2.0

    # 5. XÂY DỰNG MA TRẬN CƠ SỞ ĐỒNG NHẤT (BASIS MATRIX)
    calibrated_probs = compute_probabilities(tau_star)
    diag_probs = np.diagonal(calibrated_probs)
    
    # Tạo ma trận đường chéo: ngoài đường chéo bằng 0
    calibrated_probs = np.diag(diag_probs)
    # Ma trận Basis: Basis_{j, n} = p_n(j; tau*) - delta_{j, n}
    # Cột thứ n là basis vector biểu diễn cho việc bóc tách lớp n
    Basis = calibrated_probs - np.eye(num_classes)
    Basis = Basis.T 
    print(Basis)
    # 6. THUẬT TOÁN PEELING (BÓC TÁCH GRADIENT)
    residual = target_bias.copy()
    counts = np.zeros(num_classes, dtype=int)
    
    for step in range(batch_size):
        dot_product = np.dot(residual, Basis)
        # best_idx = np.argmax(scores)
        norm_basis = np.linalg.norm(Basis, axis=1)
        print(norm_basis)
        scores = dot_product / (norm_basis + 1e-9)
        # best_idx = np.argmax(scores)
        
        best_idx = np.argmin(residual)

        print("scores: ", scores)
        counts[best_idx] += 1
        print("residual: ",residual)
        
        component_to_remove = Basis[:, best_idx]
        print("component_to_remove: ", component_to_remove)
        residual = residual - component_to_remove

    # Dọn dẹp bộ nhớ giải phóng tài nguyên hệ thống
    del target_bias, Basis, residual, dot_products, calibrated_probs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = counts[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
            
    return sorted(predicted_labels)