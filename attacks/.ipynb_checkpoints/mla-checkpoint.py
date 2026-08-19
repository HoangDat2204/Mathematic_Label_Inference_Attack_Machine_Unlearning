# File: attacks/mla.py
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
    
    Args:
        sum_vector (np.array): Vector tổng hợp (1D array).
        basis_vectors (list hoặc np.array): Danh sách các vector cơ sở (vector đơn vị).
                                            Có thể là ma trận kích thước (Số_lớp, Số_chiều)
                                            hoặc (Số_chiều, Số_lớp).
    
    Returns:
        np.array: Mảng 1D chứa các tỉ lệ (0.0 - 1.0), tổng bằng 1.
    """
    # 1. Chuẩn hóa dữ liệu đầu vào sang NumPy Array
    s_vec = np.array(sum_vector).flatten()
    b_mat = np.array(basis_vectors)
    
    # Xử lý chiều ma trận để thực hiện phép nhân vô hướng (Dot Product)
    # Mục tiêu: Tạo ra mảng projections có số phần tử bằng số lượng vector basis
    if b_mat.ndim == 2:
        # Trường hợp 1: basis_vectors là danh sách các hàng [N_vectors, Dim]
        if b_mat.shape[1] == s_vec.shape[0]:
            projections = np.dot(b_mat, s_vec)
            
        # Trường hợp 2: basis_vectors là ma trận cột [Dim, N_vectors] (giống MLA code cũ)
        elif b_mat.shape[0] == s_vec.shape[0]:
            projections = np.dot(s_vec, b_mat)
        else:
            raise ValueError(f"Lỗi kích thước: sum_vector {s_vec.shape} không khớp với basis {b_mat.shape}")
    else:
        raise ValueError("basis_vectors phải là mảng 2 chiều hoặc danh sách các vector.")

    # 2. Xử lý nhiễu (Filter Negative Values)
    # Các giá trị hình chiếu âm (do trực giao không hoàn hảo) sẽ được gán bằng 0
    # Tương đương hàm ReLU
    clean_projections = np.maximum(projections, 0)
    
    # 3. Tính toán tỉ lệ (Normalize)
    total_score = np.sum(clean_projections)
    
    # Tránh lỗi chia cho 0 nếu vector tổng quá nhỏ hoặc ngược hướng hoàn toàn
    if total_score > 1e-9:
        ratios = clean_projections / total_score
    else:
        # Nếu không có tín hiệu dương nào, trả về mảng 0
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



def attack_mla(proxy_gradients, batch_size, confident, num_classes=10, weights=None, biases=None):
    """
    MLA Attack: Bias Peeling với Adaptive Diagonal Scaling.
    """
    
    # 1. TRÍCH XUẤT TARGET VECTOR
    target_bias = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            target_bias = proxy_gradients[name].detach().cpu().numpy().flatten()
            break
    
    if target_bias is None:
        print("[MLA Error] Không tìm thấy Bias lớp cuối phù hợp.")
        return []
    # if isinstance(confident, torch.Tensor):
    #     target_bias = confident.detach().cpu().numpy().flatten()
    # else:
    #     # Nếu lỡ nó đã là list hoặc numpy rồi thì convert cho chắc
    #     target_bias = np.array(confident).flatten()

    # print("Target Bias (Hardcoded):", target_bias)

    # 2. TÍNH TOÁN GIÁ TRỊ CƠ SỞ (Cơ bản)
    # Bây giờ target_bias đã là NumPy array, lệnh np.sum sẽ chạy bình thường
    negative_elements = target_bias[target_bias < 0]
    
    if len(negative_elements) > 0:
        sum_negative = np.sum(negative_elements)
    else:
        sum_negative = np.sum(target_bias)
        
    # Giá trị Diagonal cơ bản (Base Average)
    base_diagonal_val = sum_negative / batch_size
    
    # 3. [MỚI] ĐIỀU CHỈNH GIÁ TRỊ THEO PHÂN PHỐI (ADAPTIVE SCALING)
    # Bước A: Tạo vector phân phối xác suất từ Target Bias
    # Vector này phản ánh độ tự tin của tín hiệu hiện tại
    Basis = create_synthetic_basis_matrix(num_classes, base_diagonal_val)
    probs = calculate_distribution_ratios(target_bias, Basis)
    # Bước B: Đo độ lệch (Skewness) bằng Max Probability
    # max_p càng gần 1 -> Lệch (Tín hiệu rõ)
    # max_p càng gần 1/NumClasses -> Đều (Nhiễu nhiều)
    max_p = np.max(probs)
    
    # Bước C: Tính hệ số điều chỉnh (Boost Factor)
    # Logic: Càng đều (max_p thấp) -> Càng cần tăng độ âm (nhân với số > 1)
    # base_diagonal_val là số âm. Nhân với số > 1 sẽ làm nó âm hơn (độ lớn tăng).
    
    # Hệ số nhạy (Sensitivity): beta = 1.0 (Có thể chỉnh lên 2.0 nếu muốn gắt hơn)
    alpha =1.0 #1.0 #1.0 #1.5
    beta = 30.0 #1.0 #2.0 #3.0

    boost_factor = alpha  + beta * (1.0 - max_p) *( 1/num_classes)
    # Tính Diagonal cuối cùng
    final_diagonal_val = base_diagonal_val * boost_factor
    
    # Debug: In ra để xem nó hoạt động thế nào
    # print(f"   [MLA Logic] Max_P: {max_p:.4f} | Base Diag: {base_diagonal_val:.4f} | Boost: {boost_factor:.2f}x -> New Diag: {final_diagonal_val:.4f}")

    # 4. TẠO MA TRẬN BASIS VỚI GIÁ TRỊ ĐÃ ĐIỀU CHỈNH
    Basis = create_synthetic_basis_matrix(num_classes, final_diagonal_val)

    print(Basis)
    # 5. CHUẨN BỊ THUẬT TOÁN PEELING
    residual = target_bias.copy()
    counts = np.zeros(num_classes, dtype=int)
    
    # 6. VÒNG LẶP BÓC TÁCH
    for step in range(batch_size):
        scores = np.dot(residual, Basis)
        best_idx = np.argmax(scores)
        counts[best_idx] += 1
        
        projection_val = 1.0
        component_to_remove =   Basis[:, best_idx]
        residual = residual - component_to_remove

    # Dọn dẹp
    del target_bias, Basis, residual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = counts[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
            
    return sorted(predicted_labels)

