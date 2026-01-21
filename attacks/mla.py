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

def create_synthetic_basis_matrix(num_classes):
    """
    Tạo ma trận cơ sở (Basis Matrix) nhân tạo theo quy luật cố định.
    Quy luật:
    - Đường chéo (Target Class): -1.0
    - Ngoài đường chéo (Non-Target): -0.11
    
    Ma trận này mô phỏng Gradient của Bias:
    - Khi xóa class i, Bias thứ i giảm mạnh (-1), các Bias khác giảm nhẹ (-0.11).
    """
    # Khởi tạo ma trận đầy -0.11
    basis_matrix = np.full((num_classes, num_classes), 0.11)
    
    # Điền đường chéo bằng -1.0
    np.fill_diagonal(basis_matrix, -1.0)
    
    # Chuẩn hóa từng cột (Basis Vector) về 1 để dùng cho Dot Product
    normalized_basis = []
    for i in range(num_classes):
        col_vec = basis_matrix[:, i]
        normalized_basis.append(normalize_to_unit(col_vec))
        
    # Stack lại thành ma trận [Num_Classes, Num_Classes]
    # Cột i là Basis Vector đại diện cho Class i
    final_basis = np.stack(normalized_basis, axis=1)
    
    return final_basis

def attack_mla(proxy_gradients, batch_size, num_classes=10):
    """
    MLA Attack: Dự đoán phân phối nhãn dùng Bias Peeling.
    Input:
        proxy_gradients: Dict chứa (W_target - W_unlearned).
                         Đây chính là Target Vector cần phân tích.
        batch_size: Tổng số lượng ảnh cần tìm (Số bước lặp).
        num_classes: Số lượng lớp (10).
    """
    
    # 1. TRÍCH XUẤT TARGET VECTOR (Bias của lớp cuối cùng)
    target_bias = None
    
    # Tìm layer bias cuối cùng
    # Ưu tiên tìm theo tên 'fc.bias' hoặc 'linear.bias' hoặc 'classifier.bias'
    # Hoặc tìm tensor 1 chiều có kích thước = num_classes
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            # Lấy data ra numpy, flatten thành vector 1D
            target_bias = proxy_gradients[name].detach().cpu().numpy().flatten()
            break
    print(target_bias)
    if target_bias is None:
        print("[MLA Error] Không tìm thấy Bias lớp cuối phù hợp.")
        return []

    # 2. CHUẨN BỊ BASIS MATRIX (Synthetic)
    # Tạo ma trận giả lập theo yêu cầu: Chéo -1, Ngoài -0.11
    Basis = create_synthetic_basis_matrix(num_classes)

    # 3. CHUẨN BỊ THUẬT TOÁN PEELING
    # Khởi tạo phần dư (Residual) ban đầu chính là Target Vector
    # [Quan trọng] Target Vector cũng nên được chuẩn hóa về 1 để so sánh hướng (Cosine Sim)
    # Tuy nhiên, trong logic bóc tách, ta cần giữ độ lớn tương đối.
    # Nhưng logic mẫu bạn đưa là: residual = normalize_to_unit(target_vector.copy())
    # Ta sẽ làm y hệt logic mẫu.
    
    residual = normalize_to_unit(target_bias.copy())
    
    # Mảng đếm kết quả
    counts = np.zeros(num_classes, dtype=int)
    
    # 4. VÒNG LẶP BÓC TÁCH (Greedy Peeling)
    # Lặp đúng bằng số lượng ảnh trong batch (batch_size)
    for step in range(batch_size):
        # a. Tính điểm tương đồng (Dot Product)
        # scores[i] = dot(Residual, Basis_Class_i)
        # Vì cả 2 đều đã normalize, đây chính là Cosine Similarity
        # Score càng lớn (càng dương) -> Càng cùng hướng.
        # Lưu ý: Bias Unlearn thường âm, Basis cũng âm -> Dot Product sẽ Dương.
        scores = np.dot(residual, Basis)
        
        # b. Chọn class có điểm cao nhất (Giống nhất)
        best_idx = np.argmax(scores)
        
        # c. Ghi nhận kết quả
        counts[best_idx] += 1
        
        # d. Loại bỏ (Peel off)
        # Tìm vector thành phần của class đó để trừ đi
        # Component = (Residual . Basis_i) * Basis_i = Projection Vector
        projection_val = scores[best_idx]
        
        # Trừ đi thành phần chiếu để loại bỏ thông tin của class này khỏi Residual
        component_to_remove = projection_val * Basis[:, best_idx]
        
        residual = residual - component_to_remove
        
        # [Optional] Re-normalize residual sau mỗi bước? 
        # Logic mẫu bạn đưa không re-normalize trong loop, chỉ update residual.
        # Ta giữ nguyên.

    # Dọn dẹp bộ nhớ
    del target_bias, Basis, residual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 5. CHUYỂN ĐỔI COUNTS THÀNH DANH SÁCH NHÃN
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = counts[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
            
    return sorted(predicted_labels)