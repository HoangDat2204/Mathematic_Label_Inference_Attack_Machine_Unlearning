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

# def compute_basis_from_aux(model, aux_loader, num_classes, device):
#     """
#     Tính Basis Matrix thực tế từ Aux Data.
#     Mỗi cột i của ma trận sẽ là Gradient Bias trung bình khi input là Class i.
    
#     Công thức Gradient Bias: g = p - y
#     (Với CrossEntropyLoss, đạo hàm theo bias chính là xác suất p trừ đi one-hot label y)
#     """
#     model.eval()
    
#     # Khởi tạo ma trận tích lũy gradient [Num_Classes, Num_Classes]
#     # Hàng k chứa tổng gradient bias khi input là class k
#     sum_grads = torch.zeros(num_classes, num_classes).to(device)
#     counts = torch.zeros(num_classes).to(device)
    
#     # print("   [MLA+] Computing Basis Matrix from Aux Data...")
    
#     with torch.no_grad():
#         j = 0 
#         for images, labels in aux_loader:
#             images, labels = images.to(device), labels.to(device)
            
#             j+=1                
#             # Forward pass
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)
#             if (j == len(aux_loader)):
#                 print("probs: ", probs)
#             # Tính Gradient Bias cho từng mẫu trong batch
#             # g_i = p_i - y_i
#             # Thay vì loop, ta dùng scatter để trừ 1 tại đúng nhãn
            
#             # One-hot targets
#             targets_one_hot = torch.zeros_like(probs)
#             targets_one_hot.scatter_(1, labels.view(-1, 1), 1)
            
#             # Gradient Bias: [Batch, Num_Classes]
#             bias_grads = probs - targets_one_hot
            
#             # Cộng dồn vào sum_grads theo class
#             for i in range(len(labels)):
#                 lbl = labels[i].item()
#                 sum_grads[lbl] += bias_grads[i]
#                 counts[lbl] += 1

#     # Tính trung bình
#     basis_vectors = []
#     for c in range(num_classes):
#         if counts[c] > 0:
#             avg_grad = sum_grads[c] / counts[c]
#             print(counts)
#         else:
#             # Fallback nếu Aux thiếu class: Tạo vector giả lập (-1 tại c, 0 còn lại)
#             avg_grad = torch.zeros(num_classes).to(device)
#             avg_grad[c] = -1.0 
            
#         # Chuyển sang numpy và đưa vào list
#         basis_vectors.append(avg_grad.cpu().numpy())

#     # Stack lại: User yêu cầu "các dòng lần lượt là đại diện".
#     # Tuy nhiên để dùng cho phép nhân ma trận trong thuật toán Peeling (Residual . Basis), 
#     # ta cần các vector cơ sở nằm ở CỘT (Columns).
#     # Basis Matrix shape: [10, 10]
    
#     # Ở đây tôi stack thành [10, 10] (Hàng i là class i) sau đó Transpose.
#     # Như vậy Cột i sẽ là vector đặc trưng của Class i.
#     raw_basis = np.stack(basis_vectors, axis=0).T
    
#     # Chuẩn hóa từng cột về vector đơn vị (để tính Cosine Similarity)
#     final_basis = np.zeros_like(raw_basis)
#     for i in range(num_classes):
#         final_basis[:, i] = raw_basis[:, i]
        
#     return final_basis

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
    basis_matrix = np.full((num_classes, num_classes), 0.11*(1e-3))
    
    # Điền đường chéo bằng -1.0
    np.fill_diagonal(basis_matrix, -1.0*(1e-3))
    
    # Chuẩn hóa từng cột (Basis Vector) về 1 để dùng cho Dot Product
    normalized_basis = []
    for i in range(num_classes):
        col_vec = basis_matrix[:, i]
        # normalized_basis.append(normalize_to_unit(col_vec))
        normalized_basis.append(col_vec)

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
    
    # residual = normalize_to_unit(target_bias.copy())
    residual = target_bias.copy()
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
        # projection_val = scores[best_idx]
        projection_val = 1
        # if (step == batch_size):
        #     print("Residual: ", residual)
        #     print("Vector Basis: ", Basis[:, best_idx])
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


# =============================================================================
# =============================================================================



# def create_synthetic_basis_matrix(num_classes, diagonal_value):
#     """
#     Tạo ma trận cơ sở (Basis Matrix) động dựa trên thống kê của Target Bias.
    
#     Quy luật mới:
#     - Đường chéo (Target Class): diagonal_value (được tính từ Proxy Gradient).
#     - Ngoài đường chéo (Non-Target): Tự động tính sao cho tổng 1 hàng = 0.
#       => diagonal + (num_classes - 1) * off_diagonal = 0
#       => off_diagonal = -diagonal / (num_classes - 1)
#     """
#     # Tính giá trị ngoài đường chéo để tổng hàng = 0
#     if num_classes > 1:
#         off_diagonal_value = -diagonal_value / (num_classes - 1)
#     else:
#         off_diagonal_value = 0.0

#     # Khởi tạo ma trận với giá trị off-diagonal
#     basis_matrix = np.full((num_classes, num_classes), off_diagonal_value)
    
#     # Điền đường chéo
#     np.fill_diagonal(basis_matrix, diagonal_value)
    
#     # Stack lại thành ma trận [Num_Classes, Num_Classes]
#     # Cột i là Basis Vector đại diện cho Class i (Vì ma trận đối xứng nên cột hay hàng như nhau)
#     normalized_basis = []
#     for i in range(num_classes):
#         col_vec = basis_matrix[:, i]
#         # Giữ nguyên độ lớn (magnitude) để thực hiện phép trừ (peeling)
#         normalized_basis.append(col_vec)

#     final_basis = np.stack(normalized_basis, axis=1)
    
#     return final_basis

# def attack_mla(proxy_gradients, batch_size, num_classes=10):
#     """
#     MLA Attack: Dự đoán phân phối nhãn dùng Bias Peeling với Dynamic Basis.
#     Input:
#         proxy_gradients: Dict chứa (W_target - W_unlearned).
#         batch_size: Tổng số lượng ảnh cần tìm.
#         num_classes: Số lượng lớp (10).
#     """
    
#     # 1. TRÍCH XUẤT TARGET VECTOR (Bias của lớp cuối cùng)
#     target_bias = None
    
#     # Tìm layer bias cuối cùng
#     for name in reversed(list(proxy_gradients.keys())):
#         if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
#             target_bias = proxy_gradients[name].detach().cpu().numpy().flatten()
#             break
            
#     if target_bias is None:
#         print("[MLA Error] Không tìm thấy Bias lớp cuối phù hợp.")
#         return []

#     # 2. TÍNH TOÁN GIÁ TRỊ CƠ SỞ (Dynamic Calculation)
#     # Quy luật: Lấy tổng các phần tử âm của vector proxygradient chia cho số lượng vector (batch_size)
    
#     # Lấy các phần tử âm
#     negative_elements = target_bias[target_bias < 0]
    
#     if len(negative_elements) > 0:
#         sum_negative = np.sum(negative_elements)
#     else:
#         # Trường hợp hiếm: không có số âm (có thể do nhiễu quá lớn hoặc LR dương/âm bị đảo)
#         # Fallback: lấy min hoặc mean
#         sum_negative = np.sum(target_bias) # Hoặc xử lý tùy ý
        
#     # Tính giá trị đường chéo (Impact trung bình của 1 mẫu)
#     # "Chia cho số lượng vector" ở đây hiểu là chia cho batch_size 
#     # (để ra impact của 1 vector đơn lẻ)
#     diagonal_val = (sum_negative / batch_size)*(2)
    
#     # 3. CHUẨN BỊ BASIS MATRIX (Synthetic)
#     # Truyền diagonal_val vào để tính toán ma trận sao cho tổng hàng = 0
#     Basis = create_synthetic_basis_matrix(num_classes, diagonal_val)

#     # 4. CHUẨN BỊ THUẬT TOÁN PEELING
#     residual = target_bias.copy()
#     # Mảng đếm kết quả
#     counts = np.zeros(num_classes, dtype=int)
    
#     # 5. VÒNG LẶP BÓC TÁCH (Greedy Peeling)
#     for step in range(batch_size):
        
#         # a. Tính điểm tương đồng (Dot Product)
#         # Target Bias (Unlearn) thường âm, Basis Diagonal cũng âm.
#         # -> Dot product của class đúng sẽ ra số Dương lớn nhất.
#         scores = np.dot(residual, Basis)
        
#         # b. Chọn class có điểm cao nhất
#         best_idx = np.argmax(scores)
        
#         # c. Ghi nhận kết quả
#         counts[best_idx] += 1
        
#         # d. Loại bỏ (Peel off)
#         # projection_val = 1 nghĩa là trừ đi đúng 1 đơn vị Basis (1 vector đơn lẻ)
#         projection_val = 1.0
        
#         component_to_remove = projection_val * Basis[:, best_idx]
        
#         residual = residual - component_to_remove
        
#         # Debug (Optional)
#         # if step == 0:
#         #     print(f"MLA Init - Diag: {diagonal_val:.5f}, RowSum: {np.sum(Basis[:,0]):.5f}")

#     # Dọn dẹp bộ nhớ
#     del target_bias, Basis, residual
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
        
#     # 6. CHUYỂN ĐỔI COUNTS THÀNH DANH SÁCH NHÃN
#     predicted_labels = []
#     for cls_idx in range(num_classes):
#         c = counts[cls_idx]
#         if c > 0:
#             predicted_labels.extend([cls_idx] * c)
            
#     return sorted(predicted_labels)

def attack_mla_plus(proxy_gradients, basis_matrix, batch_size, num_classes=10):
    """
    MLA+ Attack: Bias Peeling với Basis Matrix thực tế.
    Input:
        proxy_gradients: Dict chứa Target Vector (Unlearning Diff).
        basis_matrix: Ma trận cơ sở đã tính từ Aux Data.
        batch_size: Số lượng ảnh cần tìm.
    """
    
    # 1. TRÍCH XUẤT TARGET VECTOR
    target_bias = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'bias' in name and proxy_gradients[name].shape[0] == num_classes:
            target_bias = proxy_gradients[name].detach().cpu().numpy().flatten()
            break
            
    if target_bias is None:
        return []

    # 2. CHUẨN BỊ
    # Residual ban đầu là Target Bias
    residual = target_bias.copy()
    counts = np.zeros(num_classes, dtype=int)
    Basis = basis_matrix # Đã được chuẩn hóa từ trước
    
    # 3. VÒNG LẶP BÓC TÁCH (Greedy Peeling)
    for step in range(batch_size):
        
        # a. Tính điểm tương đồng (Projection)
        # Target Bias (Unlearn Gradient Ascent) ~ Positive Gradient * (-lr) -> Âm.
        # Basis (Gradient Descent) ~ p - y -> Âm (tại class đúng).
        # Âm * Âm -> Dương.
        # Class nào có tích vô hướng lớn nhất là ứng viên sáng giá nhất.
        scores = np.dot(residual, Basis)
        
        # b. Chọn class tốt nhất
        best_idx = np.argmax(scores)
        
        # c. Ghi nhận
        counts[best_idx] += 1
        
        # d. Bóc tách
        # Trừ đi 1 đơn vị vector cơ sở (Unit Vector)
        # Giả định: Mỗi sample đóng góp 1 lượng hướng chuẩn hóa
        projection_val = 1.0 
        
        # [Optional Refinement]: Có thể dùng projection thực tế:
        # projection_val = scores[best_idx] 
        # Nhưng với batch peeling, trừ 1 đơn vị vector đơn vị thường ổn định hơn.
        
        component_to_remove = projection_val * Basis[:, best_idx]
        residual = residual - component_to_remove
        
    # 4. XUẤT KẾT QUẢ
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = counts[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
            
    return sorted(predicted_labels)