# File: main_attack.py
import torch
import argparse
import os
import numpy as np
from collections import Counter
from configs import Config
from recovery.data import get_dataloaders
from recovery.nn.custom_cnn import get_custom_model
from recovery.unlearn import Unlearner, get_weight_difference
from torch.utils.data import DataLoader, Subset
import random
import time

# --- IMPORT CÁC THUẬT TOÁN ---
from attacks.llg import attack_llg
from attacks.llg_plus import attack_llg_plus
from attacks.zlg import attack_zlg
# from attacks.zlgp import attack_zlgp
from attacks.rlu import attack_rlu_full
# from attacks.llg_plus_p import attack_llg_plusp,  compute_impact_and_offsetp

# [NEW] Import MLA
# from attacks.mla import attack_mla, attack_mla_plus, compute_basis_from_aux
from attacks.mla import attack_mla
import torch.nn as nn

from collections import Counter


import torch

def transform_to_count_list(input_list, n):
    # Khởi tạo list mới gồm n phần tử có giá trị ban đầu là 0
    result = [0] * n
    
    # Đếm số lần xuất hiện và cộng 1 vào vị trí index tương ứng
    for num in input_list:
        if 0 <= num < n:  # Đảm bảo giá trị num nằm trong phạm vi chỉ số của list mới
            result[num] += 1
            
    return result

def get_last_layer_parameters(model):
    """
    Tự động tìm lớp nn.Linear cuối cùng trong mô hình PyTorch
    và trả về trọng số (weight) cùng bias.
    """
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            
    if last_linear is not None:
        # Lấy bản sao của tensor dưới dạng detach để tránh can thiệp vào đồ thị tính toán
        weights = last_linear.weight.detach().clone()
        biases = last_linear.bias.detach().clone() if last_linear.bias is not None else None
        return weights, biases
    else:
        raise ValueError("Không tìm thấy lớp nn.Linear nào trong mô hình.")


def measure_metrics(device, func, *args, **kwargs):
    """
    Đo đạc 3 chỉ số: Thời gian chạy, Peak VRAM tiêu thụ (MB), và tổng số FLOPs của một hàm.
    """
    is_cuda = 'cuda' in str(device)
    
    # 1. ĐỒNG BỘ VÀ RESET THÔNG SỐ VRAM
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset bộ đếm đỉnh VRAM
        mem_before = torch.cuda.memory_allocated()  # Bộ nhớ hiện tại trước khi chạy
    else:
        mem_before = 0

    flops = 0
    t0 = time.perf_counter()

    # 2. ĐO FLOPS BẰNG PYTORCH PROFILER GỐC
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            
        with torch.profiler.profile(
            activities=activities,
            with_flops=True  # Bật tính năng đếm FLOPs
        ) as prof:
            result = func(*args, **kwargs)
            
        # Cộng dồn tất cả các FLOPs từ các toán tử (như Conv2d, Linear,...)
        flops = sum(event.flops for event in prof.key_averages() if event.flops is not None)
    except Exception:
        # Fallback phòng hờ nếu môi trường không hỗ trợ đếm FLOPs bằng Profiler
        result = func(*args, **kwargs)
        flops = -1  # Ký hiệu không đo được FLOPs

    # 3. ĐỒNG BỘ VÀ ĐO ĐẠC TIME & VRAM SAU KHI CHẠY
    if is_cuda:
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()  # Lượng VRAM cao nhất đạt được khi hàm chạy
        vram_used = max(0.0, (mem_peak - mem_before) / (1024 ** 2))  # Chuyển đổi Bytes sang MB
    else:
        vram_used = 0.0

    elapsed_time = time.perf_counter() - t0
    
    return result, elapsed_time, vram_used, flops

def count_classes(loader):
    class_counts = Counter()
    
    for _, labels in loader:
        # Nếu batch_size > 1, labels sẽ là một tensor, ta chuyển về list
        # Nếu batch_size = 1, labels vẫn có thể là tensor([label])
        if labels.ndim > 0:
            class_counts.update(labels.tolist())
        else:
            class_counts.update([labels.item()])
            
    data=  sorted(class_counts.items())
    result = [t[1] for t in data]
    return result

def set_seed(seed):
    """
    Cố định seed cho tất cả các thư viện để đảm bảo kết quả tái lập được.
    """
    # 1. Python built-in random
    random.seed(seed)
    
    # 2. NumPy (Quan trọng cho việc lấy mẫu Dirichlet/Multinomial của bạn)
    np.random.seed(seed)
    
    # 3. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Cho trường hợp nhiều GPU
        
    # 4. Cấu hình backend để thuật toán Convolution luôn chạy giống nhau
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Random Seed set to: {seed}")


def compute_overlap_metric(diff_dict, original_model, num_classes=10):
    """
    Tính tích trọng số: (W_diff * W_orig) sau đó tổng theo chiều class.
    Input:
        diff_dict: Dictionary chứa chênh lệch trọng số (thường ở CPU).
        original_model: Model gốc (thường ở GPU).
    Output:
        Vector kết quả có kích thước [In_Features] (Ví dụ 512 với ResNet18).
    """
    target_key = None
    
    # 1. Tìm tên layer trọng số lớp cuối (thường là fc.weight hoặc linear.weight)
    # Layer này có shape [Num_Classes, In_Features] (Ví dụ [10, 512])
    for k in diff_dict.keys():
        if 'weight' in k and diff_dict[k].shape[0] == num_classes and len(diff_dict[k].shape) == 2:
            target_key = k
            break
            
    if target_key is None:
        print("[Metric Error] Không tìm thấy Weight lớp cuối.")
        return None

    # 2. Lấy Tensor
    # diff_dict thường nằm trên CPU (do hàm get_weight_difference trả về)
    w_diff = diff_dict[target_key] 
    
    # Lấy trọng số tương ứng từ model gốc và chuyển về CPU để tính toán
    w_orig = original_model.state_dict()[target_key].cpu()
    
    # 3. Nhân từng phần tử (Element-wise Multiplication)
    # Shape: [10, 512] * [10, 512] -> [10, 512]
    product = w_diff * w_orig
    
    # 4. Tính tổng theo chiều Class (dim=0 - chiều có kích thước 10)
    # Kết quả sẽ là vector [512]
    result_vector = torch.sum(product, dim=1)
    
    return result_vector

def create_balanced_labels(batch_size, num_classes=10):
    """
    Tạo một danh sách nhãn giả định (Baseline) với phân phối đều.
    """
    labels = []
    base_count = batch_size // num_classes
    remainder = batch_size % num_classes
    
    for i in range(num_classes):
        count = base_count + 1 if i < remainder else base_count
        labels.extend([i] * count)
        
    return sorted(labels)

def get_label_counts(ground_truth_list, num_classes=10):
    """
    Chuyển đổi list nhãn thành vector đếm số lượng xuất hiện của từng nhãn.
    """
    arr = np.array(ground_truth_list)
    count_vector = np.bincount(arr, minlength=num_classes)
    return count_vector

def compute_batch_accuracy(true_labels, pred_labels):
    if len(pred_labels) == 0: return 0.0
    count_true = Counter(true_labels)
    count_pred = Counter(pred_labels)
    correct = 0
    for label in count_true:
        correct += min(count_true[label], count_pred.get(label, 0))
    return (correct / len(true_labels)) * 100.0


def compute_class_accuracy_iou(true_labels, pred_labels):
    """
    Tính accuracy dựa trên tỷ lệ giao/hợp của các tập hợp lớp xuất hiện.
    Ví dụ: true=[2, 2, 3], pred=[1, 2, 3] -> Giao {2, 3} (2) / Hợp {1, 2, 3} (3) -> 66.67%
    """
    true_classes = set(true_labels)
    pred_classes = set(pred_labels)
    
    if len(true_classes) == 0: 
        return 0.0
        
    intersection = true_classes.intersection(pred_classes)
    union = true_classes.union(pred_classes)
    
    return (len(intersection) / len(union)) * 100.0

# --- [NEW] HÀM LẤY MẪU THEO DIRICHLET (ALPHA) ---
def sample_batch_indices(class_to_indices, alpha, batch_size, num_classes):
    """
    Lấy mẫu index dựa trên phân phối Dirichlet.
    - Alpha nhỏ (0.01, 0.1): Rất lệch (Non-IID), batch chỉ chứa 1-2 class.
    - Alpha lớn (100, 1000): Rất đều (IID), batch chứa đủ các class ngẫu nhiên.
    """
    # 1. Sinh phân phối xác suất p ~ Dirichlet(alpha)
    # Ví dụ alpha=0.1 -> p = [0.01, 0.9, 0.02...] (Lệch)
    # Ví dụ alpha=100 -> p = [0.1, 0.1, 0.1...] (Đều)
    proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
    
    # 2. Sinh số lượng mẫu cho mỗi class dựa trên p (Multinomial)
    # Ví dụ batch=8 -> counts = [0, 7, 1, 0...]
    class_counts = np.random.multinomial(batch_size, proportions)
    
    batch_indices = []
    
    # 3. Lấy ngẫu nhiên index thực tế từ dataset
    for class_idx, count in enumerate(class_counts):
        if count > 0:
            available_indices = class_to_indices[class_idx]
            replace = len(available_indices) < count
            selected = np.random.choice(available_indices, count, replace=replace)
            batch_indices.extend(selected)
            
    # Shuffle lại để không bị thứ tự theo class
    np.random.shuffle(batch_indices)
    return batch_indices


import torch
import torch.nn.functional as F

def predict_single_image_with_prob(model, image, target, device, class_names=None):
    """
    Dự đoán nhãn cho 1 ảnh lẻ, hiển thị xác suất (probability) của lớp được chọn
    và phân phối xác suất của toàn bộ các lớp.
    
    Args:
        model: Mô hình PyTorch đã được train.
        image: Tensor ảnh (đã được chuẩn hóa, kích thước [C, H, W]).
        target: Nhãn thực tế (số nguyên hoặc Tensor chứa 1 phần tử).
        device: Thiết bị chạy (cpu hoặc cuda).
        class_names: Danh sách tên các lớp (tùy chọn, ví dụ: ['airplane', 'automobile', ...])
    """
    # 1. Chuyển mô hình về chế độ đánh giá (evaluation mode)
    model.eval()
    
    # 2. Chuẩn bị tensor ảnh
    if not isinstance(image, torch.Tensor):
        raise TypeError("Ảnh đầu vào phải là một PyTorch Tensor đã được chuẩn hóa.")
    
    # Đảm bảo ảnh có 4 chiều [1, C, H, W] và đưa lên đúng thiết bị (GPU/CPU)
    image_tensor = image.unsqueeze(0).to(device)
    
    # Chuyển nhãn thực tế về dạng số nguyên thuần túy
    true_idx = target if isinstance(target, int) else target.item()
    
    # 3. Dự đoán không tính toán gradient
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Áp dụng Softmax theo chiều của các lớp (dim=1) để tính xác suất
        probabilities = F.softmax(outputs, dim=1)[0]  # Lấy phần tử đầu tiên của batch
        
    # 4. Tìm lớp có xác suất cao nhất
    max_prob, predicted_class = torch.max(probabilities, dim=0)
    
    predicted_idx = predicted_class.item()
    predicted_prob_pct = max_prob.item() * 100
    is_correct = (predicted_idx == true_idx)
    
    # --- HIỂN THỊ KẾT QUẢ ---
    print("\n" + "="*60)
    print(" KẾT QUẢ DỰ ĐOÁN CHO 1 ẢNH ".center(60, "-"))
    
    if class_names:
        true_label = class_names[true_idx]
        pred_label = class_names[predicted_idx]
        print(f"• Nhãn thực tế (True Label) : {true_label} (ID: {true_idx})")
        print(f"• Mô hình dự đoán (Predict)  : {pred_label} (ID: {predicted_idx})")
    else:
        print(f"• Nhãn thực tế (True Label) : ID {true_idx}")
        print(f"• Mô hình dự đoán (Predict)  : ID {predicted_idx}")
        
    print(f"• Độ tin cậy (Probability)   : {predicted_prob_pct:.2f}%")
    print(f"• Kết quả đánh giá          : {'ĐÚNG (CORRECT)' if is_correct else 'SAI (WRONG)'}")
    
    # Hiển thị Top 3 xác suất cao nhất để bạn dễ quan sát
    print("-" * 60)
    print("Top 3 lớp có xác suất cao nhất mô hình dự đoán:")
    top_probs, top_indices = torch.topk(probabilities, 3)
    for i in range(3):
        prob = top_probs[i].item() * 100
        idx = top_indices[i].item()
        label_name = class_names[idx] if class_names else f"ID {idx}"
        print(f"  {i+1}. {label_name:<15}: {prob:.2f}%")
    print("="*60)
    
    return is_correct, predicted_idx, predicted_prob_pct
    

def main():
    parser = argparse.ArgumentParser(description='5x2 Attack Benchmark (Including MLA)')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--unlearned_algo', default='neggrad', type=str) 

    parser.add_argument('--total_loops', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int) 
    parser.add_argument('--aux_size', default=200, type=int)
    

    parser.add_argument('--unlr', default=0.01, type=float)
    parser.add_argument('--mini_batch_size', default=256, type=int)
    parser.add_argument('--local_loops', default=1, type=int)

    #Hypeparameter for Retrain
    parser.add_argument('--pretrain_lr', default=0.01, type=float)
    parser.add_argument('--pretrain_epochs', default=1, type=int)

    parser.add_argument('--alpha', default=100.0, type=float, 
                        help='Mức độ phân phối IID: Nhỏ (0.1)=Lệch, Lớn (100)=Đều')
    parser.add_argument('--seed', default=42, type=int, help='Seed cố định (ví dụ 42)')
    

    args = parser.parse_args()
    set_seed(args.seed)
    device = Config.DEVICE
    
    attack_batch_size = args.batch_size
    
    print("="*60)
    print(f"BENCHMARK: 6 Attacks (LLG, LLG+, ZLG, RLU, MLA)")
    print(f"Config: Batch={attack_batch_size} | Alpha={args.alpha} | Loops={args.total_loops}")
    print("="*60)

    # 1. Load Data & Models
    retain_loader, forget_loader, test_loader, num_channels, img_size, num_classes = get_dataloaders(args.dataset)
    forget_dataset = forget_loader.dataset
    retain_dataset = retain_loader.dataset
    
    
    aux_loader = DataLoader(Subset(forget_dataset.dataset, list(range(args.aux_size))), batch_size=1, shuffle=False)
    target_model = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_pretrained.pth")))
    target_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_finetuned.pth")))

    unlearner = Unlearner(target_model, base_model, device)

  

    # --- [NEW] TẠO MAP INDEX THEO CLASS ---
    # Để lấy mẫu theo alpha, ta cần biết index nào thuộc class nào
    print("[Prep] Grouping indices by class for Alpha sampling...")
    class_to_indices = {i: [] for i in range(num_classes)}
    # Duyệt qua toàn bộ forget dataset
    # Lưu ý: Cách này giả định dataset có thuộc tính .targets hoặc .labels
    # Nếu dùng Subset, ta cần truy cập dataset gốc
    if hasattr(forget_dataset, 'targets'):
        targets = forget_dataset.targets
    elif hasattr(forget_dataset, 'labels'):
        targets = forget_dataset.labels
    else:
        # Fallback: Duyệt thủ công (chậm hơn chút)
        targets = [y for _, y in forget_dataset]
        
    for idx, label in enumerate(targets):
        # Lưu ý: targets có thể là tensor hoặc list
        lbl = label.item() if isinstance(label, torch.Tensor) else label
        class_to_indices[lbl].append(idx)

    # Init Results
    methods = ['llg', 'plus', 'zlg', 'rlu', 'rdm', 'mla'] # mla_p = MLA+
    # methods = ['llg', 'plus', 'zlg', 'rlu', 'rdm', 'mla', 'mla_p', 'zlgp', 'llg+p']
    results = {'approx': {m:0 for m in methods}, 'finetune': {m:0 for m in methods}, 'scrub': {m:0 for m in methods} , 'neggrad': {m:0 for m in methods}, 'retrain': {m:0 for m in methods}}
    results_class = {'approx': {m:0 for m in methods}, 'finetune': {m:0 for m in methods}, 'scrub': {m:0 for m in methods} , 'neggrad': {m:0 for m in methods}, 'retrain': {m:0 for m in methods}}
    # Vòng lặp thí nghiệm
    acc_retain_after = 0
    acc_test_after = 0
    acc_rem_forget_after = 0
    acc_batch_after = 0
    acc_batch_before = 0
    flops_dict_total = {'llg': 0 , 'plus': 0 , 'zlg': 0 , 'rlu': 0, 'mla': 0, 'rdm': 0}
    vram_total =  {'llg': 0 , 'plus': 0 , 'zlg': 0 , 'rlu': 0, 'mla': 0, 'rdm': 0}
    times_total = {'llg': 0 , 'plus': 0 , 'zlg': 0 , 'rlu': 0, 'mla': 0, 'rdm': 0}
    for loop in range(args.total_loops):
        print(f"\n>>> Loop {loop+1}/{args.total_loops} (Alpha={args.alpha})")

        # --- [NEW] LẤY MẪU THEO ALPHA ---
        # Thay vì lấy tuần tự, ta lấy ngẫu nhiên theo phân phối Dirichlet
        target_indices = sample_batch_indices(class_to_indices, args.alpha, args.batch_size, num_classes)
        
        batch_images = []
        batch_labels = []
        for idx in target_indices:
            img, lbl = forget_dataset[idx]
            batch_images.append(img)
            batch_labels.append(lbl)
        
        images = torch.stack(batch_images).to(device)
        labels = torch.tensor(batch_labels).to(device)
        true_labels = sorted(labels.tolist())
        batch_input = [(images, labels)]
        # --- A. APPROXIMATE ---
        # predict_single_image_with_prob(target_model, images[0], labels[0], device)
   
        if (args.unlearned_algo == "neggrad"):
            # Load Batch Data
           

            model_approx, acc_retain_after_epoch, acc_test_after_epoch, acc_rem_forget_after_epoch, acc_batch_after_epoch, acc_batch_before_epoch = unlearner.approximate_unlearn(
                list_of_batches=batch_input,
                retain_loader=retain_loader,
                test_loader=test_loader,
                forget_dataset=forget_dataset,
                target_indices=target_indices,
                lr=args.unlr,
                batch_size = args.mini_batch_size,
                local_epochs = args.local_loops
            )
            acc_retain_after += acc_retain_after_epoch
            acc_test_after += acc_test_after_epoch
            acc_rem_forget_after += acc_rem_forget_after_epoch
            acc_batch_after += acc_batch_after_epoch
            acc_batch_before += acc_batch_before_epoch

            # model_approx = unlearner.approximate_unlearn(
            #     list_of_batches=batch_input,
            #     lr=args.unlr
            # )
            
            # Hàm hỗ trợ đồng bộ hóa GPU trước khi ghi nhận mốc thời gian
            def sync_gpu():
                if 'cuda' in str(device):
                    torch.cuda.synchronize()

            diff_approx = get_weight_difference(target_model, model_approx)
            confident_approx = compute_overlap_metric(diff_approx, target_model, num_classes)
            
            # Khởi tạo từ điển lưu thời gian chạy của từng phương pháp
            times = {}
            vrams = {}
            flops_dict = {}
            
            preds = {}
            # for m in methods:
            #     preds[m]=[0]
            #     flops_dict[m] = [0]
            #     vrams[m] = [0
            #     times[m] = 0
                
                
            weights, bias = get_last_layer_parameters(target_model)
            # 1. Đo LLG
            # preds['llg'], times['llg'], vrams['llg'], flops_dict['llg'] = measure_metrics(
            #     device, attack_llg, diff_approx, num_classes, args.batch_size
            # )

            # 2. Đo LLG+ (Plus)
            # preds['plus'], times['plus'], vrams['plus'], flops_dict['plus'] = measure_metrics(
            #     device, attack_llg_plus, target_model, model_approx, diff_approx, args.unlr, aux_loader, args.batch_size, num_classes
            # )

            # 3. Đo ZLG
            preds['zlg'], times['zlg'], vrams['zlg'], flops_dict['zlg'] = measure_metrics(
                device, attack_zlg, target_model, model_approx, diff_approx, args.unlr, aux_loader, args.batch_size, num_classes
            )

            # # 4. Đo RLU
            # preds['rlu'], times['rlu'], vrams['rlu'], flops_dict['rlu'] = measure_metrics(
            #     device, attack_rlu_full, target_model, model_approx, diff_approx, aux_loader, args.batch_size, args.unlr, 1, num_classes, device
            # )

            # 5. Đo MLA
            preds['mla'], times['mla'], vrams['mla'], flops_dict['mla'] = measure_metrics(
                device, attack_mla, diff_approx, attack_batch_size, confident_approx, num_classes, weights, bias
            )
            print(transform_to_count_list(preds['mla'], num_classes))
            print(transform_to_count_list(true_labels, num_classes))
            # 6. Đo RDM (Random)
            # preds['rdm'], times['rdm'], vrams['rdm'], flops_dict['rdm'] = measure_metrics(
            #     device, create_balanced_labels, args.batch_size, num_classes
            # )

            # --- IN KẾT QUẢ ĐỘ CHÍNH XÁC ---
            # print(f"[Approx] LLG: | {compute_batch_accuracy(true_labels, preds['llg']):.1f}% | {compute_class_accuracy_iou(true_labels, preds['llg']):.1f}% |"
            #     f"Plus: {compute_batch_accuracy(true_labels, preds['plus']):.1f}% |  {compute_class_accuracy_iou(true_labels, preds['plus']):.1f}% | "
            print(f"ZLG: {compute_batch_accuracy(true_labels, preds['zlg']):.1f}% | {compute_class_accuracy_iou(true_labels, preds['zlg']):.1f}% |")
                # f"RLU: {compute_batch_accuracy(true_labels, preds['rlu']):.1f}% |{compute_class_accuracy_iou(true_labels, preds['rlu']):.1f}% |"
                # f"RDM: {compute_batch_accuracy(true_labels, preds['rdm']):.1f}% | {compute_class_accuracy_iou(true_labels, preds['rdm']):.1f}% |"
                # f"MLA: {compute_batch_accuracy(true_labels, preds['mla']):.1f}% |  {compute_class_accuracy_iou(true_labels, preds['mla']):.1f}% |" )

            # --- IN KẾT QUẢ TIME, VRAM & FLOPS ---
            print("\n" + "-"*80)
            print(" BÁO CÁO TÀI NGUYÊN TIÊU THỤ CỦA CÁC PHƯƠNG PHÁP ".center(80, "-"))
            for m in preds:
                # Định dạng hiển thị FLOPs (Giga FLOPs hoặc Mega FLOPs)
                f_val = flops_dict[m]
                if f_val >= 1e9:
                    f_str = f"{f_val / 1e9:.2f} GFLOPs"
                elif f_val >= 1e6:
                    f_str = f"{f_val / 1e6:.2f} MFLOPs"
                elif f_val == -1:
                    f_str = "N/A (Not Supported)"
                else:
                    f_str = f"{f_val} FLOPs"
                
                print(f"• {m.upper():<5} | Time: {times[m]:.4f}s | Peak VRAM: {vrams[m]:.4f} MB | Computations: {f_str}")
            print("-" * 80 + "\n")

            for m in preds:
                results['approx'][m] += compute_batch_accuracy(true_labels, preds[m])
                results_class['approx'][m] += compute_class_accuracy_iou(true_labels, preds[m])
                flops_dict_total[m] += flops_dict[m]
                vram_total[m] += vrams[m]
                times_total[m] += times[m]
                
        elif (args.unlearned_algo == "scrub"):
            print(f"   [SCRUB] Retraining via Alternating Min-Max Distillation...")
            model_scrub = unlearner.scrub_unlearn(retain_dataset, forget_dataset, target_indices, test_loader)
            diff_scrub = get_weight_difference(target_model, model_scrub)

            confident_scrub = compute_overlap_metric(diff_scrub, target_model, num_classes)
            preds_sc = {}
            # preds_sc['llg']  = attack_llg(diff_scrub, num_classes, args.batch_size)
            # preds_sc['plus'] = attack_llg_plus(target_model, model_scrub, diff_scrub, 0.001, aux_loader, args.batch_size, num_classes)
            preds_sc['zlg']  = attack_zlg(target_model, model_scrub, diff_scrub, 0.001, aux_loader, args.batch_size, num_classes)
            # preds_sc['rlu']  = attack_rlu_full(target_model, model_scrub, diff_scrub, aux_loader, args.batch_size, 0.001, num_epochs= 1, num_classes = num_classes, device = device)
            # preds_sc['rdm']  = create_balanced_labels(args.batch_size, num_classes)
            # preds_sc['mla']  = attack_mla(diff_scrub, batch_size=attack_batch_size, confident=confident_scrub, num_classes=num_classes, approx=True)

            # print(f"[SCRUB ] LLG: {compute_batch_accuracy(true_labels, preds_sc['llg']):.1f}% | "
                # f"Plus: {compute_batch_accuracy(true_labels, preds_sc['plus']):.1f}% | "
            print(f"ZLG: {compute_batch_accuracy(true_labels, preds_sc['zlg']):.1f}% | ")
                # f"RLU: {compute_batch_accuracy(true_labels, preds_sc['rlu']):.1f}% | "
                # f"RDM: {compute_batch_accuracy(true_labels, preds_sc['rdm']):.1f}% | "
                # f"MLA: {compute_batch_accuracy(true_labels, preds_sc['mla']):.1f}% | " )


            for m in preds_sc: 
                results['scrub'][m] += compute_batch_accuracy(true_labels, preds_sc[m])


        # --- D. NEGGRAD+ ---
        elif (args.unlearned_algo == "neggradp"):
            print(f"   [NegGrad+] Retraining via Gradient Ascent (Chance Level Clamping)...")
            # Gọi hàm neggrad_unlearn (Các tham số như epochs, lr, alpha đã có default, 
            model_ng = unlearner.neggrad_unlearn(retain_dataset, forget_dataset, target_indices, test_loader, num_classes = num_classes )
            
            # Trích xuất độ lệch trọng số (Gradient/Weight Leakage)
            diff_ng = get_weight_difference(target_model, model_ng)
            confident_ng = compute_overlap_metric(diff_ng, target_model, num_classes)
            
            # Khởi chạy các cuộc tấn công suy diễn nhãn (Label Inference Attacks)
            preds_ng = {}
            # preds_ng['llg']  = attack_llg(diff_ng, num_classes, args.batch_size)
            # preds_ng['plus'] = attack_llg_plus(target_model, model_ng, diff_ng, 0.01, aux_loader, args.batch_size, num_classes)
            preds_ng['zlg']  = attack_zlg(target_model, model_ng, diff_ng, 0.01, aux_loader, args.batch_size, num_classes)
            # preds_ng['rlu']  = attack_rlu_full(target_model, model_ng, diff_ng, aux_loader, args.batch_size, 0.01, num_epochs= 1, num_classes = num_classes, device = device)
            # preds_ng['rdm']  = create_balanced_labels(args.batch_size, num_classes)
            
            # Lưu ý: Cờ approx=True được giữ nguyên theo thiết lập ở khối SCRUB của bạn
            # preds_ng['mla']  = attack_mla(diff_ng, batch_size=attack_batch_size, confident=confident_ng, num_classes=num_classes, approx=True)

            # print(f"[NEGGRAD+] LLG: {compute_batch_accuracy(true_labels, preds_ng['llg']):.1f}% | "
                # f"Plus: {compute_batch_accuracy(true_labels, preds_ng['plus']):.1f}% | "
            print(f"ZLG: {compute_batch_accuracy(true_labels, preds_ng['zlg']):.1f}% | ")
                # f"RLU: {compute_batch_accuracy(true_labels, preds_ng['rlu']):.1f}% | "
                # f"RDM: {compute_batch_accuracy(true_labels, preds_ng['rdm']):.1f}% | "
                # f"MLA: {compute_batch_accuracy(true_labels, preds_ng['mla']):.1f}% | " )

            # Cập nhật kết quả vào từ điển tổng
            for m in preds_ng: 
                results['neggrad'][m] += compute_batch_accuracy(true_labels, preds_ng[m])
        
        elif (args.unlearned_algo == "retrain"):

            print(f"   [Retrain] Retraining from Scratch...")

            model_retrain = unlearner.retrain_from_scratch(
                retain_dataset_base=retain_dataset,
                forget_dataset_base=forget_dataset,
                indices_to_remove=target_indices,
                model_name=args.model,
                dataset_name=args.dataset,
                epochs=args.pretrain_epochs, # Nên dùng số epoch lớn (ví dụ 40) giống lúc train base_model
                lr=args.pretrain_lr,                      # LR khởi điểm lớn để học từ đầu
                num_channels=num_channels,
                img_size=img_size,
                num_classes=num_classes,
                device=device
            )

            
            diff_retrain = get_weight_difference(target_model, model_retrain)
            
            
            confident_ng = compute_overlap_metric(diff_retrain, target_model, num_classes)
            
            # Khởi chạy các cuộc tấn công suy diễn nhãn (Label Inference Attacks)
            preds_rt = {}
            preds_rt['llg']  = attack_llg(diff_retrain, num_classes, args.batch_size)
            preds_rt['plus'] = attack_llg_plus(target_model, model_retrain, diff_ndiff_retraing, 0.01, aux_loader, args.batch_size, num_classes)
            preds_rt['zlg']  = attack_zlg(target_model, model_retrain, diff_retrain, 0.01, aux_loader, args.batch_size, num_classes)
            preds_rt['rlu']  = attack_rlu_full(target_model, model_retrain, diff_retrain, aux_loader, args.batch_size, 0.01, num_epochs= 1, num_classes = num_classes, device = device)
            preds_rt['rdm']  = create_balanced_labels(args.batch_size, num_classes)            
            preds_rt['mla']  = attack_mla(diff_retrain, batch_size=attack_batch_size, confident=confident_ng, num_classes=num_classes, approx=True)

            print(f"[NEGGRAD+] LLG: {compute_batch_accuracy(true_labels, preds_rt['llg']):.1f}% | "
                f"Plus: {compute_batch_accuracy(true_labels, preds_rt['plus']):.1f}% | "
                f"ZLG: {compute_batch_accuracy(true_labels, preds_rt['zlg']):.1f}% | "
                f"RLU: {compute_batch_accuracy(true_labels, preds_rt['rlu']):.1f}% | "
                f"RDM: {compute_batch_accuracy(true_labels, preds_rt['rdm']):.1f}% | "
                f"MLA: {compute_batch_accuracy(true_labels, preds_rt['mla']):.1f}% | " )

            # Cập nhật kết quả vào từ điển tổng
            for m in preds_rt: 
                results['retrain'][m] += compute_batch_accuracy(true_labels, preds_rt[m])
        
        
        
        else:
            print("Hãy chọn thuật toán")



    # TỔNG KẾT
    print("\n" + "="*60)
    print(f"FINAL AVERAGE ACCURACY | Alpha={args.alpha} | Loops={args.total_loops}")
    print("="*60)
    print(f"{'Method':<10} | {'Approximate':<11} | {'Exact':<11} | {'SCRUB':<11} | {'NegGrad':<11} | {'Retrain':<11} " )
    print("-" * 50)
    for m in methods:
        avg_ap = results['approx'][m] / args.total_loops
        avg_sc = results['scrub'][m] / args.total_loops
        avg_ex = results['finetune'][m] / args.total_loops
        avg_neg = results['neggrad'][m] / args.total_loops
        avg_rt = results['retrain'][m] / args.total_loops

        name = "MLA (Ours)" if m.upper() == "MLA" else m.upper()
        print(f"{name:<10} | {avg_ap:10.2f}% | {avg_ex:10.2f}% | {avg_sc:10.2f}%  | {avg_neg:10.2f}% | {avg_rt:10.2f}%")
    print("="*60)

    for m in methods:
        avg_ap_class = results_class['approx'][m] / args.total_loops
        avg_sc_class = results_class['scrub'][m] / args.total_loops
        avg_ex_class = results_class['finetune'][m] / args.total_loops
        avg_neg_class = results_class['neggrad'][m] / args.total_loops
        avg_rt_class = results_class['retrain'][m] / args.total_loops

        name = "MLA (Ours)" if m.upper() == "MLA" else m.upper()
        print(f"{name:<10} | {avg_ap_class:10.2f}% | {avg_ex_class:10.2f}% | {avg_sc_class:10.2f}%  | {avg_neg_class:10.2f}% | {avg_rt_class:10.2f}%")
    print("="*60)
    

    print(f"{'Acc retain':<10} | {'Acc test':<11} | {'Acc Finetune':<11} | {'Acc forget':<11}| {'Acc forget before':<11}  " )
    print(f"{acc_retain_after / args.total_loops :<10} | {acc_test_after / args.total_loops :10.2f}% | {acc_rem_forget_after / args.total_loops :10.2f}% | {acc_batch_after / args.total_loops:10.2f}% | {acc_batch_before / args.total_loops:10.2f}%")

    for m in ['llg', 'plus', 'zlg', 'rlu', 'mla', 'rdm']:
      
        print(f"• {m.upper():<5} | Time: {times_total[m]/args.total_loops:.4f}s | Peak VRAM: {vram_total[m]/args.total_loops:.2f} MB | Computations: {flops_dict_total[m]/args.total_loops}")

if __name__ == '__main__':
    main()