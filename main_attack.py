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

from collections import Counter

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

def main():
    parser = argparse.ArgumentParser(description='5x2 Attack Benchmark (Including MLA)')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--unlearned_algo', default='neggrad', type=str) 

    parser.add_argument('--total_loops', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int) 
    parser.add_argument('--aux_size', default=200, type=int)
    

    parser.add_argument('--unlr', default=0.01, type=float)
    
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
    retain_loader, forget_loader, _, num_channels, img_size, num_classes = get_dataloaders(args.dataset, batch_size=args.batch_size)
    forget_dataset = forget_loader.dataset
    retain_dataset = retain_loader.dataset
    
    
    aux_loader = DataLoader(Subset(forget_dataset.dataset, list(range(args.aux_size))), batch_size=1, shuffle=False)
    target_model = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model   = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
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
    
    # Vòng lặp thí nghiệm
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
        print("True label : ", true_labels)
        # --- A. APPROXIMATE ---

        if (args.unlearned_algo == "neggrad"):
            # Load Batch Data
           

            model_approx = unlearner.approximate_unlearn(batch_input, lr=args.unlr)
            diff_approx = get_weight_difference(target_model, model_approx)
            

            confident_approx = compute_overlap_metric(diff_approx, target_model, num_classes)
            preds = {}
            preds['llg']  = attack_llg(diff_approx, num_classes, args.batch_size)
            preds['plus'] = attack_llg_plus(target_model, model_approx, diff_approx, args.unlr, aux_loader, args.batch_size, num_classes)
            preds['zlg']  = attack_zlg(target_model, model_approx, diff_approx, args.unlr, aux_loader, args.batch_size, num_classes)
            preds['rlu']  = attack_rlu_full(target_model, model_approx, diff_approx, aux_loader, args.batch_size, args.unlr, num_epochs= 1, num_classes = num_classes, device = device)
            preds['mla'] = attack_mla(diff_approx, batch_size=attack_batch_size, confident = confident_approx,num_classes=num_classes)
            preds['rdm'] = create_balanced_labels( args.batch_size, num_classes)

            print(f"[Approx] LLG: {compute_batch_accuracy(true_labels, preds['llg']):.1f}% | "
                f"Plus: {compute_batch_accuracy(true_labels, preds['plus']):.1f}% | "
                f"ZLG: {compute_batch_accuracy(true_labels, preds['zlg']):.1f}% | "
                f"RLU: {compute_batch_accuracy(true_labels, preds['rlu']):.1f}% | "
                f"RDM: {compute_batch_accuracy(true_labels, preds['rdm']):.1f}% | "
                f"MLA: {compute_batch_accuracy(true_labels, preds['mla']):.1f}% | " )

            for m in preds: results['approx'][m] += compute_batch_accuracy(true_labels, preds[m])
        
        
        
        # --- B. FineTune ---
        elif (args.unlearned_algo == "finetuning"):
            print(f"   [FineTune] Retraining...")
            model_fine_tune = unlearner.fine_tune_unlearn(forget_dataset, target_indices, unlr = args.unlr)
            diff_fine_tune = get_weight_difference(target_model, model_fine_tune)
                    
            target_bias = None
            for name in reversed(list(diff_fine_tune.keys())):
                if 'bias' in name and diff_fine_tune[name].shape[0] == num_classes:
                    target_bias = diff_fine_tune[name].detach().cpu().numpy().flatten()
                    break    
            print("Target_Bias: ", target_bias)

            confident_fine_tune = compute_overlap_metric(diff_fine_tune, target_model, num_classes)
            preds_ft = {}
            preds_ft['llg']  = attack_llg(diff_fine_tune, num_classes, args.batch_size)
            preds_ft['plus'] = attack_llg_plus(target_model, model_fine_tune, diff_fine_tune, 1, aux_loader, args.batch_size, num_classes)
            preds_ft['zlg']  = attack_zlg(target_model, model_fine_tune, diff_fine_tune, 1, aux_loader, args.batch_size, num_classes)
            preds_ft['rlu']  = attack_rlu_full(target_model, model_fine_tune, diff_fine_tune, aux_loader, args.batch_size, 1, num_epochs= 1, num_classes = num_classes, device = device)
            preds_ft['rdm'] = create_balanced_labels( args.batch_size, num_classes)
            preds_ft['mla'] = attack_mla(diff_fine_tune, batch_size=attack_batch_size, confident = confident_fine_tune, num_classes=num_classes, approx = False)
        
            print(f"[Exact ] LLG: {compute_batch_accuracy(true_labels, preds_ft['llg']):.1f}% | "
                f"Plus: {compute_batch_accuracy(true_labels, preds_ft['plus']):.1f}% | "
                f"ZLG: {compute_batch_accuracy(true_labels, preds_ft['zlg']):.1f}% | "
                f"RLU: {compute_batch_accuracy(true_labels, preds_ft['rlu']):.1f}% | "
                f"RDM: {compute_batch_accuracy(true_labels, preds_ft['rdm']):.1f}% | "
                f"MLA: {compute_batch_accuracy(true_labels, preds_ft['mla']):.1f}% | " )

                    
            for m in preds_ft: results['finetune'][m] += compute_batch_accuracy(true_labels, preds_ft[m])
            
        elif (args.unlearned_algo == "scrub"):

        # --- C. SCRUB ---
            print(f"   [SCRUB] Retraining via Alternating Min-Max Distillation...")
            
            # Gọi hàm scrub_unlearn với cấu hình SOTA đã gán cứng bên trong
            model_scrub = unlearner.scrub_unlearn(retain_dataset, forget_dataset, target_indices, unlr = args.unlr)
            
            # Trích xuất độ lệch trọng số (Gradient/Weight Leakage)
            diff_scrub = get_weight_difference(target_model, model_scrub)
            confident_scrub = compute_overlap_metric(diff_scrub, target_model, num_classes)
            
            
            # Khởi chạy các cuộc tấn công suy diễn nhãn (Label Inference Attacks)
            preds_sc = {}
            preds_sc['llg']  = attack_llg(diff_scrub, num_classes, args.batch_size)
            preds_sc['plus'] = attack_llg_plus(diff_scrub, m_impact, s_offset, args.batch_size, num_classes)
            preds_sc['zlg']  = attack_zlg(diff_scrub, mean_p, mean_O, args.batch_size, num_classes)
            preds_sc['rlu']  = attack_rlu_full(target_model, diff_scrub, aux_loader, args.batch_size, args.unlr,num_epochs= 1, num_classes = num_classes, device = device)
            preds_sc['rdm']  = create_balanced_labels(args.batch_size, num_classes)
            preds_sc['mla']  = attack_mla(diff_scrub, batch_size=attack_batch_size, confident=confident_scrub, num_classes=num_classes, approx=True)

            print(f"[SCRUB ] LLG: {compute_batch_accuracy(true_labels, preds_sc['llg']):.1f}% | "
                f"Plus: {compute_batch_accuracy(true_labels, preds_sc['plus']):.1f}% | "
                f"ZLG: {compute_batch_accuracy(true_labels, preds_sc['zlg']):.1f}% | "
                f"RLU: {compute_batch_accuracy(true_labels, preds_sc['rlu']):.1f}% | "
                f"RDM: {compute_batch_accuracy(true_labels, preds_sc['rdm']):.1f}% | "
                f"MLA: {compute_batch_accuracy(true_labels, preds_sc['mla']):.1f}% | " )


            for m in preds_sc: 
                results['scrub'][m] += compute_batch_accuracy(true_labels, preds_sc[m])


        # --- D. NEGGRAD+ ---
        elif (args.unlearned_algo == "neggradp"):
            print(f"   [NegGrad+] Retraining via Gradient Ascent (Chance Level Clamping)...")
            # Gọi hàm neggrad_unlearn (Các tham số như epochs, lr, alpha đã có default, 
            model_ng = unlearner.neggrad_unlearn(retain_dataset, forget_dataset, target_indices, unlr = args.unlr , num_classes = num_classes )
            
            # Trích xuất độ lệch trọng số (Gradient/Weight Leakage)
            diff_ng = get_weight_difference(target_model, model_ng)
            confident_ng = compute_overlap_metric(diff_ng, target_model, num_classes)
            
            # Khởi chạy các cuộc tấn công suy diễn nhãn (Label Inference Attacks)
            preds_ng = {}
            preds_ng['llg']  = attack_llg(diff_ng, num_classes, args.batch_size)
            preds_ng['plus'] = attack_llg_plus(diff_ng, m_impact, s_offset, args.batch_size, num_classes)
            preds_ng['zlg']  = attack_zlg(diff_ng, mean_p, mean_O, args.batch_size, num_classes)
            preds_ng['rlu']  = attack_rlu_full(target_model, diff_ng, aux_loader, args.batch_size, args.unlr, num_epochs= 1, num_classes = num_classes, device = device)
            preds_ng['rdm']  = create_balanced_labels(args.batch_size, num_classes)
            
            # Lưu ý: Cờ approx=True được giữ nguyên theo thiết lập ở khối SCRUB của bạn
            preds_ng['mla']  = attack_mla(diff_ng, batch_size=attack_batch_size, confident=confident_ng, num_classes=num_classes, approx=True)

            print(f"[NEGGRAD+] LLG: {compute_batch_accuracy(true_labels, preds_ng['llg']):.1f}% | "
                f"Plus: {compute_batch_accuracy(true_labels, preds_ng['plus']):.1f}% | "
                f"ZLG: {compute_batch_accuracy(true_labels, preds_ng['zlg']):.1f}% | "
                f"RLU: {compute_batch_accuracy(true_labels, preds_ng['rlu']):.1f}% | "
                f"RDM: {compute_batch_accuracy(true_labels, preds_ng['rdm']):.1f}% | "
                f"MLA: {compute_batch_accuracy(true_labels, preds_ng['mla']):.1f}% | " )

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
            preds_rt['plus'] = attack_llg_plus(diff_retrain, m_impact, s_offset, args.batch_size, num_classes)
            preds_rt['zlg']  = attack_zlg(diff_retrain, mean_p, mean_O, args.batch_size, num_classes)
            preds_rt['rlu']  = attack_rlu_full(target_model, diff_retrain, aux_loader, args.batch_size, args.pretrain_lr, num_epochs= 1, num_classes = num_classes, device = device)
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

if __name__ == '__main__':
    main()