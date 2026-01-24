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

# --- IMPORT CÁC THUẬT TOÁN ---
from attacks.llg import attack_llg
from attacks.llg_plus import attack_llg_plus, compute_impact_stats
from attacks.zlg import attack_zlg, estimate_model_params
from attacks.rlu import attack_rlu_full

# [NEW] Import MLA
from attacks.mla import attack_mla

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
            # Chọn 'count' ảnh từ class này (không lặp lại trong 1 batch)
            # Lưu ý: Nếu dataset quá bé không đủ ảnh thì lấy replace=True
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
    
    parser.add_argument('--total_loops', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int) 
    parser.add_argument('--aux_size', default=200, type=int)
    
    parser.add_argument('--unlearn_epochs', default=1, type=int)
    parser.add_argument('--exact_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    
    # --- [NEW] THAM SỐ ALPHA ---
    parser.add_argument('--alpha', default=100.0, type=float, 
                        help='Mức độ phân phối IID: Nhỏ (0.1)=Lệch, Lớn (100)=Đều')
    
    args = parser.parse_args()
    device = Config.DEVICE
    
    attack_batch_size = args.batch_size
    
    print("="*60)
    print(f"BENCHMARK: 5 Attacks (LLG, LLG+, ZLG, RLU, MLA (Ours))")
    print(f"Config: Batch={attack_batch_size} | Alpha={args.alpha} | Loops={args.total_loops}")
    print("="*60)

    # 1. Load Data & Models
    retain_loader, forget_loader, _, num_channels, img_size, num_classes = get_dataloaders(args.dataset, batch_size=args.batch_size)
    forget_dataset = forget_loader.dataset
    aux_loader = DataLoader(Subset(retain_loader.dataset, list(range(args.aux_size))), batch_size=32, shuffle=False)
    
    target_model = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model   = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_pretrained.pth")))
    target_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_finetuned.pth")))

    unlearner = Unlearner(target_model, base_model, device)

    # 2. Pre-compute Stats
    print("\n[Prep] Computing Aux Statistics...")
    m_impact, s_offset = compute_impact_stats(target_model, aux_loader, num_classes, device)
    mean_p, mean_O = estimate_model_params(target_model, aux_loader, num_classes, device)
    
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
    methods = ['llg', 'plus', 'zlg', 'rlu', 'rdm','mla']
    results = {'approx': {m:0 for m in methods}, 'exact': {m:0 for m in methods}}
    
    # Vòng lặp thí nghiệm
    for loop in range(args.total_loops):
        print(f"\n>>> Loop {loop+1}/{args.total_loops} (Alpha={args.alpha})")
        
        # --- [NEW] LẤY MẪU THEO ALPHA ---
        # Thay vì lấy tuần tự, ta lấy ngẫu nhiên theo phân phối Dirichlet
        target_indices = sample_batch_indices(class_to_indices, args.alpha, args.batch_size, num_classes)
        
        # Load Batch Data
        batch_images = []
        batch_labels = []
        for idx in target_indices:
            img, lbl = forget_dataset[idx]
            batch_images.append(img)
            batch_labels.append(lbl)
        
        images = torch.stack(batch_images).to(device)
        labels = torch.tensor(batch_labels).to(device)
        true_labels = sorted(labels.tolist())
        
        # In ra phân phối để kiểm tra xem Alpha có hoạt động không
        print(f"Ground Truth : {get_label_counts(true_labels, num_classes)}")
        
        batch_input = [(images, labels)]

        # --- A. APPROXIMATE ---
        model_approx = unlearner.approximate_unlearn(batch_input, lr=args.lr)
        diff_approx = get_weight_difference(target_model, model_approx)
        
        preds = {}
        preds['llg']  = attack_llg(diff_approx, num_classes, args.batch_size)
        preds['plus'] = attack_llg_plus(diff_approx, m_impact, s_offset, args.batch_size, num_classes)
        preds['zlg']  = attack_zlg(diff_approx, mean_p, mean_O, args.batch_size, num_classes)
        preds['rlu']  = attack_rlu_full(target_model, diff_approx, aux_loader, args.batch_size, args.lr, args.unlearn_epochs, num_classes, device)
        preds['mla'] = attack_mla(diff_approx, batch_size=attack_batch_size, num_classes=num_classes)
        preds['rdm'] = create_balanced_labels( args.batch_size, num_classes)
        

        print(f"[Approx] LLG: {compute_batch_accuracy(true_labels, preds['llg']):.1f}% | "
              f"Plus: {compute_batch_accuracy(true_labels, preds['plus']):.1f}% | "
              f"ZLG: {compute_batch_accuracy(true_labels, preds['zlg']):.1f}% | "
              f"RLU: {compute_batch_accuracy(true_labels, preds['rlu']):.1f}% | "
              f"RDM: {compute_batch_accuracy(true_labels, preds['rdm']):.1f}% | "
              f"MLA: {compute_batch_accuracy(true_labels, preds['mla']):.1f}%")
        
        for m in preds: results['approx'][m] += compute_batch_accuracy(true_labels, preds[m])

        # --- B. EXACT ---
        print(f"   [Exact] Retraining...")
        model_exact = unlearner.exact_unlearn(forget_dataset, target_indices, epochs=args.exact_epochs, lr=0.01)
        diff_exact = get_weight_difference(target_model, model_exact)
        
        preds_ex = {}
        preds_ex['llg']  = attack_llg(diff_exact, num_classes, args.batch_size)
        preds_ex['plus'] = attack_llg_plus(diff_exact, m_impact, s_offset, args.batch_size, num_classes)
        preds_ex['zlg']  = attack_zlg(diff_exact, mean_p, mean_O, args.batch_size, num_classes)
        preds_ex['rlu']  = attack_rlu_full(target_model, diff_exact, aux_loader, args.batch_size, 0.01, args.exact_epochs, num_classes, device)
        preds_ex['rdm'] = create_balanced_labels( args.batch_size, num_classes)
        preds_ex['mla'] = attack_mla(diff_exact, batch_size=attack_batch_size, num_classes=num_classes)
        

        print(f"[Exact ] LLG: {compute_batch_accuracy(true_labels, preds_ex['llg']):.1f}% | "
              f"Plus: {compute_batch_accuracy(true_labels, preds_ex['plus']):.1f}% | "
              f"ZLG: {compute_batch_accuracy(true_labels, preds_ex['zlg']):.1f}% | "
              f"RLU: {compute_batch_accuracy(true_labels, preds_ex['rlu']):.1f}% | "
              f"RDM: {compute_batch_accuracy(true_labels, preds_ex['rdm']):.1f}% | "
              f"MLA: {compute_batch_accuracy(true_labels, preds_ex['mla']):.1f}%")
                
        for m in preds_ex: results['exact'][m] += compute_batch_accuracy(true_labels, preds_ex[m])

    # TỔNG KẾT
    print("\n" + "="*60)
    print(f"FINAL AVERAGE ACCURACY | Alpha={args.alpha} | Loops={args.total_loops}")
    print("="*60)
    print(f"{'Method':<10} | {'Approximate':<12} | {'Exact':<12}")
    print("-" * 50)
    for m in methods:
        avg_ap = results['approx'][m] / args.total_loops
        avg_ex = results['exact'][m] / args.total_loops
        name = "MLA (Ours)" if m.upper() == "MLA" else m.upper()
        print(f"{name:<10} | {avg_ap:10.2f}% | {avg_ex:10.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()