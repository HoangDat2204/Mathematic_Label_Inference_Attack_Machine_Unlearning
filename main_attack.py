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

def compute_batch_accuracy(true_labels, pred_labels):
    if len(pred_labels) == 0: return 0.0
    count_true = Counter(true_labels)
    count_pred = Counter(pred_labels)
    correct = 0
    for label in count_true:
        correct += min(count_true[label], count_pred.get(label, 0))
    return (correct / len(true_labels)) * 100.0

def main():
    parser = argparse.ArgumentParser(description='5x2 Attack Benchmark (Including MLA)')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', default='ResNet18', type=str)
    
    parser.add_argument('--total_loops', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int) # Batch mỗi loop
    parser.add_argument('--aux_size', default=200, type=int)
    
    parser.add_argument('--unlearn_epochs', default=1, type=int)
    parser.add_argument('--exact_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    
    args = parser.parse_args()
    device = Config.DEVICE
    
    # Tính tổng số lượng ảnh cần dự đoán trong 1 lần tấn công MLA
    # Theo yêu cầu: "Batch size là toàn bộ một bộ ảnh muốn dự đoán"
    # Trong code này, mỗi loop ta unlearn một batch nhỏ (args.batch_size).
    # Vậy ta truyền args.batch_size vào hàm attack_mla.
    attack_batch_size = args.batch_size
    
    print("="*60)
    print(f"BENCHMARK: 5 Attacks (LLG, LLG+, ZLG, RLU, MLA (Ours)")
    print(f"Dataset: {args.dataset} | Batch Size: {attack_batch_size}")
    print("="*60)

    # 1. Load Data & Models (Giữ nguyên)
    retain_loader, forget_loader, _, num_channels, img_size, num_classes = get_dataloaders(args.dataset, batch_size=args.batch_size)
    forget_dataset = forget_loader.dataset
    aux_loader = DataLoader(Subset(retain_loader.dataset, list(range(args.aux_size))), batch_size=32, shuffle=False)
    
    target_model = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model   = get_custom_model(args.model, num_channels, num_classes, img_size).to(device)
    base_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_pretrained.pth")))
    target_model.load_state_dict(torch.load(os.path.join(Config.MODEL_SAVE_PATH, f"{args.model}_{args.dataset}_finetuned.pth")))

    unlearner = Unlearner(target_model, base_model, device)

    # 2. Pre-compute Stats (Giữ nguyên)
    print("\n[Prep] Computing Aux Statistics...")
    m_impact, s_offset = compute_impact_stats(target_model, aux_loader, num_classes, device)
    mean_p, mean_O = estimate_model_params(target_model, aux_loader, num_classes, device)
    
    # MLA không cần Aux Data tính toán, nó dùng Ma trận Synthetic.
    
    # Init Results
    methods = ['llg', 'plus', 'zlg', 'rlu', 'mla']
    results = {'approx': {m:0 for m in methods}, 'exact': {m:0 for m in methods}}
    
    all_indices = list(range(len(forget_dataset)))

    for loop in range(args.total_loops):
        print(f"\n>>> Loop {loop+1}/{args.total_loops}")
        start_idx = loop * args.batch_size
        if start_idx + args.batch_size > len(forget_dataset): break
        target_indices = all_indices[start_idx : start_idx + args.batch_size]
        
        batch_images = []
        batch_labels = []
        for idx in target_indices:
            img, lbl = forget_dataset[idx]
            batch_images.append(img)
            batch_labels.append(lbl)
        
        images = torch.stack(batch_images).to(device)
        labels = torch.tensor(batch_labels).to(device)
        true_labels = sorted(labels.tolist())
        print(f"Ground Truth : {true_labels}")
        
        batch_input = [(images, labels)]

        # --- A. APPROXIMATE ---
        model_approx = unlearner.approximate_unlearn(batch_input, lr=args.lr)
        diff_approx = get_weight_difference(target_model, model_approx)
        
        preds = {}
        preds['llg']  = attack_llg(diff_approx, num_classes, args.batch_size)
        preds['plus'] = attack_llg_plus(diff_approx, m_impact, s_offset, args.batch_size, num_classes)
        preds['zlg']  = attack_zlg(diff_approx, mean_p, mean_O, args.batch_size, num_classes)
        preds['rlu']  = attack_rlu_full(target_model, diff_approx, aux_loader, args.batch_size, args.lr, args.unlearn_epochs, num_classes, device)
        
        # [NEW] MLA Attack
        preds['mla'] = attack_mla(diff_approx, batch_size=attack_batch_size, num_classes=num_classes)
        
        print(f"[Approx] LLG: {compute_batch_accuracy(true_labels, preds['llg']):.1f}% | "
              f"LLG Plus: {compute_batch_accuracy(true_labels, preds['plus']):.1f}% | "
              f"ZLG: {compute_batch_accuracy(true_labels, preds['zlg']):.1f}% | "
              f"RLU: {compute_batch_accuracy(true_labels, preds['rlu']):.1f}% | "
              f"MLA (Ours): {compute_batch_accuracy(true_labels, preds['mla']):.1f}%")
        
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
        
        # [NEW] MLA Attack
        preds_ex['mla'] = attack_mla(diff_exact, batch_size=attack_batch_size, num_classes=num_classes)
        
        print(f"[Exact ] LLG: {compute_batch_accuracy(true_labels, preds_ex['llg']):.1f}% | "
              f"LLG Plus: {compute_batch_accuracy(true_labels, preds_ex['plus']):.1f}% | "
              f"ZLG: {compute_batch_accuracy(true_labels, preds_ex['zlg']):.1f}% | "
              f"RLU: {compute_batch_accuracy(true_labels, preds_ex['rlu']):.1f}% | "
              f"MLA (Ours): {compute_batch_accuracy(true_labels, preds_ex['mla']):.1f}%")
        
        for m in preds_ex: results['exact'][m] += compute_batch_accuracy(true_labels, preds_ex[m])

    # TỔNG KẾT
    print("\n" + "="*60)
    print(f"FINAL AVERAGE ACCURACY ({args.total_loops} loops)")
    print("="*60)
    print(f"{'Method':<10} | {'Approximate':<12} | {'Exact':<12}")
    print("-" * 50)
    for m in methods:
        avg_ap = results['approx'][m] / args.total_loops
        avg_ex = results['exact'][m] / args.total_loops
        if (m.upper() == "MLA"):
            m = "MLA (Ours)"
        print(f"{m.upper():<10} | {avg_ap:10.2f}% | {avg_ex:10.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()