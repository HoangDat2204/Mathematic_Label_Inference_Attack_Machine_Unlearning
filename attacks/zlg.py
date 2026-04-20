# File: attacks/zlg.py
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import Dataset


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y



def estimate_static_ZLG(model, aux_data, batch_size, n_classes):
    O_bar = 0
    pj = torch.zeros(n_classes).cuda()
    label_dict = {}

   
    y_aux = np.array([target for _, target in aux_data])
    K = n_classes
    for k in range(K):
        idx_k = np.where(y_aux == k)[0]
        label_dict[k] = list(idx_k)

    model.train()
    criterion = nn.CrossEntropyLoss()
    K = n_classes
    prop = 1

    for k in range(K):
        dict_k = label_dict[k]
        aux_num = int(prop * len(dict_k))
        aux_dict = np.random.choice(dict_k, aux_num)
        aux_dataset = LocalDataset(aux_data.dataset, aux_dict)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)

        count = 0
        for batch_idx, (inputs, targets) in enumerate(aux_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs, embedding = model.forward_with_features(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=-1)

            mean_probs = torch.mean(probs, dim=0)
            embedding_sum = torch.sum(embedding, dim=1)

            mean_embedding = torch.mean(embedding_sum, dim=0)

            O_bar += mean_embedding
            pj[k] += mean_probs[k]
            count += 1

    O_bar = O_bar / (n_classes * count)
    pj = pj / (count)
    return O_bar, pj

def attack_zlg(original_model, unlearned_model, proxy_gradients, lr ,aux_loader, batch_size, num_classes=10):
    """
    ZLG Attack (Corrected Device Mismatch).
    """
    # 1. Trích xuất Gradient Scalar (Sum Weights)
    O_bar, pj = estimate_static_ZLG( copy.deepcopy(original_model), aux_loader, batch_size, num_classes)

    new_O_bar, new_pj = estimate_static_ZLG( copy.deepcopy(unlearned_model), aux_loader, batch_size, num_classes)
    new_O_bar = (new_O_bar + O_bar) / 2
    new_pj = (new_pj + pj) / 2


    grad_vector = None
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                w_grad = proxy_gradients[name]
                grad_vector = torch.sum(w_grad,  dim=-1).detach().clone()
                break

    gradients_for_prediction = grad_vector/lr
    n = []
    for i in range(num_classes):
        nj = batch_size * (new_pj[i].detach().cpu() - gradients_for_prediction[i] / new_O_bar.detach().cpu())
        n.append(max(int(nj.item()), 0))
    prop = (batch_size) / sum(n)
    for i in range(num_classes):
        n[i] = round(n[i] * prop)
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = n[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
    
    return sorted(predicted_labels)