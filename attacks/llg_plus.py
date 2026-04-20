# File: attacks/llg_plus.py
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

def estimate_static_LLG( model, aux_data, n_classes, batch_size):
    impact = 0
    offset = torch.zeros(n_classes)
    label_dict = {}
    
    y_aux = np.array([target for _, target in aux_data])
    K = n_classes
    for k in range(K):
        idx_k = np.where(y_aux == k)[0]
        label_dict[k] = list(idx_k)

    model.train()
    criterion = nn.CrossEntropyLoss()
    K = n_classes
    g_bar = 0
    prop = 1
    for k in range(K):
        dict_k = label_dict[k]
        aux_num = int(prop*len(dict_k))
        aux_dict = np.random.choice(dict_k, aux_num)
        aux_dataset = LocalDataset(aux_data.dataset, aux_dict)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)

        g_k = 0
        count = 0
        for batch_idx, (inputs, targets) in enumerate(aux_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            grads = torch.autograd.grad(loss, model.linear.parameters())
            grads = list((_.detach().cpu().clone() for _ in grads))

            w_grad, b_grad = grads[-2], grads[-1]

            gradients_for_prediction = torch.sum(w_grad, dim=-1)
            g_k += gradients_for_prediction[k]
            for j in range(K):
                if j == k:
                    continue
                else:
                    offset[j] += gradients_for_prediction[j]
            count += 1

        g_k = g_k / count
        g_bar += g_k

    impact = g_bar * (1 + 1 / n_classes) / (n_classes * batch_size)
    offset = offset / ((K - 1) * count)
    return impact, offset



def attack_llg_plus(original_model, unlearned_model, proxy_gradients, lr ,aux_loader, batch_size, num_classes=10):
    """
    LLG+ Attack (Correct implementation of Algorithm 1).
    
    Args:
        proxy_gradients: Dict gradients.
        m_impact: Scalar float (Impact).
        s_offset_vector: Numpy array hoặc Tensor shape [num_classes] (Offset).
    """
    impact, offset = estimate_static_LLG(copy.deepcopy(original_model), aux_data = aux_loader, n_classes = num_classes, batch_size = batch_size)
    new_impact, new_offset = estimate_static_LLG( copy.deepcopy(unlearned_model), aux_data = aux_loader, n_classes = num_classes, batch_size = batch_size)
    new_impact = (new_impact + impact) / 2
    new_offset = (new_offset + offset) / 2
    grad_vector = None
    # Tìm weight layer cuối
    for name in reversed(list(proxy_gradients.keys())):
        if 'weight' in name and len(proxy_gradients[name].shape) == 2:
            if proxy_gradients[name].shape[0] == num_classes:
                w_grad = proxy_gradients[name]
                grad_vector = torch.sum(w_grad,  dim=-1).detach().clone()
                break

    h1_extraction = []
    gradients_for_prediction = grad_vector/lr
    print(gradients_for_prediction)
    print(new_offset)  
    for i_cg, class_gradient in enumerate(gradients_for_prediction):
                    if class_gradient < 0:
                        h1_extraction.append((i_cg, class_gradient))

    gradients_for_prediction -= new_offset
    prediction = []

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)

    for _ in range(batch_size - len(prediction)):
        # add minimal candidate, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one occurrence to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

    n = []
    for i in range(num_classes):
        n.append(prediction.count(i))

    predicted_labels = []
    for cls_idx in range(num_classes):
        c = n[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
    
    return sorted(predicted_labels)