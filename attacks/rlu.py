import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy


def getsize(list_input):
    rows = len(list_input) 
    cols = len(list_input[0])
    return f"{rows} x {cols}"


def learn_stat( k, n, predictions, ground_truths, batch_size):
    mis_predictions = []
    for i in range(len(predictions) - 1):
        for j in range(batch_size):
            if ground_truths[i][j] == n:
                mis_predictions.append(predictions[i][j][k])

    if len(mis_predictions) == 0:
        mis_predictions.append(0)

    return np.array(mis_predictions)


def matrix( predictions, ground_truths, batch_size, n_classes):
    mis_predictions_maxrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            mis_predictions_maxrix[i][j] = np.mean(learn_stat(i, j, predictions, ground_truths, batch_size))

    return mis_predictions_maxrix



def estimate_static_RLU( model, aux_dataset, batch_size, n_classes):
    model.train()
    aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    predictions = []
    predictions_softmax = []
    ground_truths = []
    count = 0 
    for batch_idx, (inputs, targets) in enumerate(aux_loader):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        count += 1 

        # compute output
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=-1)
        ground_truths.append(np.array(targets.detach().cpu()))
        predictions.append(np.array(outputs.detach().cpu()))
        predictions_softmax.append(np.array(probs.detach().cpu()))
        # print("="*60)
        # print("Aux Data batchidx: ", batch_idx)
        # print("Logit vector: ",  len(outputs))
        # print("ground_truths vector: ",  getsize(ground_truths))
        # print("predictions vector: ",  getsize(predictions))
        # print("predictions_softmax vector: ",  getsize(predictions_softmax))



    mis_predictions_maxrix = matrix(predictions, ground_truths, batch_size, n_classes )
    mis_predictions_softmax = matrix(predictions_softmax, ground_truths, batch_size, n_classes)
    mu = np.zeros(n_classes)
    for i in range(n_classes):
        mu[i] = (np.sum(mis_predictions_maxrix[i]) - mis_predictions_maxrix[i, i]) / (n_classes - 1)
    shift = np.zeros(n_classes)
    for i in range(n_classes):
        shift[i] = (np.sum(mis_predictions_softmax[i]) - mis_predictions_softmax[i, i]) / (n_classes - 1)
    return mu, shift





def estimated_entropy_from_grad(shift, bias, batch_size, n_classes ):
    n = n_classes
    solution = [0] * n
    bias = -np.array(bias)



    n = [0] * n_classes
    for i in range(n_classes):
        bias[i] = bias[i] + shift[i]

    for i in range(n_classes):
        if bias[i] < 0:
            bias[i] = 0

    s = np.sum(abs(bias))
    for i in range(n_classes):

        bias[i] = bias[i] / s
        solution[i] = round(bias[i] * batch_size)

    return solution



def attack_rlu_full(model_original, model_unlearned, proxy_update, aux_loader, 
                    batch_size, lr, num_epochs=1, num_classes=10, device='cpu'):
    """
    Triển khai tấn công RLU để khôi phục nhãn từ cập nhật cục bộ.
    
    Args:
        proxy_update: lr * gradient.
    
    Nguồn: Algorithm 1 [2].
    """
    model_original = model_original.to(device)
    model_unlearned = model_unlearned.to(device)
    
    model_before = copy.deepcopy(model_original)
    model_after = copy.deepcopy(model_unlearned)


    target_update = None
    # Tìm bias layer cuối
    for name in reversed(list(proxy_update.keys())):
        if 'bias' in name and proxy_update[name].shape[0] == num_classes:
            target_update = proxy_update[name].detach().cpu().numpy()
            break
            
    if target_update is None:
        print("[RLU Error] Không tìm thấy Bias lớp cuối.")
        return []

  
    u = target_update / lr
    # 3. Tính toán ma trận S và A tại trạng thái t (model gốc)
    mu, shift = estimate_static_RLU(model_before, aux_loader.dataset, batch_size, num_classes)
    new_shift = scipy.special.softmax(mu)

    n = estimated_entropy_from_grad( new_shift, u, batch_size, num_classes)
    
    predicted_labels = []
    for cls_idx in range(num_classes):
        c = n[cls_idx]
        if c > 0:
            predicted_labels.extend([cls_idx] * c)
    
    return sorted(predicted_labels)


