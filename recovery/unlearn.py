# File: recovery/unlearn.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Subset, DataLoader
from configs import Config

class Unlearner:
    def __init__(self, target_model, base_model, device='cuda'):
        self.target_model = target_model
        self.base_model = base_model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def approximate_unlearn(self, list_of_batches, lr=0.01):
        """
        Thực hiện Unlearning tuần tự trên nhiều batch (User gọi là nhiều epoch).
        Mỗi bước unlearn một batch ảnh khác nhau.
        
        Input:
            list_of_batches: List chứa các tuple (images, labels). 
                             Ví dụ: [(imgs1, lbls1), (imgs2, lbls2), ...]
                             Độ dài list chính là số 'epoch' bạn muốn.
        Output:
            model: Model sau khi đã unlearn xong toàn bộ chuỗi batch.
        """
        # Luôn bắt đầu từ Target Model (M_finetuned)
        model = copy.deepcopy(self.target_model)
        model.train()
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Unlearn tuần tự: M0 -> M1 -> M2 ...
        for i, (images, labels) in enumerate(list_of_batches):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            
            # Gradient Ascent: Maximize Loss để quên dữ liệu
            loss = -loss 
            loss.backward()
            optimizer.step()
            
            # (Optional) Có thể thêm log loss tại đây để xem nó tăng lên thế nào
            
        return model

    def exact_unlearn(self, full_dataset, indices_to_remove, epochs=5, lr=0.001):
        """
        Exact Unlearn: Loại bỏ hoàn toàn tất cả các ảnh trong indices_to_remove
        (Tổng hợp của tất cả các batch trong chuỗi)
        """
        # 1. Tạo dataset mới: D_new = D_full - {All 80 images}
        all_indices = set(range(len(full_dataset)))
        remove_indices = set(indices_to_remove)
        keep_indices = list(all_indices - remove_indices)
        
        sub_dataset = Subset(full_dataset, keep_indices)
        loader = DataLoader(sub_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        # 2. Load Base Model (Pretrained on Retain Set)
        model = copy.deepcopy(self.base_model)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        # 3. Finetune lại
        print(f"   [Exact Sim] Retraining on {len(keep_indices)} samples for {epochs} epochs...")
        for _ in range(epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        return model

def get_weight_difference(model_orig, model_new):
    diff_dict = {}
    state_orig = model_orig.state_dict()
    state_new = model_new.state_dict()
    for k in state_orig.keys():
        if 'weight' in k or 'bias' in k:
            diff_dict[k] = (state_new[k] - state_orig[k]).cpu().detach()
    return diff_dict