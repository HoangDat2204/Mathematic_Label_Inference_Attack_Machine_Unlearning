# File: configs.py
import os
import torch

class Config:
    # --- PATHS ---
    DATA_PATH = './data'          # Nơi lưu data gốc
    RESULT_PATH = './results'     # Nơi lưu kết quả
    MODEL_SAVE_PATH = './results/models'
    
    # --- DATA SETTINGS ---
    FORGET_SIZE = 10000  # Số lượng dữ liệu dùng để Finetune và sau đó Unlearn
    
    # --- TRAINING SETTINGS ---
    BATCH_SIZE = 128
    
    # Learning Rates (Tham khảo từ setup chuẩn của FL/Unlearning)
    LR_PRETRAIN = 0.01   # Learning rate cho giai đoạn train tập lớn
    LR_FINETUNE = 0.001  # Learning rate nhỏ hơn cho giai đoạn finetune 10k
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # --- SYSTEM ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2

    @staticmethod
    def ensure_dirs():
        """Tạo các thư mục cần thiết nếu chưa có"""
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.DATA_PATH, exist_ok=True)

# Tự động tạo thư mục khi import file này
Config.ensure_dirs()