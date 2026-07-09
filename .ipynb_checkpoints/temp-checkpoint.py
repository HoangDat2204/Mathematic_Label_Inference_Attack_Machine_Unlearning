import shutil
import os

# Đường dẫn file gốc
src = '/kaggle/input/datasets/luyenhunglam/train-resnet18-cifar10/Mathematic_Label_Inference_Attack_Machine_Unlearning/results/models/ResNet18_cifar10_pretrained.pth'

# Đường dẫn file đích
dst = '/kaggle/working/Mathematic_Label_Inference_Attack_Machine_Unlearning/results/ResNet18_cifar10_pretrained.pth'

# 1. Tạo các thư mục cha nếu chúng chưa tồn tại
os.makedirs(os.path.dirname(dst), exist_ok=True)

# 2. Thực hiện copy file
# shutil.copy2 sẽ giữ nguyên metadata (thời gian tạo, chỉnh sửa...) của file
shutil.copy2(src, dst)

print(f"Đã copy file thành công vào: {dst}")