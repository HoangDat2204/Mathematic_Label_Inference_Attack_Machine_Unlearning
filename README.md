# Mathematic_Label_Inference_Attack_Machine_Unlearning
"python prepare_model.py --dataset cifar10 --model ResNet18 --pretrain_epochs 10 --finetune_epochs 5"
"python main_attack.py --dataset cifar10 --model ResNet18 --batch_size 8 --aux_size 200 --unlearn_epochs 1 --exact_epochs 1 --total_loops 5 --lr 0.01 --alpha 1000"