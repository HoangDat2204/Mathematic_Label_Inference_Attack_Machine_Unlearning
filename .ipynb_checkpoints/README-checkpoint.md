# Mathematic_Label_Inference_Attack_Machine_Unlearning
python prepare_model.py --dataset cifar10 --model ResNet18 --pretrain_epochs 10 --finetune_epochs 5


#NegGrad Unlearn Method
python main_attack.py --dataset cifar10 --model ResNet18 --unlearned_algo neggrad  --aux_size 200 --total_loops 5 --unlr 0.01 --alpha 1 --batch_size 8 --seed 0


#Finetune Unlearn Method
python main_attack.py --dataset cifar10 --model ResNet18 --unlearned_algo finetuning  --aux_size 200 --total_loops 5 --unlr 0.01 --alpha 1 --batch_size 8 --seed 0


#Scrub Unlearn Method
python main_attack.py --dataset cifar10 --model ResNet18 --unlearned_algo scrub  --aux_size 200  --total_loops 5 --unlr 0.01 --alpha 1 --batch_size 8 --seed 0


#NegGrad+ Unlearn Method
python main_attack.py --dataset cifar10 --model ResNet18 --unlearned_algo neggradp  --aux_size 200 --total_loops 5 --unlr 0.01 --alpha 1 --batch_size 8 --seed 0

#Retrain Unlearn Method



python main_attack.py --dataset cifar10 --model ResNet18 --unlearned_algo retrain --pretrain_epochs 20 --pretrain_lr 0.01 --aux_size 200 --total_loops 5 --alpha 1 --batch_size 8 --seed 0