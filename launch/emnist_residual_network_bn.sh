# This takes 6 hours
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=7_1.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 7 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=7_1.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 7 --save_json

python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=7_2.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 7 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=7_2.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 7 --save_json


python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=11_1.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 11 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=11_1.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 11 --save_json

python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=11_2.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 11 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=11_2.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 11 --save_json


python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=15_1.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 15 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=15_1.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 15 --save_json

python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=15_2.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 15 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=15_2.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 15 --save_json


python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=19_1.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 19 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/1 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=19_1.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 80 140 --epochs 160 --problem classification --width 4 --depth 19 --save_json

python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log recur_residual_network_width=4_depth=19_2.txt --model recur_residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --width 4 --depth 19 --save_json
python train_model.py --checkpoint checkpoints/emnist_residual_network_bn/2 --output results/emnist_residual_network_bn --train_log residual_network_width=4_depth=19_2.txt --model residual_network_bn --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 80 140 --epochs 160 --problem classification --width 4 --depth 19 --save_json


#python train_model.py --checkpoint checkpoints/emnist_resnet_bn/1 --output results/emnist_resnet_bn --train_log resnet18_1.txt --model resnet18_emnist --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --save_json
#python train_model.py --checkpoint checkpoints/emnist_resnet_bn/2 --output results/emnist_resnet_bn --train_log resnet18_2.txt --model resnet18_emnist --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --save_json
#python train_model.py --checkpoint checkpoints/emnist_resnet_bn/3 --output results/emnist_resnet_bn --train_log resnet18_3.txt --model resnet18_emnist --dataset emnist --val_period 20 --lr 0.1 --lr_schedule 60 120 --epochs 160 --problem classification --save_json
