mkdir results_grad_acc
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/single_gpu_no_grad_acc.yaml
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/single_gpu_grad_acc_2.yaml
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/single_gpu_grad_acc_4.yaml
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/2_gpu_dp_grad_acc_2.yaml
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/2_gpu_dp_grad_acc_4.yaml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 helpers/train.py configs/2_gpu_ddp_grad_acc_2.yaml --distributed_launch --distributed_backend=nccl
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 helpers/train.py configs/2_gpu_ddp_grad_acc_4.yaml --distributed_launch --distributed_backend=nccl

CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/stop_by_iter_no_grad_acc.yaml
CUDA_VISIBLE_DEVICES=0,1 python helpers/train.py configs/stop_by_iter_grad_acc_2_dp.yaml
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 helpers/train.py configs/stop_by_iter_grad_acc_4_ddp.yaml --distributed_launch --distributed_backend=nccl

python grad_acc_check.py
