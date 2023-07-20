torchrun --nnodes=1 --nproc_per_node=3 --standalone train_ddp.py -c ./configs/8e4_ddp.yaml -ws 3
torchrun --nnodes=1 --nproc_per_node=3 --standalone train_ddp.py -c ./configs/2e4_ddp.yaml -ws 3
torchrun --nnodes=1 --nproc_per_node=3 --standalone train_ddp.py -c ./configs/4e4_ddp.yaml -ws 3
