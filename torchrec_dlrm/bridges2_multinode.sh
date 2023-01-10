#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p GPU
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH --output=output_%j.log 
#SBATCH --gpus=v100-16:16

#module load nvidia
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
#export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH

#echo $LD_LIBRARY_PATH


export MASTER_PORT=47149
export WORLD_SIZE=16
echo "WORLD_SIZE="$WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export EPOCH=100
export DATASET_PATH="/ocean/projects/asc200010p/haofeng1/criteo_TB_processed/"
cd ~/new_dlrm/torchrec_dlrm/
source ~/.bashrc
conda init
conda activate dlrm
srun -n 16 python dlrm_main.py --epochs=$EPOCH \
--in_memory_binary_criteo_path=$DATASET_PATH \
--num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" \
--embedding_dim 128 \
--batch_size 4096 \
--over_arch_layer_sizes "1024,1024,512,256,1" \
--dense_arch_layer_sizes "512,256,128" \
--shuffle_batches \
--print_sharding_plan \
--save_path="/N/scratch/haofeng/dlrm_models"
