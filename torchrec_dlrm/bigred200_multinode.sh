#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=bigred_%j.log 
#SBATCH --mem=200G

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
export TRACE_PATH="/N/scratch/haofeng/trace_result/8x4"
export SAVE_PATH="/N/scratch/haofeng/dlrm_models/"
export BATCH_SIZE=8192
cd ~/new_dlrm/torchrec_dlrm/
source ~/.bashrc
conda init
conda activate dlrm
srun -n 16 python dlrm_main.py --epochs=$EPOCH \
--in_memory_binary_criteo_path="/N/scratch/haofeng/TB/processed" \
--num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" \
--embedding_dim 128 \
--batch_size $BATCH_SIZE \
--over_arch_layer_sizes "1024,1024,512,256,1" \
--dense_arch_layer_sizes "512,256,128" \
--shuffle_batches \
--print_sharding_plan \
--save_path=$SAVE_PATH \
--trace_path=$TRACE_PATH