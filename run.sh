#!/bin/bash
#SBATCH -J job_id
#SBATCH -o evaluate-8bit-awq-autoscale.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon08 #YOUR NODE OF PREFERENCE

# Set Hugging Face token
export HUGGINGFACE_TOKEN=YOUR-TOKEN

module load shared singularity

# Execute the container
singularity exec --nv img/llama_awq.img python3 test_llama_hf_awq.py

# llama_hf.img for raw and pure compression
