#!/bin/bash
#SBATCH -J job_id
#SBATCH -o output-file.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon08 #YOUR NODE OF PREFERENCE

# Set Hugging Face token
export HUGGINGFACE_TOKEN=YOUR-TOKEN

module load shared singularity

# Execute the container
singularity exec --nv ../python_torch.img nvidia-smi
singularity exec --nv llama_hf_finetune.img python3 finetune_llama.py
