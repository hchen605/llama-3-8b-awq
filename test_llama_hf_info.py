import transformers
import torch
import time  # Import the time module
from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn.utils import prune
from evaluate import evaluate_2, get_model_size, get_compressed_model_size, get_model_info, get_quantized_model_size, get_pruned_quantized_model_size
from utils import prune_model
import argparse

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Find LLM size under pruning and quantization.")

parser.add_argument('--prune_ratio', type=float, default=0.3, help='prune_ratio')
parser.add_argument('--w_bit', type=int, default=4, help='quantization bits')

args = parser.parse_args()

# Fetch the token from the environment
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("HUGGINGFACE_TOKEN not set in environment variables.")

model_id = "meta-llama/Llama-3.1-8B"

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #load_in_4bit=True, # 8bit int
    device_map="auto"
)
#load_in_8bit (bool, optional, defaults to False) — This flag is used to enable 8-bit quantization with LLM.int8().
#load_in_4bit (bool, optional, defaults to False) — This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
prune_model(model, amount=args.prune_ratio)

# raw model size
model_size = get_model_size(model, data_width=32, group_size=128)
print(f"Raw model size: {model_size/MiB:.2f} MiB")

# model_size = get_quantized_model_size(model, data_width=32, quantized_data_width=8, group_size=128)
# print(f"8 bit Quantized model size: {model_size/MiB:.2f} MiB")

print(f"Ideal size ignoring unstructured pruned zeros: ratio: {args.prune_ratio*100}%, quantization {args.w_bit} bit")
model_size = get_pruned_quantized_model_size(model, data_width=32, quantized_data_width=args.w_bit, exclude_pruned_zeros=True)
print(f"{model_size/MiB:.2f} MiB")



#get_model_info(model)