import transformers
import torch
import time  # Import the time module
from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn.utils import prune
from evaluate import evaluate_2, get_model_size, get_compressed_model_size
from utils import pseudo_quantize_model_salient_weight_fp16, get_calib_feat, pseudo_quantize_model_weight_scaleup, pseudo_quantize_model_weight_auto_scale

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

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
    #load_in_4bit=True, # int
    device_map="auto"
)

# raw model size
#model_size = get_model_size(model, data_width=32, group_size=128)
#print(f"Raw model size: {model_size/MiB:.2f} MiB")

#prune_model(model, amount=0.3)  # Prune 50% of weights in linear layers

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_feat = get_calib_feat(model, tokenizer)
awq_ratio = 0.03
w_bit = 8
#pseudo_quantize_model_salient_weight_fp16(model, w_bit=w_bit, q_group_size=128, input_feat=input_feat, ratio=awq_ratio)
#pseudo_quantize_model_weight_scaleup(model, w_bit=w_bit, q_group_size=128, input_feat=input_feat, scale_factor=2)
pseudo_quantize_model_weight_auto_scale(model, w_bit=w_bit, q_group_size=128, input_feat=input_feat)

num_tokens = 50
start_time = time.time()
# Create a pipeline with the quantized and pruned model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=num_tokens)

# Measure time after loading the pipeline
load_time = time.time() - start_time
print(f"Pipeline loaded in {load_time:.2f} seconds.")

# Start the timer before generating text
start_time = time.time()

# Run the pipeline
result = pipe("Hey how are you doing today?", max_length=50)

# Measure time after generating text
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds.")
time_per_token = inference_time / num_tokens
print(f"Inference time per token: {time_per_token:.4f} seconds")


# Print the result
print(result)

# ==== Evaluation =====

perplexity = evaluate_2(model, tokenizer)
print(f"Perplexity: {perplexity:.2f}")

# compressed model size
model_size = get_compressed_model_size(model, data_width=w_bit)
print(f"Compressed model size: {model_size/MiB:.2f} MiB")