
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
#from functools import partial
#import gc

def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // 2048
    #nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

def evaluate_2(model, tokenizer, nsamples=40, seq_length=2048):
    # Load the Wikitext-2 test dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Tokenize the dataset and handle large inputs
    tokenized_text = tokenizer("\n\n".join(dataset['text']), return_tensors='pt', truncation=False)
    input_ids = tokenized_text['input_ids'].to(model.device)
    nsamples = input_ids.numel() // 2048
    # Ensure model is in evaluation mode
    model.eval()

    # Initialize variables to store losses
    nlls = []

    # Iterate over the dataset in chunks
    for i in tqdm.tqdm(range(nsamples), desc="Evaluating..."):
        # Determine start and end indices for the batch
        start_idx = i * seq_length
        end_idx = (i + 1) * seq_length

        if start_idx >= input_ids.size(1):
            break  # End of dataset

        # Slice the input_ids for this batch
        batch = input_ids[:, start_idx:end_idx]
        
        # Skip if the batch size is smaller than seq_length
        if batch.size(1) < seq_length:
            continue

        # Compute loss for the batch
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            loss = outputs.loss  # Cross-entropy loss for the batch
        
        # Convert to negative log-likelihood and scale by sequence length
        neg_log_likelihood = loss * batch.size(1)
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * seq_length
    perplexity = torch.exp(total_nll / total_tokens)

    return perplexity.item()

def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

def get_compressed_model_size(model: nn.Module, data_width=16):
    num_elements = 0
    for param in model.parameters():
        num_elements += param.nonzero().size(0)  # Count non-zero weights
    return num_elements * data_width

def get_model_info(model: nn.Module):

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()

    print('Parameter number: ', num_elements)

    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")

def get_quantized_model_size(model: nn.Module, data_width=16, quantized_data_width=8, group_size=-1):
    """
    Calculate the model size in memory or storage, accounting for quantized and non-quantized parameters.

    Args:
        model (nn.Module): The model to analyze.
        data_width (int): Bit-width for non-quantized parameters (default: 16 for fp16).
        quantized_data_width (int): Bit-width for quantized parameters (default: 8 for int8).
        group_size (int): Group size for quantization (default: -1 for no grouping).

    Returns:
        float: Estimated model size in bytes.
    """

    # Adjust quantized data width for grouping overhead
    if group_size != -1:
        quantized_data_width += (16 + 4) / group_size

    # Convert data widths from bits to bytes
    #data_width /= 8
    #quantized_data_width /= 8

    total_size = 0

    # Iterate over model parameters
    for name, param in model.named_parameters():
        # Check if the parameter belongs to an nn.Linear layer
        is_linear_weight = any(
            isinstance(module, nn.Linear) and hasattr(module, "weight") and param is getattr(module, "weight", None)
            for module in model.modules()
        )

        if is_linear_weight:
            # Apply quantized size for Linear layer weights
            total_size += param.numel() * quantized_data_width
        else:
            # Apply regular size for non-quantized parameters
            total_size += param.numel() * data_width

    return total_size


def get_pruned_quantized_model_size(
    model: nn.Module,
    data_width: int = 16,
    quantized_data_width: int = 8,
    group_size: int = -1,
    exclude_pruned_zeros: bool = False
) -> float:
    """
    Calculate the model size in bits (or bytes) accounting for quantized and non-quantized parameters,
    with optional exclusion of zeroed-out weights for unstructured pruning.

    Args:
        model (nn.Module): The model to analyze.
        data_width (int): Bit-width for non-quantized parameters (default: 16 for FP16).
        quantized_data_width (int): Bit-width for quantized parameters (default: 8 for int8).
        group_size (int): Group size for quantization (default: -1 for no grouping).
        exclude_pruned_zeros (bool): If True, and if pruning is unstructured (zeroes),
                                      do not count zeroed weights toward the size. 
                                      This simulates a sparse storage scenario.

    Returns:
        float: Estimated model size in bits (or bytes, depending on usage).
    """

    # If your original code wants the result in *bytes*, 
    # convert data_width from bits -> bytes: data_width /= 8
    # But if you want the result directly in *bits*, keep as is.
    # For clarity here, I'll assume we are counting bits.
    # If you prefer bytes, just do data_width /= 8, etc.
    # data_width /= 8
    # quantized_data_width /= 8

    # Adjust quantized data width for grouping overhead if relevant
    # (e.g., storing extra metadata per group)
    if group_size != -1:
        # Example overhead: 16 bits for scale, 4 bits for zero_point, per group
        # This is just a conceptual example; adjust as needed.
        overhead_per_group = 16 + 4  # bits
        quantized_data_width += overhead_per_group / group_size

    total_size = 0.0

    # Identify the set of linear weight parameters (to treat them as quantized)
    linear_weight_params = set()
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
            linear_weight_params.add(module.weight)

    for param_name, param in model.named_parameters():
        # Check if this parameter is one of the linear weights
        if param in linear_weight_params:
            # Quantized parameter
            if exclude_pruned_zeros:
                # Count only nonzero weights if we want to exclude pruned zeros
                num_elems = param.nonzero().size(0)
            else:
                # Otherwise, count all (including zeroed weights)
                num_elems = param.numel()
            total_size += num_elems * quantized_data_width
        else:
            # Normal (non-quantized) parameter
            if exclude_pruned_zeros:
                # Same logic: only count nonzero if you want to ignore zeros
                num_elems = param.nonzero().size(0)
            else:
                num_elems = param.numel()
            total_size += num_elems * data_width

    return total_size
