
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
    nsamples = 40
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

