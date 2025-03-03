# llama-3-8b-awq

This is the practice for Activation-aware Weight Quantization (AWQ) on LLM (Llama 3.1 8B) Compression and Acceleration. A quick test to show usability.

Some codes are editted and modified from the author's original [repo](https://github.com/mit-han-lab/llm-awq).


## Usage

### Pure puring + quantization

```
python test_llama_hf_pr_q.py
```

### AWQ test
```
python test_llama_hf_awq.py
```

### Finetune with LoRA
```
python finetune_llama.py
```

### Check Model Size
```
python3 test_llama_hf_info.py --prune_ratio 0.3 --w_bit 32

```

