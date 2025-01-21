import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import torch

# Step 1: Fetch the token from the environment and login
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("HUGGINGFACE_TOKEN not set in environment variables.")

# Step 2: Set the model ID and load the pre-trained LLaMA model
model_id = "meta-llama/Llama-3.1-8B"  # Update with your desired model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Enable multi-GPU with DataParallel
#if torch.cuda.device_count() > 1:
#    model = torch.nn.DataParallel(model)

# Add padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if isinstance(model, torch.nn.DataParallel):
        model.module.resize_token_embeddings(len(tokenizer))  # Access underlying model
    else:
        model.resize_token_embeddings(len(tokenizer))

print('--- model loaded ---')
# Step 3: Load and Preprocess Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print('--- dataset loaded ---')

def preprocess_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()  # Copy input_ids as labels for causal LM
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

# Step 4: Apply LoRA for Parameter-Efficient Fine-Tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=2,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False
)

# Step 6: Set Up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# Step 7: Fine-Tune the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuning completed and model saved!")
