from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_from_disk("data/oasst_preprocessed")

def tokenize_fn(batch):
    tokenized = tokenizer(
        batch["text"], 
        truncation = True, 
        padding = "max_length", 
        max_length = 512,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize_fn, batched = True, remove_columns = ["text"])

model = GPT2LMHeadModel.from_pretrained(model_name)


lora_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    target_modules = ["c_attn", "c_proj"],  # QKV Projection
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


args = TrainingArguments(
    output_dir = "outputs/lora",
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate = 3e-4,
    num_train_epochs = 2,
    fp16 = True,
    logging_steps = 100,
    save_strategy = "epoch"
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
)

trainer.train()
trainer.save_model("outputs/lora/final")