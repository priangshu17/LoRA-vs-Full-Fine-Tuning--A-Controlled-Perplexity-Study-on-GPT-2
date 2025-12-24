from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_from_disk

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 


dataset = load_from_disk("data/oasst_preprocessed")


def tokenize_fn(batch):
    tokenized = tokenizer(
        batch["text"], 
        truncation = True, 
        padding = "max_length", 
        max_length = 256,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


dataset = dataset.map(tokenize_fn, batched = True, remove_columns = ["text"])

model = GPT2LMHeadModel.from_pretrained(model_name)

args = TrainingArguments(
    output_dir = "outputs/full_ft",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate = 5e-5,
    num_train_epochs = 2,
    fp16 = True,
    logging_steps = 100,
    save_strategy = "epoch"
)

print(dataset)
print(dataset.column_names)
print(dataset[0])

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
    data_collator = default_data_collator,
)

trainer.train()
trainer.save_model("outputs/full_ft/final")