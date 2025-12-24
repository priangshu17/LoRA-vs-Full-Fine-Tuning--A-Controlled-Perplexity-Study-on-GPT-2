import torch 
import math 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk

def compute_ppl(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path).cuda()
    model.eval()
    
    
    dataset = load_from_disk("data/oasst_preprocessed").select(range(1000))
    
    losses = []
    
    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors = "pt", truncation = True, max_length = 512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        
        with torch.no_grad():
            outputs = model(**inputs, labels = inputs["input_ids"])
            losses.append(outputs.loss.item())
            
    return math.exp(sum(losses) / len(losses))


if __name__  == "__main__":
    print("Full FT PPL:", compute_ppl("outputs/full_ft/final"))
    print("LoRA PPL:", compute_ppl("outputs/lora/final"))