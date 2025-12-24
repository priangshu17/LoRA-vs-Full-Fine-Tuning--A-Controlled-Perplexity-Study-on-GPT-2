from datasets import load_dataset

def prepare_dataset():
    dataset = load_dataset("OpenAssistant/oasst1")
    
    def extract_text(example):
        if example["role"] == "assistant":
            return {"text": example["text"]}
        return {"text": ""}
    
    dataset = dataset["train"].map(extract_text)
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    dataset.save_to_disk("data/oasst_preprocessed")
    
if __name__ == "__main__":
    prepare_dataset()