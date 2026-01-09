import json
import torch
from transformers import AutoTokenizer

def parseJsonl(filename, current_tokenizer):
    with open(filename) as file:
        for line in file:
            current_json = json.loads(line)
            tensor = torch.tensor(current_tokenizer.encode(f"<!~start_sentence>{current_json['text']}<!~end_sentence/>"))
            print(tensor.shape)


if __name__ == "__main__":
    base = "dataset/mathematics"
    file_names = ["mathematics_000000.jsonl"]
    current_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", extra_special_tokens={"eos":"<!~start_sentence>","bos":"<!~end_sentence/>"})
    for name in file_names:
        parseJsonl(f"{base}/{name}", current_tokenizer)