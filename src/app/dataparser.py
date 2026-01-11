import json
import torch
import pandas as pd
from transformers import AutoTokenizer

def parseJsonl(filename, current_tokenizer):
    token_vals = []
    shard_idx = 0
    batch_idx = 0
    df = pd.DataFrame(columns=['tensor', 'batch', 'shard'])
    count = 0
    df_idx = 0
    with open(filename) as file:
        for line in file:
            current_json = json.loads(line)
            token_vals.extend(current_tokenizer.encode(f"<!~start_sentence>{current_json['text']}<!~end_sentence/>"))
            if len(token_vals) > 4000:
                tensor = torch.tensor(token_vals[:4000], dtype=torch.int32)
                print(f"tensor : {tensor.shape}, batch : {batch_idx}, shard : {shard_idx}")
                df.loc[df_idx] = [tensor.numpy(), batch_idx, shard_idx]
                df_idx += 1
                batch_idx += 1
                if batch_idx != 0 and batch_idx % 8 == 0:
                    batch_idx = 0
                    shard_idx += 1
                    if shard_idx != 0 and shard_idx % 8 == 0:
                        shard_idx = 0
        try:
            width = 5
            paraquet_filename = f"dataset/mathematics/parquets/{count:0{width}}.parquet"
            df.to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
            print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
            count += 1
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    base = "dataset/mathematics"
    file_names = ["mathematics_000000.jsonl"]
    current_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", extra_special_tokens={"eos":"<!~start_sentence>","bos":"<!~end_sentence/>"})
    for name in file_names:
        parseJsonl(f"{base}/{name}", current_tokenizer)