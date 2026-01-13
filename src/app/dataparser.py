import json
import torch
import pandas as pd
from transformers import AutoTokenizer

def parseJsonl(base, filenames, current_tokenizer):
    shard_idx = 0
    batch_idx = 0
    count = 0
    try:
        for name in filenames:
            token_vals = []
            df = pd.DataFrame(columns=['tensor', 'batch', 'shard'])
            current_file_name = f"{base}/{name}" 
            df_idx = 0
            with open(current_file_name) as file:
                for line in file:
                    current_json = json.loads(line)
                    token_vals.extend(current_tokenizer.encode(f"<!~start_sentence>{current_json['text']}<!~end_sentence/>"))
                    if len(token_vals) > 4000:
                        tensor = torch.tensor(token_vals[:4000], dtype=torch.int32)
                        print(f"{current_file_name} : tensor : {tensor.shape}, batch : {batch_idx}, shard : {shard_idx}")
                        df.loc[df_idx] = [tensor.numpy(), batch_idx, shard_idx]
                        df_idx += 1
                        batch_idx += 1
                        if batch_idx != 0 and batch_idx % 8 == 0:
                            batch_idx = 0
                            shard_idx += 1
                            if shard_idx != 0 and shard_idx % 8 == 0:
                                shard_idx = 0
                paraquet_filename = f"dataset/mathematics/parquets/{count:06d}.parquet"
                df.to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
                count += 1
    except Exception as e:
        print(f"An error occurred: {e}")

def readParquet(base, file_name, tokenizer):
    read_df = pd.read_parquet(f"{base}/{file_name}")
    print(len(read_df))



if __name__ == "__main__":
    base = "dataset/mathematics"
    file_names = ["mathematics_000000.jsonl","mathematics_000001.jsonl",
                  "mathematics_000002.jsonl","mathematics_000003.jsonl",
                  "mathematics_000004.jsonl","mathematics_000005.jsonl",
                  "mathematics_000006.jsonl","mathematics_000007.jsonl",
                  "mathematics_000008.jsonl","mathematics_000009.jsonl",
                  "mathematics_000010.jsonl","mathematics_000011.jsonl"]
    current_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", extra_special_tokens={"eos":"<!~start_sentence>","bos":"<!~end_sentence/>"})
    # parseJsonl(base, file_names, current_tokenizer)
   