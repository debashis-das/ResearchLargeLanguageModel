import json
import torch
import pandas as pd
from transformers import AutoTokenizer
import random

def parseJsonl(base, filenames, current_tokenizer, train=True):
    shard_idx = 0
    batch_idx = 0
    token_idx = 0
    count = 0
    try:
        df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
        pad_id = current_tokenizer.pad_token_id
        for name in filenames:
            token_vals = []
            current_file_name = f"{base}/{name}" 
            with open(current_file_name) as file:
                for line in file:
                    current_json = json.loads(line)
                    if train:
                        token_vals.extend(current_tokenizer.encode(f"<!~start_sentence>{current_json['text']}<!~end_sentence/>"))
                    else:
                        index = random.randint(1, 32000)
                        current_tokens = current_tokenizer.encode(f"<!~start_sentence>{current_json['text']}<!~end_sentence/>")[:index]
                        pad_length = 32000 - len(current_tokens)
                        pad_tensor = torch.full((pad_length,), pad_id)
                        token_vals.extend(current_tokens[:index])
                        token_vals.extend(pad_tensor)
                        print(f"Decoded ({len(token_vals)}) : {current_tokenizer.decode(token_vals)}")
                    if len(token_vals) > 32000:
                        for shard_idx in range(8):
                            tensor = torch.tensor(token_vals[4000*shard_idx:4000*shard_idx+4000], dtype=torch.int32)
                            print(f"{current_file_name} : tensor : {tensor.shape}, batch : {batch_idx}, shard : {shard_idx}, token : {token_idx}")
                            df[shard_idx].loc[len(df[shard_idx])-1] = [tensor.numpy(), batch_idx, shard_idx, token_idx]
                        batch_idx += 1
                        if batch_idx != 0 and batch_idx % 8 == 0:
                            batch_idx = 0
                        token_idx += 1
                    if len(df[0]) == 100000:
                        for shard_idx in range(8):
                            if train:
                                paraquet_filename = f"dataset/mathematics/parquets/{shard_idx}/{count:06d}.parquet"
                            else:
                                paraquet_filename = f"dataset/mathematics/parquets/test/{shard_idx}/{count:06d}.parquet"
                            df[shard_idx].to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                            print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
                        count += 1
                        token_idx = 0
                        df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
    except Exception as e:
        print(f"An error occurred: {e}")

def readParquet(base, file_name, tokenizer):
    read_df = pd.read_parquet(f"{base}/{file_name}")
    print(read_df.loc[len(read_df)-1]['shard'])
    print(read_df.loc[len(read_df)-1]['batch'])




if __name__ == "__main__":
    base = "dataset/mathematics"
    file_names = ["mathematics_000000.jsonl","mathematics_000001.jsonl",
                  "mathematics_000002.jsonl","mathematics_000003.jsonl",
                  "mathematics_000004.jsonl","mathematics_000005.jsonl",
                  "mathematics_000006.jsonl","mathematics_000007.jsonl",
                  "mathematics_000008.jsonl","mathematics_000009.jsonl",
                  "mathematics_000010.jsonl","mathematics_000011.jsonl",
                  "mathematics_000012.jsonl","mathematics_000013.jsonl",
                  "mathematics_000014.jsonl","mathematics_000015.jsonl",
                  "mathematics_000016.jsonl","mathematics_000017.jsonl",
                  "mathematics_000018.jsonl","mathematics_000019.jsonl",
                  "mathematics_000020.jsonl","mathematics_000021.jsonl",
                  "mathematics_000022.jsonl","mathematics_000023.jsonl",
                  "mathematics_000024.jsonl","mathematics_000025.jsonl",
                  "mathematics_000026.jsonl","mathematics_000027.jsonl",
                  "mathematics_000028.jsonl","mathematics_000029.jsonl",
                  "mathematics_000030.jsonl","mathematics_000031.jsonl",
                  "mathematics_000032.jsonl","mathematics_000033.jsonl",
                  "mathematics_000034.jsonl","mathematics_000035.jsonl",
                  "mathematics_000036.jsonl","mathematics_000037.jsonl",
                  "mathematics_000038.jsonl","mathematics_000039.jsonl",
                  "mathematics_000040.jsonl","mathematics_000041.jsonl",
                  "mathematics_000042.jsonl","mathematics_000043.jsonl",
                  "mathematics_000044.jsonl","mathematics_000045.jsonl",
                  "mathematics_000046.jsonl","mathematics_000047.jsonl",
                  "mathematics_000048.jsonl","mathematics_000049.jsonl",
                  "mathematics_000050.jsonl","mathematics_000051.jsonl",
                  "mathematics_000052.jsonl","mathematics_000053.jsonl",
                  "mathematics_000054.jsonl","mathematics_000055.jsonl",
                  "mathematics_000056.jsonl","mathematics_000057.jsonl",
                  "mathematics_000058.jsonl","mathematics_000059.jsonl",
                  "mathematics_000060.jsonl","mathematics_000061.jsonl",
                  "mathematics_000062.jsonl","mathematics_000063.jsonl",
                  "mathematics_000064.jsonl","mathematics_000065.jsonl",
                  "mathematics_000066.jsonl","mathematics_000067.jsonl",
                  "mathematics_000068.jsonl","mathematics_000069.jsonl",
                  "mathematics_000070.jsonl","mathematics_000071.jsonl",
                  "mathematics_000072.jsonl","mathematics_000073.jsonl",
                  "mathematics_000074.jsonl","mathematics_000075.jsonl",
                  "mathematics_000076.jsonl","mathematics_000077.jsonl",
                  "mathematics_000078.jsonl","mathematics_000079.jsonl",
                  "mathematics_000080.jsonl","mathematics_000081.jsonl",
                  "mathematics_000082.jsonl","mathematics_000083.jsonl",
                  "mathematics_000084.jsonl","mathematics_000085.jsonl",
                  "mathematics_000086.jsonl","mathematics_000087.jsonl",
                  "mathematics_000088.jsonl","mathematics_000089.jsonl",
                  "mathematics_000090.jsonl","mathematics_000091.jsonl",
                  "mathematics_000092.jsonl","mathematics_000093.jsonl",
                  "mathematics_000094.jsonl","mathematics_000095.jsonl",
                  "mathematics_000096.jsonl","mathematics_000097.jsonl",
                  "mathematics_000098.jsonl","mathematics_000099.jsonl",
                  "mathematics_000100.jsonl","mathematics_000101.jsonl",
                  "mathematics_000102.jsonl","mathematics_000103.jsonl",
                  "mathematics_000104.jsonl","mathematics_000105.jsonl",
                  "mathematics_000106.jsonl","mathematics_000107.jsonl",
                  "mathematics_000108.jsonl","mathematics_000109.jsonl"]
    current_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", extra_special_tokens={"eos":"<!~start_sentence>","bos":"<!~end_sentence/>"})
    parseJsonl(base, file_names, current_tokenizer)

    # readParquet(base, "parquets/000011.parquet", current_tokenizer)
   