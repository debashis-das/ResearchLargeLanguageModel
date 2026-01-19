import torch
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset

reasoning_start = "<start_working_out>" 
reasoning_end   = "<end_working_out>" 
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"
    
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")

def format_dataset(x):
    expected_answer = x["answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generations"][0]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]

def parseParquetToTensor(total_files, current_tokenizer):
    shard_idx = 0
    batch_idx = 0
    token_idx = 0
    count = 0
    try:
        df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
        for idx in range(total_files):
            # current_file_name = f"dataset/deepseek-r1/train-{idx:05d}-of-00010.parquet"
            token_vals = []
            # sft_df_input = pd.read_parquet(current_file_name)
            sf_ds_input = load_dataset("open-r1/OpenR1-Math-220k")
            for index, record in enumerate(sf_ds_input['train']):
                # print(F"Processing {record['messages']}")
                token_vals.extend(current_tokenizer.apply_chat_template(record['messages'], tokenize = True, add_generation_prompt = True))
                if len(token_vals) > 32000:
                    for shard_idx in range(8):
                        tensor = torch.tensor(token_vals[4000*shard_idx:4000*shard_idx+4000], dtype=torch.int32)
                        print(f"open-r1/OpenR1-Math-220k : tensor : {tensor.shape}, batch : {batch_idx}, shard : {shard_idx}, token : {token_idx}")
                        df[shard_idx].loc[len(df[shard_idx])-1] = [tensor.numpy(), batch_idx, shard_idx, token_idx]
                    batch_idx += 1
                    if batch_idx != 0 and batch_idx % 8 == 0:
                        batch_idx = 0
                    token_idx += 1
                if len(df[0]) == 100000:
                    for shard_idx in range(8):
                        paraquet_filename = f"dataset/deepseek-r1/shards/{shard_idx}/{count:06d}.parquet"
                        df[shard_idx].to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                        print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
                    count += 1
                    token_idx = 0
                    df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
        if len(df[0]) > 10000:        
            for shard_idx in range(8):
                paraquet_filename = f"dataset/deepseek-r1/shards/{shard_idx}/{count:06d}.parquet"
                df[shard_idx].to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
            count += 1
            token_idx = 0
            df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", extra_special_tokens={"eos_token":"</s>"})
    tokenizer.chat_template = chat_template
    total_files=10
    parseParquetToTensor(total_files, tokenizer)
