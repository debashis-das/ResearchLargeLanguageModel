import torch
from transformers import AutoTokenizer
import pandas as pd
import random

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
        eos_token = current_tokenizer.encode(current_tokenizer.eos_token)
        df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx']) for _ in range(8)]
        for idx in range(total_files):
            current_file_name = f"dataset/deepseek-r1/train-{idx:05d}-of-00010.parquet"
            filler_file_name = f"dataset/mathematics/parquets/0/000000.parquet"
            token_vals = []
            sft_df_input = pd.read_parquet(current_file_name)
            filler_df_input = pd.read_parquet(filler_file_name)
            filler_df_len = len(filler_df_input)
            for _, record in sft_df_input.iterrows():
                message = current_tokenizer.apply_chat_template(record['messages'], tokenize = True, add_generation_prompt = True)
                message = eos_token + message + eos_token
                message_len = len(message)
                message_count = random.randint(2,4)
                filler_count = message_count+1
                total_message_size = message_count*message_len
                # print(f"Message length : {message_len}, Count : {message_count}, Total message size : {total_message_size}")
                filler_space = 32000 - total_message_size
                if filler_space < 0:
                    continue
                # print(f"filler space : {filler_space}")
                residue = 0
                balanced_space = filler_space // filler_count
                previous_index = 0
                for i in range(filler_count):
                    # print(f"Balanced space : {i*balanced_space} - {(i+1)*balanced_space}")
                    current_index = random.randint(i*balanced_space,(i+1)*balanced_space)
                    residue = (i+1)*balanced_space - current_index
                    # print(f"Current Index : {current_index}, residue : {residue}")
                    filler_len = current_index+residue-previous_index
                    while filler_len>0:
                        filler_record = filler_df_input.iloc[random.randint(0,filler_df_len-1)]
                        filler_sidx = random.randint(1,1024)
                        current_filler_record = filler_record['tensor'][filler_sidx:filler_sidx+filler_len]
                        filler_len -= len(current_filler_record)
                        token_vals.extend(current_filler_record)
                    previous_index = current_index
                    if i < message_count:
                        token_vals.extend(message)
                # print(f"Text created : {len(token_vals)} : {current_tokenizer.decode(token_vals)}")  
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
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", 
                      extra_special_tokens={"bos_token":"<s>", 
                      "eos_token":"</s>", "pad_token":"</s>"})
    tokenizer.chat_template = chat_template
    total_files=10
    parseParquetToTensor(total_files, tokenizer)