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
    # expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    # thoughts = x["generated_solution"]
    # thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    # thoughts = thoughts.strip()
    # Add our custom formatting
    # final_prompt = \
    #     reasoning_start + thoughts + reasoning_end + \
    #     solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        # {"role" : "assistant", "content" : final_prompt},
    ]

def parseParquetToTensor(current_tokenizer):
    shard_idx = 0
    batch_idx = 0
    token_idx = 0
    count = 0
    try:
        eos_token_id = current_tokenizer.eos_token_id
        pad_token_id = current_tokenizer.pad_token_id
        df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx', 'generation_idx','answer_tensor', 'final_prompt']) for _ in range(8)]
        for idx in range(1):
            current_file_name = f"dataset/unsloth/cot-{idx:05d}-of-00001.parquet"
            filler_file_name = f"dataset/mathematics/parquets/0/000000.parquet"
            reward_df_input = pd.read_parquet(current_file_name)
            filler_df_input = pd.read_parquet(filler_file_name)
            filler_df_len = len(filler_df_input)
            for _, record in reward_df_input.iterrows():
                token_vals = []
                message_count = random.randint(2,4)
                message = current_tokenizer.apply_chat_template(format_dataset(record), tokenize = True, add_generation_prompt = True)
                answer = current_tokenizer.encode(record["expected_answer"])

                thoughts = record["generated_solution"]
                thoughts = thoughts.replace("<think>", "").replace("</think>", "")

                final_prompt = \
                    reasoning_start + thoughts + reasoning_end + \
                    solution_start + record["expected_answer"] + solution_end

                final_prompt_tensor = torch.tensor(current_tokenizer.encode(final_prompt), dtype=torch.int32)
                answer_tensor = torch.tensor(answer, dtype=torch.int32)

                message = [eos_token_id] + message + [eos_token_id]
                message_len = len(message*message_count)
                filler_space = 32000 - (message_len+5000)
                if filler_space < 0:
                    continue
                filler_len = random.randint(0,filler_space)
                current_filler_len = filler_len
                filler_record = filler_df_input.iloc[random.randint(0,filler_df_len-1)]
                while current_filler_len > 0:
                    filler_sidx = random.randint(1,1024)
                    current_filler_record = filler_record['tensor'][filler_sidx:filler_sidx+current_filler_len]
                    current_filler_len -= len(current_filler_record)
                    # filler
                    token_vals.extend(current_filler_record)
                # print(f"filler space ({filler_len}) : {len(token_vals)}")
                # pad
                pad_length = 32000 - (len(token_vals)+message_count*message_len)
                # print(f"pad length : {pad_length}")
                if pad_length <= 0:
                    continue
                pad_tensors = torch.full((pad_length//message_count,), pad_token_id)
                # message
                generation_idx = []
                for _ in range(message_count):
                    token_vals.extend(message)
                    generation_idx.append(len(token_vals))
                    token_vals.extend(pad_tensors)
                    # print(f"token({len(message)}) spac with pad({len(pad_tensors)}) : {len(token_vals)}")
                token_vals.extend(pad_tensors)
                # print(f"Token values ({len(token_vals)}) : {generation_idx}")
                if len(token_vals) > 32000:
                    for shard_idx in range(8):
                        tensor = torch.tensor(token_vals[4000*shard_idx:4000*shard_idx+4000], dtype=torch.int32)
                        print(f"{current_file_name} : tensor : {tensor.shape}, batch : {batch_idx}, shard : {shard_idx}, token : {token_idx}, generation_idx : {generation_idx}, answer_tensor : {answer_tensor.shape}, final_prompt : {final_prompt_tensor.shape} ")
                        df[shard_idx].loc[len(df[shard_idx])-1] = [tensor.numpy(), batch_idx, shard_idx, token_idx, generation_idx, answer_tensor.numpy(), final_prompt_tensor.numpy()]
                    batch_idx += 1
                    if batch_idx != 0 and batch_idx % 8 == 0:
                        batch_idx = 0
                    token_idx += 1
                if len(df[0]) == 100000:
                    for shard_idx in range(8):
                        paraquet_filename = f"dataset/unsloth/shards/{shard_idx}/{count:06d}.parquet"
                        df[shard_idx].to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                        print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
                    count += 1
                    token_idx = 0
                    df = [pd.DataFrame(columns=['tensor', 'batch', 'shard', 'token_idx', 'generation_idx','answer_tensor', 'final_prompt']) for _ in range(8)]
        if len(df[0]) > 10000:        
            for shard_idx in range(8):
                paraquet_filename = f"dataset/unsloth/shards/{shard_idx}/{count:06d}.parquet"
                df[shard_idx].to_parquet(paraquet_filename, compression="zstd", engine="pyarrow")
                print(f"Successfully created a new Parquet file: '{paraquet_filename}'")
            count += 1
            token_idx = 0
    except Exception as e:
        print(f"An error occurred: {e}")

def messageOpenr1(current_tokenizer):
    current_file_name = f"dataset/unsloth/cot-{0:05d}-of-00001.parquet"
    sft_df_input = pd.read_parquet(current_file_name)
    message = current_tokenizer.apply_chat_template(format_dataset(sft_df_input.iloc[0]), tokenize = False, add_generation_prompt = True)
    print(message)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", 
                      extra_special_tokens={"bos_token":"<s>", 
                      "eos_token":"</s>", "pad_token":"</s>"})
    tokenizer.chat_template = chat_template
    # messageOpenr1(tokenizer)
    parseParquetToTensor(tokenizer)