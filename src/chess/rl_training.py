
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import re
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess.chess_validator import ChessGame
from lora.GRPORewardModel import GRPORewardModel
from lora.LoRAFineTuning import LoRAFineTuning

model_path = "/home/model"
# model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"

tokenizer = AutoTokenizer.from_pretrained(model_path)
dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto"
        )
model_with_lora = LoRAFineTuning(model, tokenizer, device=model.device)
device = model.device
optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-5)

def rl_train(load_path = ""):
    try:
        if len(load_path) > 0:
            checkpoint = torch.load(load_path, map_location=device)
            model_with_lora.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
            print(f"Model loaded successfully from {load_path} with loss: {checkpoint['loss']}")

        grpo_reward_model = GRPORewardModel(tokenizer, model_with_lora, grpo_batch=28, model_device=device, dtype=dtype)
        training_timestep = 0
        recover = False
        for i in range(5):
            if recover:
                checkpoint = torch.load(f"model/qwen-0.6b-with-loRA-rl-model-params", map_location=device)
                model_with_lora.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
                training_timestep = checkpoint['epoch_per_parquet']
                grpo_reward_model.set_model_after_recovery(model_with_lora)
                print(f"Model recovered successfully from qwen-0.6b-with-loRA-rl-model-params with loss: {checkpoint['loss']}")
                recover = False
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            for _, row in df_shuffled.iterrows():
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
                attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device)
                if training_timestep % 50 == 0:
                    current_rl_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/000003-rl.parquet"
                    df_rl_input = pd.read_parquet(current_rl_paraquet)
                    row_rl = df_rl_input.sample(n=1).iloc[0]
                    input_ids_rl = torch.tensor(row_rl['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                    attention_mask_rl = torch.tensor(row_rl['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                    generation_ids = grpo_reward_model.generate(input_ids_rl, attention_mask=attention_mask_rl, max_new_tokens=50)
                    print(f"Generated text: {tokenizer.decode(generation_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text
                try:
                    loss = grpo_reward_model(input_ids, attention_mask=attention_mask)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    training_timestep += 1
                    # if training_timestep % 1 == 0:
                    print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    traceback.print_exc()
                    torch.save({
                                'parquet_idx': i,
                                'epoch_per_parquet': training_timestep,
                                'model_state_dict': grpo_reward_model.state_dict(),
                                'optimizer_state_dic': optimizer.state_dict(),
                                'loss': loss
                                }, f"model/qwen-0.6b-with-loRA-rl-model-params")
                    print(f"Model training complete saved with name : qwen-0.6b-with-loRA-rl-model-params")
                    recover = True
                finally:
                    for i in range(torch.cuda.device_count()):
                        print(f"[GPU {i}] Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB, Max Allocated: {torch.cuda.max_memory_allocated(i)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB, Max Reserved: {torch.cuda.max_memory_reserved(i)/1024**2:.2f} MB")
                    if training_timestep % 500 == 0 and loss is not None:
                        torch.save({
                                    'parquet_idx': i,
                                    'epoch_per_parquet': training_timestep,
                                    'model_state_dict': grpo_reward_model.state_dict(),
                                    'optimizer_state_dic': optimizer.state_dict(),
                                    'loss': loss
                                    }, f"model/qwen-0.6b-with-loRA-rl-model-params")
                        print(f"Model training complete saved with name : qwen-0.6b-with-loRA-rl-model-params")
                    gc.collect()
                    torch.cuda.empty_cache()
                    del input_ids
                    del attention_mask
    except Exception as e:
        print(f"An error occurred during Parquet generation test: {e}")

if __name__ == "__main__":
    rl_train(load_path = f"model/qwen-0.6b-with-loRA-sft-model-params")
    # rl_train()
    # current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
    # i = 0
    # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
    # df_input = pd.read_parquet(current_paraquet)
    # df_shuffled = df_input.sample(frac=1, ignore_index=True)
    # for _, row in df_shuffled.iterrows():
    #     input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
    #     attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device)
    #     print(tokenizer.decode(input_ids, skip_special_tokens=True))
