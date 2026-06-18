
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
optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-4)

def rl_train(load_path = ""):
    try:
        if len(load_path) > 0:
            checkpoint = torch.load(load_path, map_location=device)
            model_with_lora.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
            print(f"Model loaded successfully from {load_path} with loss: {checkpoint['loss']}")

        grpo_reward_model = GRPORewardModel(tokenizer, model_with_lora, grpo_batch=8, device=device, dtype=dtype)
        training_timestep = 0
        for i in range(1,2):
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            for _, row in df_shuffled.iterrows():
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
                attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device)
                try:
                    loss = grpo_reward_model(input_ids, attention_mask=attention_mask, training_timestep=training_timestep)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    training_timestep += 1
                    if training_timestep % 1 == 0:
                        print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    traceback.print_exc()
                finally:
                    if training_timestep % 1000 == 0 and loss is not None:
                        torch.save({
                                    'parquet_idx': i,
                                    'epoch_per_parquet': training_timestep,
                                    'model_state_dict': grpo_reward_model.state_dict(),
                                    'optimizer_state_dic': optimizer.state_dict(),
                                    'loss': loss
                                    }, f"model/qwen-0.6b-with-loRA-rl-model-params")
                        print(f"Model training complete saved with name : qwen-0.6b-with-loRA-rl-model-params")
                    del input_ids
                    del attention_mask
                    gc.collect()
                    torch.cuda.empty_cache()
                        
    except Exception as e:
        print(f"An error occurred during Parquet generation test: {e}")

if __name__ == "__main__":
    # rl_train(load_path = f"model/qwen-0.6b-with-loRA-sft-model-params")
    rl_train()
