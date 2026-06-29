
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess.chess_validator import ChessGame
from lora.GRPORewardModel import GRPORewardModel
from lora.LoRAFineTuning import LoRAFineTuning

def rl_train(load = False, tokenizer=None, model=None, model_with_lora=None, device=None, dtype=None, optimizer=None, replay_buffer=None):
    try:
        grpo_reward_model = GRPORewardModel(tokenizer, model_with_lora, grpo_batch=28, model_device=device, dtype=dtype)
        training_timestep = 0
        if load:
            # Optimizer state recovery
            optimizer_with_timestep_state = torch.load(f"model/optimizer_with_timestep_state_dict.pt", map_location=device)
            optimizer.load_state_dict(optimizer_with_timestep_state['optimizer_state_dic'])
            # Model recovery with LoRA parameters
            model_with_lora = LoRAFineTuning(model, tokenizer, device=model.device)
            model_with_lora.load_lora_parameters("model/lora_paramters.pt")
            training_timestep = optimizer_with_timestep_state['epoch_per_parquet']
            grpo_reward_model.set_model_after_recovery(model_with_lora)
            print(f"Model loaded successfully")
        recover = False
        for i in range(5):
            if recover:
                # Optimizer state recovery
                optimizer_with_timestep_state = torch.load(f"model/optimizer_with_timestep_state_dict.pt", map_location=device)
                optimizer.load_state_dict(optimizer_with_timestep_state['optimizer_state_dic'])
                # Model recovery with LoRA parameters
                model_with_lora = LoRAFineTuning(model, tokenizer, device=model.device)
                model_with_lora.load_lora_parameters("model/lora_parameters.pt")
                training_timestep = optimizer_with_timestep_state['epoch_per_parquet']
                grpo_reward_model.set_model_after_recovery(model_with_lora)
                print(f"Model recovered successfully after error at timestep: {training_timestep}")
                recover = False
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            for _, row in df_shuffled.iterrows():
                if len(replay_buffer) > 0 and training_timestep % 50 == 0:
                    row_replay = replay_buffer.sample(n=1).iloc[0]
                    input_ids = torch.tensor(row_replay['input_ids'], dtype=torch.long, device=device)
                    attention_mask = torch.tensor(row_replay['attention_mask'], dtype=dtype, device=device)
                else:
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
                    loss, weights = grpo_reward_model(input_ids, attention_mask=attention_mask)
                    if (weights > 0).any():
                        replay_buffer.loc[len(replay_buffer)] = [input_ids.cpu().numpy(), attention_mask.cpu().numpy()]
                        print("Added to replay buffer : Positive reward found in the batch")
                    del weights
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    training_timestep += 1
                    # if training_timestep % 1 == 0:
                    print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    traceback.print_exc()
                    grpo_reward_model.model.save_lora_parameters("model/lora_parameters.pt")
                    torch.save({'optimizer_state_dic': optimizer.state_dict(), 'epoch_per_parquet': training_timestep}, f"model/optimizer_with_timestep_state_dict.pt")
                    print(f"Model training complete saved")
                    recover = True
                finally:
                    for i in range(torch.cuda.device_count()):
                        print(f"[GPU {i}] Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB, Max Allocated: {torch.cuda.max_memory_allocated(i)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB, Max Reserved: {torch.cuda.max_memory_reserved(i)/1024**2:.2f} MB")
                    if training_timestep % 500 == 0 and loss is not None:
                        grpo_reward_model.model.save_lora_parameters("model/lora_parameters.pt")
                        torch.save({'optimizer_state_dic': optimizer.state_dict(), 'epoch_per_parquet': training_timestep}, 
                                   f"model/optimizer_with_timestep_state_dict.pt")
                        if len(replay_buffer) > 0:
                            replay_buffer.to_parquet(f"model/replay_buffer.parquet", compression="zstd", engine="pyarrow")
                            print(f"Replay buffer saved with name : replay_buffer_{training_timestep}.parquet")
                        print(f"Model training complete saved")

                    gc.collect()
                    torch.cuda.empty_cache()
                    del input_ids
                    del attention_mask
    except Exception as e:
        print(f"An error occurred : {e}")
        traceback.print_exc()

def rl_execute():
    model_path = "/home/model"
    # model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
    load = True
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
    replay_buffer = pd.DataFrame(columns=['input_ids', 'attention_mask'])
    rl_train(tokenizer=tokenizer, model=model, model_with_lora=model_with_lora, 
             device=device, dtype=dtype, optimizer=optimizer, 
             replay_buffer=replay_buffer, load=load)
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
