
import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback

import pandas as pd
import torch

from lora.GRPORewardModel import GRPORewardModel

GRPO_BATCH_SIZE = 14
SKIP_THRESHOLD = 3

def rl_train(tokenizer_path=None, model_path=None, loRA_parameters_path=None, dtype=None):
    try:
        grpo_reward_model = GRPORewardModel(tokenizer_path, model_path, grpo_batch=GRPO_BATCH_SIZE, 
                                            loRA_parameters_path=loRA_parameters_path)
        device = grpo_reward_model.model_device
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, grpo_reward_model.parameters()), lr=1e-5)
        training_timestep = 0
        recover = False
        consecutive_skips = 0
        replay_index = 0
        replay_paraquet_path = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{replay_index:06d}-rl-replay-buffer.parquet"
        replay_df_shuffled = pd.read_parquet(replay_paraquet_path).sample(frac=1, ignore_index=True)
        replay_counter = 0
        for i in range(4):
            if recover:
                # Optimizer state recovery
                optimizer_with_timestep_state = torch.load(f"model/optimizer_with_timestep_state_dict.pt", map_location=device)
                optimizer.load_state_dict(optimizer_with_timestep_state['optimizer_state_dic'])
                training_timestep = optimizer_with_timestep_state['epoch_per_parquet']
                # Model recovery with LoRA parameters
                del grpo_reward_model
                grpo_reward_model = GRPORewardModel(tokenizer_path, model_path, grpo_batch=GRPO_BATCH_SIZE, 
                                            loRA_parameters_path=loRA_parameters_path)
                print(f"Model recovered successfully after error at timestep: {training_timestep}")
                recover = False
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            for _, row in df_shuffled.iterrows():
                if training_timestep % 10 == 0:
                    if replay_counter >= len(replay_df_shuffled):
                        replay_counter = 0
                        replay_index = (replay_index + 1) % 4  # Cycle through the replay buffer parquet files
                        replay_paraquet_path = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{replay_index:06d}-rl-replay-buffer.parquet"
                        replay_df_shuffled = pd.read_parquet(replay_paraquet_path).sample(frac=1, ignore_index=True)
                    row_replay = replay_df_shuffled.iloc[replay_counter]
                    replay_counter += 1
                    input_ids = torch.tensor(row_replay['input_ids'], dtype=torch.long, device=device)
                    attention_mask = torch.tensor(row_replay['attention_mask'], dtype=dtype, device=device)
                else:
                    input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
                    attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device)
                if (training_timestep+1) % 50 == 0:
                    current_rl_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/000003-rl.parquet"
                    df_rl_input = pd.read_parquet(current_rl_paraquet)
                    row_rl = df_rl_input.sample(n=1).iloc[0]
                    input_ids_rl = torch.tensor(row_rl['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                    attention_mask_rl = torch.tensor(row_rl['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                    generation_ids = grpo_reward_model.generate(input_ids_rl)
                    print(f"Generated text: {grpo_reward_model.tokenizer.decode(generation_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text
                try:
                    kl_pull = consecutive_skips >= SKIP_THRESHOLD
                    loss, rewards = grpo_reward_model(input_ids, attention_mask=attention_mask, kl_pull=kl_pull)
                    if loss is None:
                        consecutive_skips += 1
                        print(f"Skipping optimizer step: loss is None (all rewards were negative). Consecutive skips: {consecutive_skips}")
                        continue  # Skip this iteration if loss is None (all rewards were negative)
                    is_uniform_batch = torch.tanh(rewards.float()).std() < 1e-4
                    if is_uniform_batch:
                        print(f"KL-pull step fired after {consecutive_skips} consecutive skips")
                        # leave consecutive_skips as-is so kl_pull keeps firing on the next uniform batch
                    else:
                        consecutive_skips = 0
                    print("Rewards : ", rewards)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not torch.isfinite(loss):
                        print(f"Skipping optimizer step: non-finite loss ({loss.item()})")
                        continue
                    non_finite_grads = [
                        (name, p.grad.isnan().sum().item(), p.grad.isinf().sum().item())
                        for name, p in grpo_reward_model.named_parameters()
                        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
                    ]
                    if non_finite_grads:
                        print(f"Non-finite GRADIENTS before clipping (name, nan_count, inf_count): {non_finite_grads}")
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, grpo_reward_model.parameters()), max_norm=1.0
                    )
                    optimizer.step()
                    non_finite_weights = [
                        name for name, p in grpo_reward_model.named_parameters()
                        if p.requires_grad and not torch.isfinite(p).all()
                    ]
                    if non_finite_weights:
                        print(f"Non-finite trainable WEIGHTS after optimizer.step() (grads were clean): {non_finite_weights}")
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
    loRA_parameters_path = "model/sft_lora_parameters.pt"
    
    rl_train(tokenizer_path=model_path, model_path=model_path, loRA_parameters_path=loRA_parameters_path)

if __name__ == "__main__":
    rl_execute()
    # for i in range(4):
    #     current_paraquet = f"src\\chess\\paraquets\\{i:06d}-rl.parquet"
    #     df_input = pd.read_parquet(current_paraquet)
    #     print(f"DataFrame from {current_paraquet} has {len(df_input)} rows.")
