
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora.LoRAFineTuning import LoRAFineTuning

model_path = "/home/model"
# model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
tokenizer = AutoTokenizer.from_pretrained(model_path)
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto"
        )
model_with_lora = LoRAFineTuning(model, tokenizer, dtype=dtype, device=model.device)
device = model.device

def sft_train():
    try:
        training_timestep = 0
        optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-5)
        end = False
        for i in range(4):
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-sl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-sl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            loss = None
            batch = 14
            current_batch = 0
            prompt_length = []
            for _, row in df_shuffled.iterrows():
                try:
                    if training_timestep > 1000:
                        end = True
                        break
                    if current_batch == 0:
                        input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                        attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                        prompt_length.append(row['prompt_length'])
                        current_batch += 1
                        continue
                    elif current_batch < batch:
                        input_ids = torch.cat([input_ids, torch.tensor(row['input_ids'], dtype=torch.long, device=device).unsqueeze(0)], dim=0)
                        attention_mask = torch.cat([attention_mask, torch.tensor(row['attention_mask'], dtype=dtype, device=device).unsqueeze(0)], dim=0)
                        prompt_length.append(row['prompt_length'])
                        current_batch += 1
                        continue
                    else:
                        current_batch = 0
                        if training_timestep % 100 == 0 and training_timestep != 0:
                            current_rl_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
                            df_rl_input = pd.read_parquet(current_rl_paraquet)
                            row_rl = df_rl_input.sample(n=1).iloc[0]
                            input_ids_rl = torch.tensor(row_rl['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                            generation_ids = model_with_lora.generate(input_ids_rl, max_new_tokens=300, temperature=0.7)
                            print(f"Generated text: {tokenizer.decode(generation_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text
                        _, loss = model_with_lora(input_ids, attention_mask=attention_mask, prompt_length=prompt_length)
                        loss.backward()
                        training_timestep += 1
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                        for i in range(torch.cuda.device_count()):
                            print(f"[GPU {i}] Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB, Max Allocated: {torch.cuda.max_memory_allocated(i)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB, Max Reserved: {torch.cuda.max_memory_reserved(i)/1024**2:.2f} MB")
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    raise
                finally:   
                    if (training_timestep % 500 == 0 and loss is not None) or end:
                        model_with_lora.save_lora_parameters("model/sft_lora_parameters.pt")
                        torch.save({'optimizer_state_dic': optimizer.state_dict(), 'epoch_per_parquet': training_timestep}, 
                                f"model/sft_optimizer_with_timestep_state_dict.pt")
                        print(f"Model training complete saved")
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred during training the model: {e}")
        traceback.print_exc()

def sft_execute():
    try:
        sft_train()
    finally:
        model_with_lora.load_lora_parameters("model/sft_lora_parameters.pt")
        i=3
        current_rl_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
        df_rl_input = pd.read_parquet(current_rl_paraquet)
        row_rl = df_rl_input.sample(n=1).iloc[0]
        input_ids_rl = torch.tensor(row_rl['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask_rl = torch.tensor(row_rl['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
        output_ids = model_with_lora.generate(input_ids_rl, max_new_tokens=300, temperature=0.7)
        print(f"Generated text: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text

if __name__ == "__main__":
    sft_execute()
