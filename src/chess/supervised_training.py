
import gc
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
        accumulation_steps = 4
        training_timestep = 0
        running_loss = torch.zeros([1], dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-5)
        end = False
        for i in range(2):
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-sl.parquet"
            # current_paraquet = f"src\\chess\\paraquets\\{i:06d}-sl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            df_shuffled = df_input.sample(frac=1, ignore_index=True)
            loss = None
            for _, row in df_shuffled.iterrows():
                if training_timestep > 25:
                    end = True
                    break
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                if training_timestep % 10 == 0:
                    current_rl_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
                    df_rl_input = pd.read_parquet(current_rl_paraquet)
                    row_rl = df_rl_input.sample(n=1).iloc[0]
                    input_ids_rl = torch.tensor(row_rl['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                    attention_mask_rl = torch.tensor(row_rl['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                    generation_ids = model_with_lora.generate(input_ids_rl, max_new_tokens=200, temperature=0.7)
                    print(f"Generated text: {tokenizer.decode(generation_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text
                try:
                    _, loss = model_with_lora(input_ids, attention_mask=attention_mask)
                    loss = loss / accumulation_steps
                    loss.backward()
                    running_loss += loss.item()*accumulation_steps
                    training_timestep += 1
                    if training_timestep % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        print(f"Training timestep: {training_timestep}, Loss: {running_loss/accumulation_steps}")
                        if running_loss/accumulation_steps < 0.78:
                            print(f"Loss is very low, stopping training at timestep: {training_timestep}, Loss: {running_loss/accumulation_steps}")
                            return
                        running_loss = torch.zeros([1], dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    raise
                finally:
                    if (training_timestep % 25 == 0 and loss is not None) or end:
                        model_with_lora.save_lora_parameters("model/sft_lora_parameters.pt")
                        torch.save({'optimizer_state_dic': optimizer.state_dict(), 'epoch_per_parquet': training_timestep}, 
                                   f"model/sft_optimizer_with_timestep_state_dict.pt")
                        print(f"Model training complete saved")
                    del input_ids
                    del attention_mask
                    gc.collect()
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
        output_ids = model_with_lora.generate(input_ids_rl, max_new_tokens=200, temperature=0.7)
        print(f"Generated text: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")  # Debugging line to check generated text

if __name__ == "__main__":
    sft_execute()
