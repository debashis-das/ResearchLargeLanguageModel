
import gc

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora.LoRAFineTuning import LoRAFineTuning

# model_path = "/home/model"
model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
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
        for i in range(2):
            # current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-sl.parquet"
            current_paraquet = f"src\\chess\\paraquets\\{i:06d}-sl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            loss = None
            for _, row in df_input.iterrows():
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device).unsqueeze(0)
                if training_timestep % 50 == 0:
                    generated_ids = model_with_lora.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100)
                    print(f"Generated text: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")  
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
                        if running_loss/accumulation_steps < 0.5:
                            print(f"Loss is very low, stopping training at timestep: {training_timestep}, Loss: {running_loss/accumulation_steps}")
                            return
                        running_loss = torch.zeros([1], dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                    raise
                finally:
                    if training_timestep % 100 == 0 and loss is not None:
                        
                        torch.save({
                                    'parquet_idx': i,
                                    'epoch_per_parquet': training_timestep,
                                    'model_state_dict': model_with_lora.state_dict(),
                                    'optimizer_state_dic': optimizer.state_dict(),
                                    'loss': loss
                                    }, f"model/qwen-0.6b-with-loRA-sft-model-params")
                        print(f"Model training complete saved with name : qwen-0.6b-with-loRA-sft-model-params")
                    del input_ids
                    del attention_mask
                    gc.collect()
                    torch.cuda.empty_cache()
    except Exception as e:
        print(f"An error occurred during training the model: {e}")

if __name__ == "__main__":
    sft_train()