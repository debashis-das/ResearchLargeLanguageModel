
import gc

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora.LoRAFineTuning import LoRAFineTuning

model_path = "/home/model"
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
        batch_size = 8
        optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-4)
        for i in range(2):
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-sl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            batch_stack_input = []
            batch_stack_attention_mask = []
            training_timestep = 0
            loss = None
            for _, row in df_input.iterrows():
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
                attention_mask = torch.tensor(row['attention_mask'], dtype=dtype, device=device)
                if len(batch_stack_input) < batch_size:
                    batch_stack_input.append(input_ids)
                    batch_stack_attention_mask.append(attention_mask)
                else:
                    batch_input_ids = torch.stack(batch_stack_input)
                    batch_attention_masks = torch.stack(batch_stack_attention_mask)
                    try:
                        _, loss = model_with_lora(batch_input_ids, attention_mask=batch_attention_masks)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        training_timestep += 1
                        if training_timestep % 100 == 0:
                            print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                    except Exception as e:
                        print(f"An error occurred during model training: {e}")
                        raise
                    finally:
                        if training_timestep % 1000 == 0 and loss is not None:
                            torch.save({
                                        'parquet_idx': i,
                                        'epoch_per_parquet': training_timestep,
                                        'model_state_dict': model_with_lora.state_dict(),
                                        'optimizer_state_dic': optimizer.state_dict(),
                                        'loss': loss
                                        }, f"model/qwen-0.6b-with-loRA-sft-model-params")
                            print(f"Model training complete saved with name : qwen-0.6b-with-loRA-sft-model-params")
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_stack_input = []
                        batch_stack_attention_mask = []
    except Exception as e:
        print(f"An error occurred during training the model: {e}")

if __name__ == "__main__":
    sft_train()