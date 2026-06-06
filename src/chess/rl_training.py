
import gc
import json
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess.chess_validator import ChessGame
from lora.GRPORewardModel import GRPORewardModel
from lora.LoRAFineTuning import LoRAFineTuning

model_path = "/home/model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO loading of state dict of the model with LoRA parameters
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )
model_with_lora = LoRAFineTuning(model, tokenizer, device=model.device)
optimizer = torch.optim.AdamW(model_with_lora.parameters(), lr=1e-4)

def board_state(moves, chess_board: ChessGame, reward = 0.0):
    moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
    # Match: move_number. white_move [black_move]
    pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
    count_valid_moves = 0
    all_valid = True
    for m in re.finditer(pattern, moves_clean):
        # move_no = int(m.group(1))
        white   = m.group(2)
        black   = m.group(3)  # None if Black didn't play (resignation)
        if white:
            ok, _ = chess_board.push_san(white)
            if ok:
                count_valid_moves += 1
            else:
                all_valid = False
                break
        if black:
            ok, _ = chess_board.push_san(black)
            if ok:
                count_valid_moves += 1
            else:
                all_valid = False
                break
    if count_valid_moves == 0:
        reward -= 10.0
    else:
        reward += count_valid_moves * 0.5
    if all_valid:
        reward += 5.0
    return chess_board, reward, all_valid

# reward for proper format of the output
def chess_reward_function(output_with_prompt):
    game = ChessGame() 
    # input extraction and create board state based on the input moves
    extract_input_json_string = (output_with_prompt.split("Input JSON:")[-1].strip()).split("Output should be generated")[0].strip()
    extracted_input_json = json.loads(extract_input_json_string)
    input_moves = extracted_input_json.get("moves")
    play_as = extracted_input_json.get("play_as")
    game, _, _ = board_state(input_moves, game)
    extract_json_string = (output_with_prompt.split("``` Output JSON ```")[-1].strip()).split("``` Output JSON END```")[0].strip()
    # check if extracted string is a json
    reward = 0.0
    try:
        extracted_json = json.loads(extract_json_string)
        if extracted_json:
            reward += 0.5
        if extracted_json.get("moves", None):
            reward += 0.5
        if extracted_json is not None:
            moves = extracted_json.get("moves").strip()
            if moves[:len(input_moves)] == input_moves:
                reward += 1.0
                processed_idx = len(input_moves)
                moves_to_make = moves[processed_idx:]
                try:
                    _, reward, all_valid = board_state(moves_to_make, game, reward)
                    if all_valid:
                        result = moves.rsplit(" ")[-1]
                        if play_as == "white" and result == "1-0":
                            reward += 10.0
                        elif play_as == "black" and result == "0-1":
                            reward += 10.0
                        elif play_as in ["white", "black"] and result == "1/2-1/2":                            
                            reward += 5.0
                except Exception as e:
                    print(f"An error occurred during move processing: {e}")
                    reward -= 1.0
    except json.JSONDecodeError:
        reward -= 1.0
    return reward

def rl_train():
    try:
        model_with_grpo_reward = GRPORewardModel(tokenizer, model_with_lora, grpo_batch=8, reward=[chess_reward_function])
        for i in range(2):
            current_paraquet = f"/home/ResearchLargeLanguageModel/src/chess/paraquets/{i:06d}-rl.parquet"
            df_input = pd.read_parquet(current_paraquet)
            training_timestep = 0
            for _, row in df_input.iterrows():
                input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
                attention_mask = torch.tensor(row['attention_mask'], dtype=torch.bfloat16, device=device)
                try:
                    _, loss = model_with_grpo_reward(input_ids=input_ids, attention_mask=attention_mask)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    training_timestep += 1
                    if training_timestep % 100 == 0:
                        print(f"Training timestep: {training_timestep}, Loss: {loss.item()}")
                except Exception as e:
                    print(f"An error occurred during model training: {e}")
                finally:
                    if training_timestep % 1000 == 0:
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
                        
    except Exception as e:
        print(f"An error occurred during Parquet generation test: {e}")

if __name__ == "__main__":
    rl_train()