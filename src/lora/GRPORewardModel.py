import json
import re

from torch import nn
import torch
from transformers import AutoTokenizer

from chess.chess_validator import ChessGame
from lora.LoRAFineTuning import LoRAFineTuning

class GRPORewardModel(nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, grpo_batch: int, device="cpu", dtype=torch.float16, total_generation_length=100, generation_evalution_length=50):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.grpo_batch = grpo_batch
        self.model = model
        self.device = device
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.generation_evalution_length = generation_evalution_length
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.99

    
    def board_state(self, moves, chess_board: ChessGame, reward = 0.0):
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        all_valid = True
        move_no = 0
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
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
        return chess_board, reward, all_valid, move_no

    def extract_jsons(self, text: str) -> list[str]:
        """
        Extract all JSON objects and arrays from a string, including nested ones.
        Returns a list of valid JSON strings found in the input.
        """
        results = []
        i = 0
        
        while i < len(text):
            if text[i] in ('{', '['):
                # Try to parse a JSON starting at position i
                for end in range(len(text), i, -1):
                    candidate = text[i:end]
                    try:
                        json.loads(candidate)
                        results.append(candidate)
                        i = end  # Skip past this JSON
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    i += 1
            else:
                i += 1
        return results

    # reward for proper format of the output
    def chess_reward_function(self, output_with_prompt, moves_to_consider=10):
        game = ChessGame()
        # input extraction and create board state based on the input moves
        extract_input_json_string = (output_with_prompt.split("Input JSON:")[-1].strip()).split("Output should be generated")[0].strip()
        extracted_input_json = json.loads(extract_input_json_string)
        input_moves = extracted_input_json.get("moves")
        play_as = extracted_input_json.get("play_as")
        game, _, _, move_no = self.board_state(input_moves, game)
        extract_json_string = output_with_prompt.split("``` Output JSON ```")[-1].strip()
        reward = 0.0
        jsons = self.extract_jsons(extract_json_string+'"}')
        if len(jsons) == 0 or len(jsons) > 1:
            reward -= 5.0
        for extract in jsons:
            try:
                extracted_json = json.loads(extract)
                # if extracted_json:
                #     reward += 0.5
                # if extracted_json.get("moves", None):
                #     reward += 0.5
                if extracted_json is not None:
                    moves = extracted_json.get("moves").strip()
                    if moves[:len(input_moves)] == input_moves:
                        processed_idx = len(input_moves)
                        moves_to_make = moves[processed_idx:]
                        try:
                            _, reward, all_valid, move_no = self.board_state(moves_to_make, game, reward)
                            # print(f"Reward after processing moves: {reward}, all_valid: {all_valid}, move_no: {move_no}")
                            if all_valid:
                                result = moves.rsplit(" ")[-1]
                                if play_as == "white" and result == "1-0":
                                    reward += 10.0
                                elif play_as == "black" and result == "0-1":
                                    reward += 10.0
                                elif play_as in ["white", "black"] and result == "1/2-1/2":                            
                                    reward += 5.0
                            if move_no >= moves_to_consider:
                                break
                        except Exception as e:
                            print(f"An error occurred during move processing: {e}")
                            reward -= 1.0
                    else:
                        reward -= 2.0
                        break
            except json.JSONDecodeError:
                reward -= 1.0
        return reward


    def extract_reward(self, tensor_per_generation: torch.Tensor, moves_to_consider=10):
        value_fn_str = self.tokenizer.decode(tensor_per_generation, skip_special_tokens=True)
        reward = self.chess_reward_function(value_fn_str, moves_to_consider=moves_to_consider)
        return reward
    
    # batch_size provided as input is always 1 
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.unsqueeze(0)
        x = x.unsqueeze(0)
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        # print(f"Input shape : {X.shape}")
        output_tensor = self.model.generate(X, max_new_tokens=self.total_generation_length)
        # print(f"Output tensor shape: {output_tensor.shape}")
        reward_consideration_reverse_idx = (self.total_generation_length - self.generation_evalution_length)*-1
        advantage_batch = []
        nll_batch = [] 
        for tensor_per_generation in output_tensor:
            # considered_tensor = tensor_per_generation[...,:reward_consideration_reverse_idx]
            print(f"Generation : {self.tokenizer.decode(tensor_per_generation, skip_special_tokens=True)}")
            print("-------------------------------------------------------------")
            continue
        exit(0)
        #     reward = self.extract_reward(considered_tensor, moves_to_consider=5)
        #     value_t_with_k_reward = self.extract_reward(tensor_per_generation, moves_to_consider=20)
        #     # print(f"Reward extracted: {reward} : Value function with k reward extracted: {value_t_with_k_reward}")
        #     value_t_reward = 0.0
        #     mask_addition = considered_tensor.shape[-1] - attention_mask.shape[-1]
        #     extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
        #     training_mask = torch.cat([attention_mask, extra_mask], dim=-1)
        #     # print(f"Attention mask shape after concatenation : {attention_mask.shape} : {extra_mask.shape} : {training_mask.shape}")
        #     _, nll = self.model(considered_tensor.unsqueeze(0), attention_mask=training_mask)
        #     advantage = reward +(value_t_with_k_reward - value_t_reward)
        #     nll_batch.append(nll)
        #     advantage_batch.append(torch.tensor(advantage, dtype=self.dtype, device=self.device))
        # nll_batch = torch.stack(nll_batch)
        # advantage_batch = torch.tanh(torch.stack(advantage_batch))
        # print(f"Advantage : {[a.item() for a in advantage_batch]} : Loss : {[loss.item() for loss in nll_batch]}")
        # loss_batch = nll_batch * self.gamma * advantage_batch
        # print(f"Loss batch : {[loss.item() for loss in loss_batch]}")
        # return loss_batch.mean()
        return None


            
            

        




        