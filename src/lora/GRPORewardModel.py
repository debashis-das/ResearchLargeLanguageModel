import gc
import re
import traceback

from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess.chess_validator import ChessGame
from lora.LoRAFineTuning import LoRAFineTuning

class GRPORewardModel(nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, grpo_batch: int, device="cpu", dtype=torch.float16, total_generation_length=400):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.grpo_batch = grpo_batch
        self.device = device
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.reward_moves = 5
        self.value_fn_moves = 10
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.99
        self.epsilon = 1e-6
        self.beta = 1.0
        # base model initalization
        self.model = model
        base_model_path = "/home/model"
        # base_model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
        dtype = torch.float16
        self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    dtype=dtype,
                    device_map="cpu"
                )
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def use_base_model(self, tokens, attention_mask):
        self.base_model.eval()
        with torch.no_grad():
            base_logits = self.base_model(tokens.to("cpu"), attention_mask=attention_mask.to("cpu")).logits
        base_logits = base_logits[:, :-1, :].contiguous()  # Shift logits for next-token prediction
        return base_logits.to(self.device)

    def board_state(self, moves, chess_board: ChessGame, reward = 0.0, ignore_moves_till = 0, play_as="white"):
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        all_valid = True
        move_no = 0
        for m in re.finditer(pattern, moves_clean):
            move_no = int(m.group(1))
            if move_no <= ignore_moves_till:
                continue
            white   = m.group(2)
            black   = m.group(3)  # None if Black didn't play (resignation)

            if play_as == "white" and move_no == ignore_moves_till:
                continue
            if play_as == "black" and move_no == ignore_moves_till:
                ok, _ = chess_board.push_san(black)
                if ok:
                    count_valid_moves += 1
                    if ignore_moves_till > 0:
                        print(f"[Init] count_valid_moves : {count_valid_moves} : move_no : {move_no} : black move : {black}")
                else:
                    print(f"Invalid move for black: {black}")
                    all_valid = False
                    break
                continue
            if white:
                ok, _ = chess_board.push_san(white)
                if ok:
                    count_valid_moves += 1
                    if ignore_moves_till > 0:
                        print(f"[White] count_valid_moves : {count_valid_moves} : move_no : {move_no} : white move : {white}")
                else:
                    print(f"Invalid move for white: {white}")
                    all_valid = False
                    break
            if black:
                ok, _ = chess_board.push_san(black)
                if ok:
                    count_valid_moves += 1
                    if ignore_moves_till > 0:
                        print(f"[Black] count_valid_moves : {count_valid_moves} : move_no : {move_no} : black move : {black}")
                else:
                    print(f"Invalid move for black: {black}")
                    all_valid = False
                    break
        if count_valid_moves == 0:
            reward -= 10.0
        else:
            reward += count_valid_moves * 0.5
        if all_valid:
            reward += 5.0
        else:
            move_no = move_no - 1  # Adjust move number if the last move was invalid
        return chess_board, reward, all_valid, move_no

    # reward for proper format of the output
    def chess_reward_function(self, prompt, generation):
        try:
            print(f"{prompt} \n\n\n -------\n {generation}")
            game = ChessGame()
            # print(f"Processing output for reward calculation: {prompt} | {generation}")
            # input extraction and create board state based on the input moves
            input_moves =  (prompt.rsplit("Generation Instructions:")[0].strip().rsplit("moves:")[-1].strip())
            play_as = (prompt.split("play_as:")[1].strip().split("moves:")[0].strip())
            # print(f"Input moves extracted for board state initialization: {input_moves}")
            game, _, _, move_no = self.board_state(input_moves, game, play_as=play_as)
            same_generations = 0
            moves_generation = []
            moves_generations_with_extra_text = generation.split("moves:")
            dont_consider = False
            reward = 0.0
            reward_list = []
            for m in range(0, len(moves_generations_with_extra_text)):
                if m%2 == 0:
                    if play_as == "white" and "play_as: black" in moves_generations_with_extra_text[m]:
                        reward -= 10.0
                        dont_consider = True
                    elif play_as == "black" and "play_as: white" in moves_generations_with_extra_text[m]:
                        reward -= 10.0
                        dont_consider = True
                    else:
                        dont_consider = False
                elif m%2 == 1 and not dont_consider:
                    moves = moves_generations_with_extra_text[m].strip()
                    current_reward = 0.0
                    same_generations += 1
                    if same_generations > 1:
                        current_reward -= 10.0
                        print(f"Multiple generations detected. Penalizing reward. Current reward: {current_reward}")
                    moves = moves.strip()
                    generation_move_no = move_no
                    _, current_reward, all_valid, generation_move_no = self.board_state(moves, game, current_reward, ignore_moves_till = move_no, play_as=play_as)
                    if all_valid:
                        result = moves.rsplit(" ")[-1]
                        if play_as == "white" and "1-0" in moves:
                            current_reward += 10.0
                        elif play_as == "black" and "0-1" in moves:
                            current_reward += 10.0
                        elif play_as in ["white", "black"] and "1/2-1/2" in moves:                            
                            current_reward += 5.0
                    reward_list.append((current_reward, generation_move_no))    
        except Exception as e:
            print(f"An error occurred during move processing: {e}")
            traceback.print_exc()
            reward -= 1.0
        max_move_with_reward = reward_list[0]
        for gen in reward_list:
            if gen[1] > max_move_with_reward[1]:
                max_move_with_reward = gen
        print(f"Generated new moves ({reward_list}) : {max_move_with_reward[1] - move_no} : Reward : {max_move_with_reward[0]}")
        return max_move_with_reward[0]

    def extract_reward(self, tensor_per_generation: torch.Tensor, input_sequence_length: int):
        prompt = self.tokenizer.decode(tensor_per_generation[:input_sequence_length], skip_special_tokens=True)
        generation = self.tokenizer.decode(tensor_per_generation[input_sequence_length:], skip_special_tokens=True)
        reward = self.chess_reward_function(prompt, generation)
        return reward

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.unsqueeze(0)
        x = x.unsqueeze(0)
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        output_tensor = self.model.generate(X, max_new_tokens=self.total_generation_length)
        reward_batch = []
        probs_ratio_batch = []
        for tensor_per_generation in output_tensor:
            considered_tensor = tensor_per_generation
            reward = self.extract_reward(considered_tensor, input_sequence_length=x.shape[-1])
            mask_addition = considered_tensor.shape[-1] - attention_mask.shape[-1]
            extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
            training_mask = torch.cat([torch.zeros_like(attention_mask), extra_mask], dim=-1)
            logits, _ = self.model(considered_tensor.unsqueeze(0), attention_mask=training_mask)
            base_logits = self.use_base_model(considered_tensor.unsqueeze(0), attention_mask=training_mask)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            base_probs = torch.nn.functional.softmax(base_logits, dim=-1)
            probs_ratio = probs / (base_probs + 1e-8)
            divergence = torch.nn.functional.log_softmax(logits, dim=-1) - torch.nn.functional.log_softmax(base_logits, dim=-1)
            # reward to be calculated per token
            reward_batch.append(torch.tensor(reward, dtype=self.dtype, device=self.device))
            probs_ratio_batch.append(probs_ratio)
        print(f"Reward batch : {reward_batch}")
        print(f"Probs ratio batch : {probs_ratio_batch}")
        reward_batch = torch.stack(reward_batch)
        probs_ratio_batch = torch.stack(probs_ratio_batch)
        advantage = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
        loss = torch.min(probs_ratio_batch*advantage, torch.clamp(probs_ratio_batch, 1.0 - self.epsilon, 1.0 + self.epsilon)*advantage) - self.beta * divergence
        return loss.mean()

    def debug_logs(self, X, idx, tensor_per_generation):
        print(f"Input      : {self.tokenizer.decode(X[idx], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
        print(f"Generation : {self.tokenizer.decode(tensor_per_generation[X.shape[-1]:], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
