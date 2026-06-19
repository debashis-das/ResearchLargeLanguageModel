from copy import deepcopy
import gc
import re
import traceback

from torch import nn
import torch
from transformers import AutoTokenizer

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
        self.gamma = 0.99
        self.epsilon = 0.05
        self.beta = 0.02
        self.target_kl = 0.01

        # base model initalization
        self.model = model
        # adding tiny noise to the model parameters to avoid identical outputs from the base model and the fine-tuned model
        for p in self.model.parameters():
            p.data += 0.001 * torch.randn_like(p)    

        self.base_model = deepcopy(model)
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def use_base_model(self, tokens, attention_mask):
        self.base_model.eval()
        with torch.no_grad():
            base_logits, _ = self.base_model(tokens, attention_mask=attention_mask)
        return base_logits

    def board_state(self, moves, chess_board: ChessGame, reward = 0.0, ignore_moves_till = 0, play_as="white"):
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        all_valid = True
        move_no = 0
        # print(f"-----------> PLay as : {play_as}")
        for m in re.finditer(pattern, moves_clean):
            try:
                move_no = int(m.group(1))
                if move_no < ignore_moves_till:
                    continue
                white   = m.group(2)
                black   = m.group(3)  # None if Black didn't play (resignation)

                if play_as == "white" and move_no == ignore_moves_till:
                    continue
                if play_as == "black" and move_no == ignore_moves_till and black is not None:
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
                if white and white is not None:
                    # print(f"Processing move number {move_no} : White move : {white}")
                    ok, _ = chess_board.push_san(white)
                    if ok:
                        count_valid_moves += 1
                        if ignore_moves_till > 0:
                            print(f"[White] base moves : {ignore_moves_till} count_valid_moves : {count_valid_moves} : move_no : {move_no} : white move : {white}")
                    else:
                        # print(f"Invalid move for white: {white}")
                        all_valid = False
                        break
                if black and black is not None:
                    # print(f"Processing move number {move_no} : Black move : {black}")
                    ok, _ = chess_board.push_san(black)
                    if ok:
                        count_valid_moves += 1
                        if ignore_moves_till > 0:
                            print(f"[Black] base moves : {ignore_moves_till} count_valid_moves : {count_valid_moves} : move_no : {move_no} : black move : {black}")
                    else:
                        # print(f"Invalid move for black: {black}")
                        all_valid = False
                        break
                if black is None or white is None:
                    break
            except Exception as e:
                print(f"An error occurred while processing moves: {e}")
                traceback.print_exc()
                all_valid = False
                break
        if count_valid_moves == 0:
            reward -= 10.0
        else:
            reward += count_valid_moves * 0.5
        if all_valid:
            reward += 5.0
        return chess_board, reward, all_valid, move_no

    # reward for proper format of the output
    def chess_reward_function(self, prompt, generation):
        try:
            # print(f"{prompt} \n\n\n-------------------------------\n\n\n")
            game = ChessGame()
            # print(f"Processing output for reward calculation: {prompt} | {generation}")
            # input extraction and create board state based on the input moves
            input_moves =  (prompt.rsplit("Generation Instructions:")[0].strip().rsplit("moves:")[-1].strip())
            play_as = (prompt.split("play_as:")[-1].strip().split("moves:")[0].strip())
            # print(f"Input moves extracted for board state initialization: {input_moves}")
            game, _, _, move_no = self.board_state(input_moves, game, play_as=play_as)
            same_generations = 0
            if "moves:" not in generation:
                print(f"No moves generated. Penalizing reward. Generation: {generation}")
                return -10.0
            moves_generations_with_extra_text = generation.split("moves:")
            # print(f"moves_generations_with_extra_text : {len(moves_generations_with_extra_text)} : {moves_generations_with_extra_text}")
            if len(moves_generations_with_extra_text) == 0:
                current_reward = 10.0
                _, current_reward, all_valid, generation_move_no = self.board_state(generation.strip(), game, current_reward, ignore_moves_till = move_no, play_as=play_as)
                print(f"[Ideal case] Generated new moves : {generation_move_no - move_no} : Reward : {current_reward}")
                return current_reward
            dont_consider = False
            reward = 0.0
            reward_list = []
            for m in range(0, len(moves_generations_with_extra_text)):
                # print(f"Processing generation segment : {moves_generations_with_extra_text[m]}")
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
                    moves = moves_generations_with_extra_text[m]
                    current_reward = 0.0
                    same_generations += 1
                    if same_generations > 1:
                        current_reward -= 10.0
                        print(f"Multiple generations detected. Penalizing reward. Current reward: {current_reward}")
                    moves = moves.strip()
                    generation_move_no = move_no
                    _, current_reward, all_valid, generation_move_no = self.board_state(moves, game, current_reward, ignore_moves_till = move_no, play_as=play_as)
                    if all_valid:
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
        max_move_with_reward = reward_list[0] if len(reward_list) > 0 else (0.0, move_no)
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

    def sanatize_logits(self, logits: torch.Tensor):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-50, max=50)  # Clamp logits to avoid extreme values
        logits = torch.nn.functional.softmax(logits.float(), dim=-1)  
        logits = torch.nan_to_num(logits, nan=0.0)
        logits = logits / logits.sum(dim=-1, keepdim=True)  # Normalize to get probabilities
        return logits
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, training_timestep: int):
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
        x = x.unsqueeze(0)
        input_sequence_length = x.shape[-1]
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        attention_mask = attention_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the attention mask for the batch size
        output_tensor = self.model.generate(X, max_new_tokens=self.total_generation_length, temperature=0.8)
        mask_addition = output_tensor.shape[-1] - attention_mask.shape[-1]
        extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
        extra_mask = extra_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the extra mask for the batch size
        training_mask = torch.cat([torch.zeros_like(attention_mask), extra_mask], dim=-1)
        
        logits, _ = self.model(output_tensor, attention_mask=training_mask)
        base_logits = self.use_base_model(output_tensor, attention_mask=training_mask)

        logits = self.sanatize_logits(logits)
        base_logits = self.sanatize_logits(base_logits)

        actions = output_tensor[..., 1:]
        if torch.isnan(logits).any():
            print("NaN in logits!")
            exit()
        if torch.isnan(base_logits).any():
            print("NaN in base_logits!")
            exit()

        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1).float()
        base_log_probs_all = torch.nn.functional.log_softmax(base_logits, dim=-1).float()        # print(f"log_probs : {torch.isnan(log_probs).any()} : base_log_probs : {torch.isnan(base_log_probs).any()}")

        log_probs = torch.gather(log_probs_all, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        base_log_probs = torch.gather(base_log_probs_all, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        
        del log_probs_all
        del base_log_probs_all

        ratio_clamp = torch.clamp(log_probs - base_log_probs, min=-10, max=10)
        probs_ratio_batch = torch.exp(ratio_clamp)
        divergence = log_probs - base_log_probs

        del attention_mask
        del mask_addition
        del extra_mask
        del training_mask
        del X
        del x
        del log_probs
        del base_log_probs
        del ratio_clamp
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"probs_ratio_batch : {torch.isnan(probs_ratio_batch).any()} : divergence : {torch.isnan(divergence).any()}")
        reward_batch = []
        for tensor_per_generation in output_tensor:
            considered_tensor = tensor_per_generation
            reward = self.extract_reward(considered_tensor, input_sequence_length=input_sequence_length)
            # reward to be calculated per token
            reward_batch.append(torch.tensor(reward, dtype=self.dtype, device=self.device))
        # print(f"Probs ratio batch : {probs_ratio_batch}")
        reward_batch = torch.stack(reward_batch).float()
        advantage = reward_batch
        
        # print(f"Reward batch : {reward_batch} : Advantage : {advantage}")
        advantage = advantage.unsqueeze(-1).unsqueeze(-1)
        
        product = advantage * probs_ratio_batch
        product = torch.nan_to_num(product, 0.0)
        product_with_clipping = torch.clamp(probs_ratio_batch, 1.0 - self.epsilon, 1.0 + self.epsilon)*advantage
        # print(f"product : {product.max()} : product_with_clipping : {product_with_clipping.max()} : divergence : {divergence.max()}")
        # print(f"product : {product.min()} : product_with_clipping : {product_with_clipping.min()} : divergence : {divergence.min()}")
        # print(f"product : {product.mean()} : product_with_clipping : {product_with_clipping.mean()} : divergence : {divergence.mean()}")
        
        loss = -torch.min(product, product_with_clipping) + self.beta * divergence
        print(f"Loss : {loss.mean()} : Advantage : {advantage.mean()} : Product : {product.mean()} : Product with clipping : {product_with_clipping.mean()} : Divergence : {divergence.mean()}")

        del reward_batch
        del advantage
        del product
        del product_with_clipping
        del divergence
        gc.collect()
        torch.cuda.empty_cache()

        if torch.isnan(loss).any():
            print("NaN in loss!")
            exit()
        return loss.mean()

    def debug_logs(self, X, idx, tensor_per_generation):
        print(f"Input      : {self.tokenizer.decode(X[idx], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
        print(f"Generation : {self.tokenizer.decode(tensor_per_generation[X.shape[-1]:], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
