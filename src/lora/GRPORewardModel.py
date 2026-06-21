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

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, grpo_batch: int, model_device="cpu", dtype=torch.float16, total_generation_length=300):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.grpo_batch = grpo_batch
        self.model_device = model_device
        # self.loss_device = loss_device
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.gamma = 0.99
        self.epsilon = 0.05
        self.beta = 0.02

        # base model initalization
        self.model = model
        # adding tiny noise to the model parameters to avoid identical outputs from the base model and the fine-tuned model
        for p in self.model.parameters():
            p.data += 0.0005 * torch.randn_like(p)    
        self.base_model = deepcopy(model)
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def use_base_model(self, tokens, attention_mask):
        self.base_model.eval()
        with torch.no_grad():
            base_logits, _ = self.base_model(tokens, attention_mask=attention_mask, with_no_loss=True)
        return base_logits.detach()

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
            reward = 0.0
            if "moves:" not in generation:
                print(f"No moves generated. Penalizing reward.")
                return -10.0
            reward += 1.0  # Reward for generating output in the expected format
            moves_generations_with_extra_text = generation.split("moves:")
            # print(f"moves_generations_with_extra_text : {len(moves_generations_with_extra_text)} : {moves_generations_with_extra_text}")
            if len(moves_generations_with_extra_text) == 0:
                current_reward = 10.0
                _, current_reward, all_valid, generation_move_no = self.board_state(generation.strip(), game, current_reward, ignore_moves_till = move_no, play_as=play_as)
                print(f"[Ideal case] Generated new moves : {generation_move_no - move_no} : Reward : {current_reward}")
                return current_reward
            dont_consider = False
            reward_list = []
            for m in range(0, len(moves_generations_with_extra_text)):
                # print(f"Processing generation segment : {moves_generations_with_extra_text[m]}")
                if m%2 == 0:
                    if play_as == "white" and "play_as: black" in moves_generations_with_extra_text[m]:
                        reward -= 0.5
                        dont_consider = True
                    elif play_as == "black" and "play_as: white" in moves_generations_with_extra_text[m]:
                        reward -= 0.5
                        dont_consider = True
                    else:
                        dont_consider = False
                elif m%2 == 1 and not dont_consider:
                    moves = moves_generations_with_extra_text[m]
                    current_reward = reward
                    same_generations += 1
                    if same_generations > 1:
                        current_reward -= 0.2
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
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = torch.exp(logits - logsumexp)  # Normalize logits to prevent overflow in softmax
        # logits = torch.nan_to_num(logits, nan=0.0)
        # logits = logits / logits.sum(dim=-1, keepdim=True)  # Normalize to get probabilities
        return logits  # Convert back to half precision
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
        x = x.unsqueeze(0)
        input_sequence_length = x.shape[-1]
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        attention_mask = attention_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the attention mask for the batch size
        with torch.no_grad():
            output_tensor = self.model.generate(X.to(self.model_device), max_new_tokens=self.total_generation_length, temperature=0.01)

        mask_addition = output_tensor.shape[-1] - attention_mask.shape[-1]
        extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
        extra_mask = extra_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the extra mask for the batch size
        training_mask = torch.cat([torch.zeros_like(attention_mask), extra_mask], dim=-1)

        actions = output_tensor[..., 1:]
        logits, _ = self.model(output_tensor, attention_mask=training_mask, with_no_loss=True)
        logits = self.sanatize_logits(logits)

        if torch.isnan(logits).any():
            print("NaN in logits!")
            exit()
        logsumexp = torch.logsumexp(logits, dim=-1)
        selected_logits = torch.gather(logits, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        log_probs = selected_logits - logsumexp

        del attention_mask
        del mask_addition
        del extra_mask
        del X
        del x
        del logits
        del logsumexp
        del selected_logits
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            base_logits = self.use_base_model(output_tensor.detach(), attention_mask=training_mask)
            logsumexp_base = torch.logsumexp(base_logits, dim=-1)
            selected_base_logits = torch.gather(base_logits, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            base_log_probs = selected_base_logits - logsumexp_base

        if torch.isnan(base_logits).any():
            print("NaN in base_logits!")
            exit()

        del training_mask
        del base_logits
        del logsumexp_base
        del selected_base_logits
        gc.collect()
        torch.cuda.empty_cache()

        if torch.isnan(base_log_probs).any():
            print("NaN in base_log_probs!")
            exit()

        if torch.isnan(log_probs).any():
            print("NaN in log_probs!")
            exit()

        ratio_clamp = torch.clamp(log_probs - base_log_probs, min=-10, max=10)
        probs_ratio_batch = torch.exp(ratio_clamp)
        divergence = log_probs - base_log_probs

        if torch.isnan(divergence).any():
            print("NaN in divergence!")
            exit()

        del log_probs
        del base_log_probs
        del ratio_clamp
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            reward_batch = []
            for tensor_per_generation in output_tensor.detach().cpu():
                considered_tensor = tensor_per_generation
                reward = self.extract_reward(considered_tensor, input_sequence_length=input_sequence_length)
                # reward to be calculated per token
                reward_batch.append(torch.tensor(reward, dtype=self.dtype))
            # print(f"Probs ratio batch : {probs_ratio_batch}")
            reward_batch = torch.stack(reward_batch)
        reward_batch = reward_batch.to(self.model_device)

        advantage = (reward_batch - reward_batch.mean())*2.0  # Normalize advantages and scale
        advantage = torch.clamp(advantage, min=0.0)  # Only consider positive advantages for the loss calculation
        if advantage.mean() <= 0:
            advantage = reward_batch
        # print(f"Reward batch : {reward_batch} : Advantage : {advantage}")
        advantage = advantage.unsqueeze(-1).unsqueeze(-1)
        
        product = advantage.float() * probs_ratio_batch.float()
        product = torch.nan_to_num(product, 0.0)
        product = product.half()

        product_with_clipping = torch.clamp(probs_ratio_batch, 1.0 - self.epsilon, 1.0 + self.epsilon)*advantage
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
