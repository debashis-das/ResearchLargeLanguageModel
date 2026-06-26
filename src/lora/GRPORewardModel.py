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

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, grpo_batch: int, model_device="cpu", dtype=torch.float16, total_generation_length=10):
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
            p.data += 0.005 * torch.randn_like(p)    
        self.base_model = deepcopy(model)
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def use_base_model(self, tokens, attention_mask):
        self.base_model.eval()
        with torch.no_grad():
            base_logits, _ = self.base_model(tokens, attention_mask=attention_mask, with_no_loss=True)
        return base_logits.detach()

    def board_state(self, moves, chess_board: ChessGame, reward = 0.0, ignore_moves_till = 0, play_as="white"):
        # print(f"Processing moves for board state: {moves} | Ignore moves till: {ignore_moves_till} | Play as: {play_as}")
        moves_clean = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', moves.strip())
        # print("Moves cleaned for board state processing: ", moves_clean)
        # Match: move_number. white_move [black_move]
        pattern = r'(\d+)\.\s+(\S+)(?:\s+(?!\d+\.)(\S+))?'
        count_valid_moves = 0
        move_no = 0
        # print(f"-----------> PLay as : {play_as}")
        for m in re.finditer(pattern, moves_clean):
            try:
                move_no = int(m.group(1))
                if move_no < ignore_moves_till:
                    continue
                white   = m.group(2)
                black   = m.group(3)  # None if Black didn't play (resignation)
                if ignore_moves_till > 0:
                    print(f"[Ignore till {ignore_moves_till}] Processing move number {move_no} : White move : {white} : Black move : {black}")
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
                        break
                if black is None or white is None:
                    break
            except Exception as e:
                print(f"An error occurred while processing moves: {e}")
                traceback.print_exc()
                break
        if count_valid_moves == 0:
            reward -= 1.0
        else:
            reward += count_valid_moves * 0.5
        print(f"Board state processing complete. Total valid moves: {count_valid_moves}, Reward: {reward}, Last move number processed: {move_no}")
        return chess_board, reward, move_no

    # reward for proper format of the output
    def chess_reward_function(self, prompt, generation):
        try:
            # print(f"{prompt} \n\n\n-------------------------------\n\n\n")
            game = ChessGame()
            # print(f"Processing output for reward calculation: {prompt} | {generation}")
            # input extraction and create board state based on the input moves
            input_moves =  (prompt.rsplit("Task:")[0].strip().rsplit("moves:")[-1].strip())
            play_as = (prompt.split("play_as:")[-1].strip().split("moves:")[0].strip())
            # print(f"Input moves extracted for board state initialization: {input_moves}")
            game, _, move_no = self.board_state(input_moves, game, play_as=play_as)
            # print(f"Base moves : {move_no} : Play as : {play_as}")
            print(f"Processing output for reward calculation: {generation}")
            _, current_reward, generation_move_no = self.board_state(generation.strip(), game, ignore_moves_till = move_no, play_as=play_as)
            if current_reward < 0:
                return -1.0
            print(f"[Ideal case] Generated new moves : {generation_move_no - move_no} : Reward : {current_reward}")
            if play_as == "white" and "1-0" in generation:
                current_reward += 10.0
            elif play_as == "black" and "0-1" in generation:
                current_reward += 10.0
            elif play_as in ["white", "black"] and "1/2-1/2" in generation:                            
                current_reward += 5.0
            return current_reward
        except Exception as e:
            print(f"An error occurred during move processing: {e}")
            traceback.print_exc()
        return 0.0

    def extract_reward(self, tensor_per_generation: torch.Tensor, input_sequence_length: int):
        prompt = self.tokenizer.decode(tensor_per_generation[:input_sequence_length], skip_special_tokens=True)
        generation = self.tokenizer.decode(tensor_per_generation[input_sequence_length:], skip_special_tokens=True)
        reward = self.chess_reward_function(prompt, generation)
        print("-------------------------------------------------------------")
        return reward

    def sanatize_logits(self, logits: torch.Tensor, actions: torch.Tensor):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = torch.clamp(logits, min=-50, max=50)  # Clamp logits to avoid extreme values
        selected_logits = torch.gather(logits, dim=-1, index=actions).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        # print(f"logits: {logits.shape} : actions: {actions.shape} : logsumexp: {logsumexp.shape}")
        # logits = torch.nan_to_num(logits, nan=0.0)
        # logits = logits / logits.sum(dim=-1, keepdim=True)  # Normalize to get probabilities
        # print(f"Selected logits: {selected_logits.shape}")
        log_probs = selected_logits - logsumexp
        return log_probs  # Convert back to half precision
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
        x = x.unsqueeze(0)
        input_sequence_length = x.shape[-1]
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        attention_mask = attention_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the attention mask for the batch size
        with torch.no_grad():
            output_tensor = self.model.generate(X.to(self.model_device), max_new_tokens=self.total_generation_length, temperature=0.5)

        mask_addition = output_tensor.shape[-1] - attention_mask.shape[-1]
        extra_mask = torch.ones(mask_addition, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0)
        extra_mask = extra_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the extra mask for the batch size
        training_mask = torch.cat([torch.zeros_like(attention_mask), extra_mask], dim=-1)

        actions = output_tensor[..., 1:]
        logits, _ = self.model(output_tensor, attention_mask=training_mask, with_no_loss=True)
        log_probs = self.sanatize_logits(logits, actions.unsqueeze(-1))

        if torch.isnan(log_probs).any():
            print("NaN in log_probs!")
            exit()

        del attention_mask
        del mask_addition
        del extra_mask
        del X
        del x
        del logits
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            base_logits = self.use_base_model(output_tensor.detach(), attention_mask=training_mask)
            logsumexp_base = torch.logsumexp(base_logits, dim=-1)
            selected_base_logits = torch.gather(base_logits, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            base_log_probs = selected_base_logits - logsumexp_base

        if torch.isnan(base_log_probs).any():
            print("NaN in base_logits!")
            exit()

        del training_mask
        del base_logits
        del logsumexp_base
        del selected_base_logits
        gc.collect()
        torch.cuda.empty_cache()

        divergence = torch.clamp(log_probs - base_log_probs, min=-10, max=10)
        ratio = torch.exp(divergence)

        if torch.isnan(divergence).any():
            print("NaN in divergence!")
            exit()

        del log_probs
        del base_log_probs
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            reward_batch = []
            for tensor_per_generation in output_tensor.detach().cpu():
                considered_tensor = tensor_per_generation
                reward = self.extract_reward(considered_tensor, input_sequence_length=input_sequence_length)
                # reward to be calculated per token
                reward_batch.append(torch.tensor(reward, dtype=self.dtype))
                print("-------------------------------------------------------------")
            # print(f"Probs ratio batch : {probs_ratio_batch}")
            reward_batch = torch.stack(reward_batch)
        reward_batch = reward_batch.to(self.model_device)
        advantage = reward_batch - reward_batch.mean()
        advantage = advantage / (advantage.abs().mean() + 1e-6) # Normalize advantages
        weights = 1.5*torch.tanh(advantage) + 0.1  # smooth gating
        weights = weights + 0.01 * torch.sign(weights)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        
        product = weights.float() * ratio.float()
        product_clamped = weights.float() * torch.clamp(ratio.float(), 1.0 - self.epsilon, 1.0 + self.epsilon)
        loss = -torch.min(product, product_clamped) + self.beta * divergence
        print(f"Loss : {loss.mean()} : Advantage weight : {weights.mean()} : Product : {product.mean()} : Product with clipping : {product_clamped.mean()} : Divergence : {divergence.mean()}")

        del reward_batch
        del advantage
        del product
        del product_clamped
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

if __name__ == "__main__":
    sample_prompt = f"""
        <system>
        You are a strong chess engine.
        Output must be strictly in SAN format with move numbers.
        No explanations, no extra text.

        <user>
        play_as: white
        moves: "1. d4 e6 2. a3 Nc6 3. Nc3 Bb4 4. axb4 a5 5. b5 Nb4 "

        Task:
        - Continue the game
        - Play optimally
        - End only at checkmate or resignation
        - Output only moves

        <assistant>"""
    generation_text = f"""6. g4 g5 7. h4 gxh4 8. g5 h3 9. g6 h2 10. g7 h1=Q 11. g8=Q+ Ke7 12. Qg5+ Kd6 13. Qc5#"""
    # print(chess_reward_function(sample_prompt, generation_text))
    
