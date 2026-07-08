import re
import traceback

from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from chess.chess_validator import ChessGame
from lora.LoRAFineTuning import LoRAFineTuning

SAN_REGEX = re.compile(
        r"""^(
            # Castling
            O-O(-O)?[+#]? |

            # Piece moves (optional piece letter)
            ([KQRBN])?                  # piece
            ([a-h])?                   # disambiguation file
            ([1-8])?                   # disambiguation rank
            x?                         # capture
            [a-h][1-8]                 # target square

            (= [QRBN])?                # promotion (with space handled below)
            (= [QRBN])?                # promotion (no space variant)

            [+#]?                      # check or mate
        )$""",
        re.VERBOSE
    )

class GRPORewardModel(nn.Module):

    def __init__(self, 
                 tokenizer_path: str, 
                 model_path: str, 
                 grpo_batch: int,
                 loRA_parameters_path: str = None, 
                 dtype=torch.float16, 
                 total_generation_length=300):
        super(GRPORewardModel, self).__init__()
        self.tokenizer_path = tokenizer_path
        self.grpo_batch = grpo_batch
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.gamma = 0.99
        self.epsilon = 0.05
        self.beta = 0.02
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        init_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device_map="auto"
                )
        self.model_device = init_model.device
        self.model = LoRAFineTuning(init_model, self.tokenizer, dtype=dtype, device=init_model.device)
        if loRA_parameters_path:
            self.model.load_lora_parameters(loRA_parameters_path)

        # Reference model must be loaded independently — sharing init_model causes zero divergence,
        # frozen LoRA weights, and nested LoRA corruption on the second injection pass.
        # Loaded in int8 since it's frozen (no gradients) and only used for reference log-probs.
        base_init_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
        self.base_model = LoRAFineTuning(base_init_model, self.tokenizer, dtype=torch.bfloat16, device=base_init_model.device)
        for param in self.base_model.parameters():
            param.requires_grad = False

    def use_base_model(self, tokens, attention_mask):
        self.base_model.eval()
        base_model_device = self.base_model.device
        with torch.no_grad():
            logits, _ = self.base_model(tokens.to(base_model_device), attention_mask=attention_mask.to(base_model_device), with_no_loss=True)
        logits.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        logits.clamp_(min=-50, max=50)  # int8 quant noise can spike a token's logits; keep in sync with sanatize_logits
        return logits.to(self.model_device).detach()

    def is_valid_san(self, move: str) -> bool:
        """
        Validate if a move is syntactically valid SAN (no board legality).
        """
        move = move.strip()
        # Normalize spaces like "e8 = Q"
        move = move.replace(" = ", "=")
        return bool(SAN_REGEX.match(move))

    def board_state(self, prompt_moves, generation, play_as="white"):
        chess_board = ChessGame()
        san = r'(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)'
        pattern = rf'(\d+)\.\s+({san})(?:\s+(?!\d+\.)({san}))?'
        san_str = re.findall(pattern, prompt_moves)
        init_moves_made = 0
        count_valid_moves = 0
        reward = 0.0
        for m in san_str:
            for san_values in m:
                for san_values in san_values.split(" "):
                    if len(san_values.strip()) > 0 :
                        if san_values.isdigit():
                            init_moves_made = int(san_values)
                        elif san_values[-1] == "." and san_values[:-1].isdigit():
                            init_moves_made = int(san_values[:-1])
                        else:
                            if self.is_valid_san(san_values):
                                ok, _ = chess_board.push_san(san_values)
                                if ok:
                                    count_valid_moves += 1

        san = r'(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)'
        pattern = rf'(({san}\s+)?(?:\d+\.\s+{san}(?:\s+(?!\d+\.){san})?\s*)+)'
        san_str = re.findall(pattern, generation)
        atleast_one_valid_move = False
        generation_move_no = -1
        all_correct_moves = 0
        for m in san_str:
            for san_values in m:
                    for san_values in san_values.split(" "):
                        if len(san_values.strip()) > 0 :
                            if san_values.isdigit() and int(san_values) < init_moves_made:
                                generation_move_no = int(san_values)
                            elif san_values[-1] == "." and san_values[:-1].isdigit():
                                generation_move_no = int(san_values[:-1])
                            else:
                                if self.is_valid_san(san_values):
                                    atleast_one_valid_move = True
                                    if generation_move_no < init_moves_made or generation_move_no != -1:
                                        continue
                                    ok, _ = chess_board.push_san(san_values)
                                    if ok:
                                        reward += 1
                                        all_correct_moves += 1
                                        print(f"Valid move made: {san_values} : Current reward: {reward}")
        if generation.count("moves:") > 1:
            reward -= 1.0
        if atleast_one_valid_move:
            reward += 0.5
        if all_correct_moves > 20:
            reward += 15.0
        elif all_correct_moves > 10:
            reward += 10.0
        elif all_correct_moves > 5:
            reward += 5.0
        if reward >= 1:
            print(chess_board.board_string())
        if reward == 0 and not atleast_one_valid_move:
            reward = -1.0
        return reward

    def extract_reward(self, tensor_per_generation: torch.Tensor, input_sequence_length: int):
        prompt = self.tokenizer.decode(tensor_per_generation[:input_sequence_length], skip_special_tokens=True)
        generation = self.tokenizer.decode(tensor_per_generation[input_sequence_length:], skip_special_tokens=True)
        # print(f"Prompt : {prompt} \n\n Generation : {generation}")
        try:
            init_prompt = prompt.strip().rsplit("<user>")[-1].strip()
            input_moves =  (init_prompt.rsplit("moves:")[-1].strip())
            play_as = (init_prompt.split("play_as:")[-1].strip().split("moves:")[0].strip())
            return self.board_state(input_moves, generation.strip(), play_as=play_as)
        except Exception as e:
            print(f"An error occurred during move processing: {e}")
            traceback.print_exc()
        return 0.0

    def sanatize_logits(self, logits: torch.Tensor, actions: torch.Tensor):
        logits.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        logits.clamp_(min=-50, max=50)  # Clamp logits to avoid extreme values
        selected_logits = torch.gather(logits, dim=-1, index=actions).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        log_probs = selected_logits - logsumexp
        return log_probs  # Convert back to half precision
    
    def generate(self, input_ids):
        with torch.no_grad():
            self.model.eval()
            output_tensor = self.model.generate(input_ids.to(self.model_device), 
                                            max_new_tokens=self.total_generation_length, 
                                            sampling=True,
                                            temperature=0.7
                                            )
        return output_tensor

    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        # self.current_step += 1
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
        x = x.unsqueeze(0)
        input_sequence_length = x.shape[-1]
        X = x.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the input tensor for the batch size
        attention_mask = attention_mask.repeat_interleave(repeats=self.grpo_batch, dim=0)  # Repeat the attention mask for the batch size
        output_tensor = self.generate(X.to(self.model_device))
        
        with torch.no_grad():
            reward_batch = []
            for tensor_per_generation in output_tensor.detach().cpu():
                considered_tensor = tensor_per_generation
                reward = self.extract_reward(considered_tensor, input_sequence_length=input_sequence_length)
                # reward to be calculated per token
                reward_batch.append(torch.tensor(reward, dtype=self.dtype))
            reward_batch = torch.stack(reward_batch)

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

        divergence = torch.clamp(log_probs - base_log_probs, min=-10, max=10)
        ratio = torch.exp(divergence)

        if torch.isnan(divergence).any():
            print("NaN in divergence!")
            exit()

        del log_probs
        del base_log_probs

        reward_batch = reward_batch.to(self.model_device).float()
        normalized_reward = torch.tanh(reward_batch).float()
        std = normalized_reward.std()
        print(f"Normalized reward : {normalized_reward.dtype}", normalized_reward)
        print(f"Std : {std.dtype}", std)
        advantage = torch.zeros_like(normalized_reward) if std < 1e-4 and torch.all(normalized_reward < 0) else normalized_reward # Normalize advantages
        print("Advantage after normalization: ", advantage)
        weights = advantage * 2.0
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        print("Advantage weights : ", weights)
        # `ratio` is unbounded above (exp of a divergence clamped only to +-10, i.e. up to ~22026),
        # so the unclamped `product` term can reach magnitudes that overflow the fp16 LoRA
        # parameters' gradients during backward. Clamp the loss itself, before backward ever
        # sees it, to the same order of magnitude as the other sanitized quantities in this file.
        product = weights.float() * ratio.float()
        product_clamped = weights.float() * torch.clamp(ratio.float(), 1.0 - self.epsilon, 1.0 + self.epsilon)
        loss = -torch.min(product, product_clamped) + self.beta * divergence
        loss.nan_to_num_(nan=0.0, posinf=50.0, neginf=-50.0)
        loss.clamp_(min=-50.0, max=50.0)
        print(f"Advantage {weights.mean()}: {weights}")
        print(f"Reward batch : {reward_batch.mean()} : {reward_batch}")
        print(f"Loss : {loss.mean()} : Advantage : {weights.mean()} : Product : {product.mean()} : Product with clipping : {product_clamped.mean()} : Divergence : {divergence.mean()}")

        del advantage
        del product
        del product_clamped
        del divergence
        del weights

        if torch.isnan(loss).any():
            print("NaN in loss!")
            exit()
        
        return loss.mean(), reward_batch

    def debug_logs(self, X, idx, tensor_per_generation):
        print(f"Input      : {self.tokenizer.decode(X[idx], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")
        print(f"Generation : {self.tokenizer.decode(tensor_per_generation[X.shape[-1]:], skip_special_tokens=True)}")
        print("-------------------------------------------------------------")

if __name__ == "__main__":
    # reward_batch = torch.tensor([0.5000]*18, dtype=torch.float32)
    reward_batch = torch.tensor([0.5]*18, dtype=torch.float32)
    normalized_reward = torch.tanh(reward_batch)
    print("Normalized reward : ", normalized_reward)
    std = normalized_reward.std()
    print("Std : ", std)
    print(f"Normalized reward : {normalized_reward - normalized_reward.mean()}")
    advantage = torch.zeros_like(normalized_reward) if std < 1e-4 and torch.all(normalized_reward < 0) else normalized_reward # Normalize advantages
    print(f"Advantage normalization : {(normalized_reward - normalized_reward.mean()) / (std + 1e-6)}")
    print("Advantage : ", advantage)
    weights = advantage * 2.0
    weights = weights.unsqueeze(-1).unsqueeze(-1)  
    print("Advantage weights : ", weights.mean())  

    
