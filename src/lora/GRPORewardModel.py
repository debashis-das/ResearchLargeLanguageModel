from torch import nn
import torch
from transformers import AutoTokenizer

from lora.LoRAFineTuning import LoRAFineTuning

class GRPORewardModel(nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, reward: list, grpo_batch: int, device="cpu", dtype=torch.float16, total_generation_length=500, generation_evalution_length=100):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.grpo_batch = grpo_batch
        self.model = model
        self.reward = reward
        self.device = device
        self.dtype = dtype
        self.total_generation_length = total_generation_length
        self.generation_evalution_length = generation_evalution_length
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.8

    def extract_reward(self, tensor_per_generation: torch.Tensor):
        value_fn_str = self.tokenizer.decode(tensor_per_generation, skip_special_tokens=True)
        reward = 0.0
        for reward_function in self.reward:
            reward = reward_function(value_fn_str)
        return reward
    
    # batch_size provided as input is always 1 
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        x = x.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        X = x.repeat_interleave(self.grpo_batch, dim=0)
        print(f"Input shape after repeat_interleave: {X.shape}")
        output_tensor = self.model.generate(X, max_new_tokens=self.total_generation_length)
        print(f"Output tensor shape: {output_tensor.shape}")
        reward_consideration_reverse_idx = (self.total_generation_length - self.generation_evalution_length)*-1
        loss_batch = [] 
        for tensor_per_generation in output_tensor:
            considered_tensor = tensor_per_generation[...,:reward_consideration_reverse_idx]
            reward = self.extract_reward(considered_tensor)
            value_t_with_k_reward = self.extract_reward(tensor_per_generation)
            print(f"Reward extracted: {reward} : Value function with k reward extracted: {value_t_with_k_reward}")
            value_t_reward = 0.0
            mask_addition = considered_tensor.shape[-1] - attention_mask.shape[-1]
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], mask_addition), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1)
            print(f"Attention mask shape after concatenation: {attention_mask.shape}")
            _, nll = self.model(considered_tensor.unsqueeze(0), attention_mask=attention_mask)
            advantage = self.gamma * reward +(value_t_with_k_reward - value_t_reward)
            loss = nll * advantage
            print(f"Advantage : {advantage}, loss : {loss}")
            loss_batch.append(loss)
        return self.softmax(torch.stack(loss_batch)).mean()



            
            

        




        