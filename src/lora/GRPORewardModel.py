from torch import nn
import torch
from transformers import AutoTokenizer

from lora.LoRAFineTuning import LoRAFineTuning

class GRPORewardModel(nn.Module):

    def __init__(self, tokenizer: AutoTokenizer, model: LoRAFineTuning, reward: list, grpo_batch: int, alpha: float = 1.0, gamma: float = 2.0):
        super(GRPORewardModel, self).__init__()
        self.tokenizer = tokenizer
        self.iterations = [89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946]
        self.moves = [i for i in range(1, len(self.iterations) + 1)]
        self.current_iteration = 0
        self.current_index = 0 
        self.grpo_batch = grpo_batch
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.reward = reward

    # exclude output tensor from the whole prompt and only consider the newly generated tokens for move extraction
    # TODO: implement move extraction logic based on the output tensor and the defined moves and iterations
    # return the next n moves to be extracted based on the generated tensor and the index list which contains the indices of the generated text in the original prompt
    def move_extractor(self, whole_output_tensor_with_prompt: torch.Tensor):
        pass
        
    # Here x is the current state of the model with prompt
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        X = x.repeat_interleave(self.grpo_batch, dim=0)
        output_tensor = self.model.generate(X, max_new_tokens=100)
        extract_next_n_moves, index_list = self.move_extractor(output_tensor)
        if self.current_iteration < self.iterations[self.current_index]:
            current_number_of_moves = self.moves[self.current_index]
        else:
            self.current_index += 1
            current_number_of_moves = self.moves[self.current_index]
        extract_for_nll = extract_next_n_moves[...,:current_number_of_moves]
        
        reward = []
        for reward_function in self.reward:
            reward = reward_function(extract_next_n_moves)




        