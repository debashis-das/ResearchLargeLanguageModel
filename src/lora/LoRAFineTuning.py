import gc
import traceback
import logging

from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import torch
from torch.utils.checkpoint import checkpoint

from lora.loRALinear import loRALinear

class LoRAFineTuning(nn.Module):
    """
    LoRA Fine Tuning by injecting LoRA modules into the specified linear layers of the model. 
    The projections parameter allows you to specify which linear layers to inject the LoRA modules into.
    If not provided, it defaults to injecting into the query, key, and value projection layers of the attention mechanism. 
    The forward method implements the forward pass through the model, while the generate method implements text generation using the model with LoRA fine-tuning.
    """
    _default_projections = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj"         # MLP
    ]
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 projections=None, 
                 rank=16, alpha=32, 
                 dtype=torch.float16, device=torch.device("cuda")):
        super().__init__()
        if projections is None:
            projections = self._default_projections
        assert type(projections) is list
        self.projections = projections
        self.tokenizer = tokenizer
        self.rank = rank
        self.alpha = alpha
        self.device = device
        self.model = model
        self.dtype = dtype
        self.create_module_dict_inject_loRA()
        self.loss_function = nn.CrossEntropyLoss()
        self.layers = self.model.config.num_hidden_layers
        self.multi_gpu_dict = {}  
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency     
    
    def create_module_dict_inject_loRA(self):
        self.module_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.requires_grad_(False)
                if any(projection in name for projection in self.projections):
                    # print(f"Injecting LoRA module into {name} with shape {module.weight.shape}")
                    loRA_linear = loRALinear(module, rank=self.rank, alpha=self.alpha, dtype=self.dtype, device=self.device)
                    parent_path, attr_name = name.rsplit(".", 1)
                    parent_module = self.model.get_submodule(parent_path)
                    setattr(parent_module, attr_name, loRA_linear)

        for name, param in self.model.named_parameters():
            if param.requires_grad and "loRA" in name:
                continue
            else:
                param.requires_grad_(False)
        
        for name, module in self.model.named_modules():
            self.module_dict[name] = module

    def qwen_attention_mask(self, batch_size, attention_mask: torch.Tensor| None):
        # Qwen model expects attention mask of shape [batch, seq_len] with 1 for tokens to attend to and 0 for tokens to ignore.
        # We need to convert it to the shape [batch, 1, seq_len, seq_len] with -inf for tokens to ignore and 0 for tokens to attend to.
        if attention_mask is None:
            return None
        min_val = torch.finfo(self.dtype).min
        mask = []
        for idx, postion in enumerate(attention_mask[-1].tolist()):
            if idx == 0 and postion == 0:
                mask.append([min_val] * len(attention_mask[-1]))
            elif postion == 1:
                mask.append([0.0] * (idx+1) + [min_val] * (len(attention_mask[-1]) - (idx+1)))
            else:
                mask.append(mask[-1])
        final_mask = torch.tensor(mask, device=attention_mask.device, dtype=self.dtype)
        final_mask = final_mask.unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        return final_mask

    def action_per_layer(self, current_layer_number, X, attention_mask=None, position_embeddings=None, cache_position=None, kv_cache=None):
        try:
            x_projection = self.module_dict[f"model.layers.{current_layer_number}.input_layernorm"](X)
            attn_output, _ = self.module_dict[f"model.layers.{current_layer_number}.self_attn"](hidden_states=x_projection, 
                                                                                                            attention_mask=attention_mask, 
                                                                                                            position_embeddings=position_embeddings,
                                                                                                            past_key_values=kv_cache,
                                                                                                            cache_position=cache_position
                                                                                                            )
            o_projection_residual = attn_output + X
            o_projection_norm = self.module_dict[f"model.layers.{current_layer_number}.post_attention_layernorm"](o_projection_residual)
            o_mlp = self.module_dict[f"model.layers.{current_layer_number}.mlp"](o_projection_norm)
            return o_mlp + o_projection_residual
        except Exception as e:
            traceback_info = traceback.format_exc()
            logging.error(f"Error in layer {current_layer_number} : {e} : {traceback_info}")
            raise

    def forward(self, X, attention_mask=None, with_no_loss=False):
        try:
            assert len(X.shape) in (1, 2), (
                f"Expected input_ids of shape [seq_len] or [batch, seq_len], got {X.shape}"
            )
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            batch_size, seq_len = X.shape
            attention_mask = self.qwen_attention_mask(batch_size, attention_mask)
            position_ids = torch.arange(seq_len, device=X.device).unsqueeze(0)
            input = self.module_dict["model.embed_tokens"](X)
            cos, sin = self.module_dict["model.rotary_emb"](input, position_ids)  # (cos, sin)
            for layer_number in range(self.model.config.num_hidden_layers):
                input = self.action_per_layer(layer_number, input, attention_mask=attention_mask, position_embeddings=(cos, sin))
            input = self.module_dict["model.norm"](input)
            logits_batch = self.module_dict["lm_head"](input)   # [batch, seq_len, vocab_size]
            if with_no_loss:
                return logits_batch[:, :-1, :], None
            # Shift logits and labels for next-token prediction
            output_logits = logits_batch[:, :-1, :].contiguous()  # Shift logits for next-token prediction
            B, S, V = logits_batch.shape
            logits = logits_batch.view(B * S, V)
            X = X.view(B * S)
            shifted_logits = logits[...,:-1,:].contiguous()
            shifted_labels = X[..., 1:].contiguous()
            # Compute loss
            loss = self.loss_function(shifted_logits, shifted_labels)
            return output_logits, loss
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, temperature=0.0):
        try:
            print(f"Generating text with input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape if attention_mask is not None else 'None'}, max_new_tokens: {max_new_tokens}, temperature: {temperature}")
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            # self.multi_gpu_spread(single_gpu=True)
            self.eval()  # Set the model to evaluation mode
            kv_cache = DynamicCache(config=self.model.config)  
            assert len(input_ids.shape) in (1, 2), (
                f"Expected input_ids of shape [seq_len] or [batch, seq_len], got {input_ids.shape}"
            )
            X = input_ids
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            batch, init_seq_len = input_ids.shape
            # prefill
            cache_position = torch.arange(init_seq_len, device=X.device)  # Positions for the initial sequence
            position_ids = cache_position.unsqueeze(0)
            X = self.module_dict["model.embed_tokens"](X)
            position_embeddings = self.module_dict["model.rotary_emb"](X, position_ids)  # (cos, sin)
            attention_mask = self.qwen_attention_mask(batch_size=batch, attention_mask=attention_mask) if attention_mask is not None else None
            for layer_number in range(self.model.config.num_hidden_layers):
                X = self.action_per_layer(layer_number, X, attention_mask=attention_mask, position_embeddings=position_embeddings, cache_position=cache_position, kv_cache=kv_cache)
            X = self.module_dict["model.norm"](X)
            logits = self.module_dict["lm_head"](X)   # [batch, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]   # [batch, vocab_size]
            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Greedy decoding
            else:
                next_token = self.temperature_sampling(temperature, next_token_logits, batch_size=batch)
            generated_ids = next_token
            # with tqdm(
            #     total       = max_new_tokens,
            #     desc        = "Generating text",
            #     unit        = "tokens",
            #     bar_format  = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            #     ncols       = 120,       # width of the bar
            #     colour      = "green",   # optional color
            # ) as pbar: 
                # generate new tokens one by one.
            for idx in range(max_new_tokens):
                cache_position = torch.tensor([init_seq_len + idx], device=self.device)  # Positions for the new token
                position_ids = cache_position.unsqueeze(0)
                next_token = self.module_dict["model.embed_tokens"](next_token)
                position_embeddings = self.module_dict["model.rotary_emb"](next_token, position_ids)  # (cos, sin)
                for layer_number in range(self.model.config.num_hidden_layers):
                    next_token = self.action_per_layer(layer_number, next_token, position_embeddings=position_embeddings, cache_position=cache_position, kv_cache=kv_cache)
                next_token = self.module_dict["model.norm"](next_token)
                logits = self.module_dict["lm_head"](next_token)   # [batch, seq_len, vocab_size]
                next_token_logits = logits[:, -1, :]   # [batch, vocab_size]
                if temperature == 0.0:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Greedy decoding
                else:
                    next_token = self.temperature_sampling(temperature, next_token_logits, batch_size=batch)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                # if idx % 100 == 0:
                    # print(f"Generated token {idx+1}/{max_new_tokens}")  
                    # pbar.update(1)
            # print(f"Input prompt: {self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)}")  # Debugging line to check input prompt
            # print(f"Generated text: {self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")  
            return torch.cat([input_ids, generated_ids], dim=-1)
        except Exception as e:
            exec_info = traceback.format_exc()
            logging.error(f"Error during text generation {exec_info}")
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            self.train() 

    def temperature_sampling(self, temperature, next_token_logits, batch_size):
        next_token_logits = next_token_logits / temperature  # Apply temperature scaling
        next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        next_token_logits = torch.clamp(next_token_logits, min=-50, max=50)  # Clamp logits to avoid extreme values
        next_token_logits = next_token_logits.float()  # Ensure logits are in float32 for softmax
        next_token_logits = torch.nn.functional.softmax(next_token_logits, dim=-1)  
        next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0)
        next_token_logits = next_token_logits / next_token_logits.sum(dim=-1, keepdim=True)  # Normalize to get probabilities
        next_token = torch.distributions.Categorical(next_token_logits).sample((batch_size,))  # Sample from the distribution
        del next_token_logits
        return next_token
    
if __name__ == "__main__":
    model_path = "C:\\Users\\DebashisDas\\personal\\models\\Qwen"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto"
            )
    for name, module in model.named_modules():
        print(name)
    