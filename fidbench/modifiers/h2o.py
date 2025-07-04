import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from ..modifier import Modifier

import math
from typing import Tuple, Optional
import copy


global num_layers
num_layers = 0


class Qwen2Attention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_key_value_groups = self.num_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def _reset_masks(self):
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if not hasattr(self, 'layer_idx'):
            global num_layers
            self.layer_idx = copy.deepcopy(num_layers)
            num_layers += 1

        bsz, q_len, _ = hidden_states.size()

        try:
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        except:
            import IPython
            IPython.embed(header='debug')

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        previous_scores = []
        attn_output_list = []
        attention_masks_next_list = []

        num_heads = query_states.shape[1]
        for head_idx in range(num_heads):
            query_head = query_states[:, head_idx: head_idx + 1, ...]
            key_head = key_states[:, head_idx: head_idx + 1, ...]
            value_head = value_states[:, head_idx: head_idx + 1, ...]

            attn_weights = query_head @ key_head.transpose(-2,-1) / math.sqrt(self.head_dim)

            if attention_mask is None and q_len > 1:
                seq_idx = torch.arange(q_len, dtype=torch.int64, device=hidden_states.device)
                attention_mask = torch.where(seq_idx[:, None] >= seq_idx[None, :], 0, float('-inf')).to(hidden_states.dtype)[None, None, :, :]
            
            if attention_mask is not None:
                attn_weights += attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            if self.attention_masks_next is not None:
                attention_mask_next_head = self.attention_masks_next[:, head_idx: head_idx + 1]
                attn_weights = attn_weights * attention_mask_next_head + (1 - attention_mask_next_head) * torch.finfo(attn_weights.dtype).min

            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

            current_scores_sum = attn_weights.sum(0).sum(1)

            if not self.previous_scores == None:
                current_scores_sum[:, :-1] += self.previous_scores[head_idx: head_idx + 1]
            else:
                self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
                self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
                self.cache_budget = self.heavy_budget + self.recent_budget
                self.cache_budget_records.append(self.cache_budget)
                self.input_length.append(attn_weights.shape[-1])

            dtype_attn_weights = attn_weights.dtype
            attn_weights_devices = attn_weights.device

            previous_scores.append(current_scores_sum)

            attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

            attn_tokens_all = previous_scores[-1].shape[-1]
        
            if attn_tokens_all > self.cache_budget:
                if not self.recent_budget == 0:
                    attn_mask[:, :-self.recent_budget] = 0
                    selected_set = previous_scores[-1][:, :-self.recent_budget]
                else:
                    attn_mask[:, :] = 0
                    selected_set = previous_scores[-1]

                if not self.heavy_budget == 0:
                    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

            attention_masks_next_list.append(attn_mask.clone().unsqueeze(0).unsqueeze(2))

            score_mask = attn_mask[:,:-1]
            score_mask[:, -self.recent_budget:] = 1

            previous_scores[-1] = previous_scores[-1] * score_mask

            attn_output = torch.matmul(attn_weights, value_head)
            attn_output_list.append(attn_output)
        
        attn_output = torch.cat(attn_output_list, dim=1)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        self.attention_masks_next = torch.cat(attention_masks_next_list, dim=1)
        self.previous_scores = torch.cat(previous_scores, dim=0)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_kvcache_llama_heavy_recent(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)

        if isinstance(module, Qwen2Attention):
            new_module = Qwen2Attention_heavy_hitter(config)
            device = next(model._modules[name].parameters()).device
            dtype = next(model._modules[name].parameters()).dtype
            state_dict = model._modules[name].state_dict()
            new_module = new_module.to(device=device, dtype=dtype)
            new_module.load_state_dict(state_dict)
            model._modules[name] = new_module

    return model


class H2O(Modifier):
    def reset(self):
        ...


    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        model.config.heavy_ratio = self.conf['heavy_ratio']
        model.config.recent_ratio = self.conf['recent_ratio']
        model = convert_kvcache_llama_heavy_recent(model, model.config)

        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []
    

    def reset_mask(self):
        for layer in self.model.model.layers:
            layer.self_attn._reset_masks()


    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        self.reset_mask()
        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        output = self.model(input_ids=p_ids)
        kv_cache = output.past_key_values

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        for tok, label in zip(
                torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                torch.chunk(g_ids[:, 1:], turns, dim=-1)):

            output = self.model(input_ids=tok, past_key_values=kv_cache)
            logits, kv_cache = output.logits, output.past_key_values

            label = label.ravel().item()
            next_1 = logits.argmax(dim=-1).ravel().item()
            next_5 = logits.topk(k=5, dim=-1).indices.ravel().tolist()

            acc1 += next_1 == label
            acc5 += label in next_5

        acc1 /= turns
        acc5 /= turns

        return acc1, acc5

    @torch.no_grad()
    def compute_ppl(self, input_ids):
        assert input_ids.shape[0] == 1, 'only support batch size 1'
        assert input_ids.ndim == 2

        self.reset_mask()
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        output = self.model(input_ids=input_ids)
        logits = output.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean')

        ppl = torch.exp(loss).item()

        return ppl
