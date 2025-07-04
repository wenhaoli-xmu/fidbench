import torch
import torch.nn.functional as F
import types
from .utils import check_and_apply_qk_rope, do_projection
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from ..modifier import Modifier
from flash_attn_interface import flash_attn_func


def model_forward(self, input_ids, kv_cache=None):
    """
    Input
    -----
    :input_ids: input indices
    :kv_cache: key value cache
    :kwargs: To absorb useless arguments passed by lib peft
    """
    hidden_states, kv_cache = self.model(input_ids, kv_cache)
    logits = self.lm_head(hidden_states)
    return CausalLMOutputWithPast(
        logits=logits,
        past_key_values=kv_cache)


def model_model_forward(self, input_ids, kv_cache):

    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    for layer_idx, layer in enumerate(self.layers):
        tmp = kv_cache[layer_idx]
        hidden_states, tmp = layer(hidden_states, tmp)
        kv_cache[layer_idx] = tmp
        
    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache


def layer_forward(self, hidden_states, kv_cache):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kv_cache = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual.to(hidden_states.device) + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache


def self_attn_forward(self, hidden_states, kv_cache):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    head_dim = embed_dim // num_heads

    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim)

    # position embedding
    past_length = 0 if kv_cache is None else kv_cache[0].shape[2]

    pos = torch.arange(past_length, past_length + keys.shape[2])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    ques = ques.to(torch.float8_e4m3fn)
    keys = keys.to(torch.float8_e4m3fn)
    vals = vals.to(torch.float8_e4m3fn)

    if kv_cache is not None:
        key_cache, val_cache = kv_cache
        keys = torch.cat([key_cache, keys], dim=2)
        vals = torch.cat([val_cache, vals], dim=2)
    kv_cache = (keys, vals)

    # attention computation
    attn_output = flash_attn_func(
        ques.transpose(1,2), 
        keys.transpose(1,2), 
        vals.transpose(1,2),
        causal=True)[0].transpose(1,2)
    attn_output = attn_output.to(torch.bfloat16)

    attn_output = attn_output.transpose(1,2).flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output, kv_cache



class FlashAttnentionFP8(Modifier):
    def reset(self):
        ...


    def __init__(self, model, save_ckp, load_ckp, config):

        # 修改各种forward函数
        model.forward = types.MethodType(model_forward, model)
        model.model.forward = types.MethodType(model_model_forward, model.model)

        for layer in model.model.layers:
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []

    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        output = self.model(input_ids=p_ids)
        kv_cache = output.past_key_values

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        for tok, label in zip(
                torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                torch.chunk(g_ids[:, 1:], turns, dim=-1)):

            output = self.model(input_ids=tok, kv_cache=kv_cache)
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

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        output = self.model(input_ids=input_ids)
        logits = output.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).to(shift_labels.device),
            shift_labels.view(-1),
            reduction='mean'
        )

        ppl = torch.exp(loss).item()

        return ppl
