from ..modifier import Modifier
import torch
import torch.nn as nn
import bitsandbytes as bnb

class QuantizedInference(Modifier):

    def __init__(self, model, save_ckp=None, load_ckp=None, config=None):

        # 如果不需要配置文件，则不需要下面几行
        # self.get_conf(config)
        # a = self.conf['a']
        # b = self.conf['b']
        # ...

        super().__init__(model, save_ckp, load_ckp)


    # 不用管，return [] 就行
    def ft_params(self):
        return []

    # 不用管，pass就行
    def reset(self):
        pass

    # 这个必须有，输入为p_ids (prefilling tokens) 以及 g_ids (generation tokens)，分别对应上下文，以及base model生成的那部分
    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        # pre-fill, 可能要变动
        output = self.model(input_ids=p_ids)
        kv_cache = output.past_key_values

        print(f"p_ids shape: {p_ids.shape}, g_ids shape: {g_ids.shape}")
        print(f"p_ids content (first 10): {p_ids[:, :10]}")
        print(f"g_ids content (first 10): {g_ids[:, :10]}")

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        if turns <= 0:
            print(f"Warning: turns is {turns}, g_ids shape is {g_ids.shape}. Returning 0 accuracy.")
            return 0.0, 0.0

        print(f"Initial turns: {turns}")

        for i, (tok, label_tensor) in enumerate(zip(
                torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                torch.chunk(g_ids[:, 1:], turns, dim=-1))):

                # generation, 可能要变动
                output = self.model(input_ids=tok, past_key_values=kv_cache) # Changed kv_cache to past_key_values
                logits, kv_cache = output.logits, output.past_key_values

                label = label_tensor.ravel().item()
                next_1 = logits.argmax(dim=-1).ravel().item()
                next_5 = logits.topk(k=5, dim=-1).indices.ravel().tolist()

                print(f"Turn {i+1}:")
                print(f"  tok: {tok}")
                print(f"  label: {label}")
                print(f"  next_1: {next_1}")
                print(f"  next_5: {next_5}")
                print(f"  logits (shape): {logits.shape}") # Added logits shape

                acc1 += next_1 == label
                acc5 += label in next_5

        print(f"Final raw acc1: {acc1}, acc5: {acc5}, turns: {turns}")

        acc1_val = acc1 / turns if turns > 0 else 0.0
        acc5_val = acc5 / turns if turns > 0 else 0.0

        print(f"Calculated acc1: {acc1_val}, acc5: {acc5_val}")

        return acc1_val, acc5_val