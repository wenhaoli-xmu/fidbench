# -*- coding: utf-8 -*-
"""
GPTQ 4-bit 权重量化 Modifier
"""
import os
import shutil
from ..modifier import Modifier
import torch
import torch.nn.functional as F
from awq import AutoAWQForCausalLM                      
from transformers import AutoTokenizer, AwqConfig

class AWQMethod(Modifier):


    def __init__(self,
                 model, tokenizer,
                 save_ckp: str | None = None,
                 load_ckp: str | None = None,
                 config : dict | None = None):

        self.get_conf(config)
        cfg = self.conf or {}
        if load_ckp and os.path.isdir(load_ckp) and os.listdir(load_ckp):
            print(f"[AWQ] loading quantized model from {load_ckp}")
            final_model = AutoAWQForCausalLM.from_quantized(
                load_ckp, device_map="auto", trust_remote_code=True)
            tokenizer   = AutoTokenizer.from_pretrained(load_ckp, use_fast=True)

        else:
            w_bit        = cfg.get("w_bit",          4)
            q_group_size = cfg.get("q_group_size", 128)
            zero_point   = cfg.get("zero_point",   True)
            version      = cfg.get("version",      "GEMM")


            base_id = getattr(model, "name_or_path", None)
            if base_id is None:
                raise ValueError("Can't find model.name_or_path; "
                                 "please set `model_name` in awq.json")

            print(f"[AWQ] reload weights via AutoAWQForCausalLM.from_pretrained({base_id})")
            final_model = AutoAWQForCausalLM.from_pretrained(base_id)
            tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

            quant_config = {"zero_point":   zero_point, 
                            "q_group_size": q_group_size,
                            "w_bit":        w_bit, 
                            "version":      version}

            print("[AWQ] start quantization …")
            final_model.quantize(tokenizer, quant_config=quant_config)

            if save_ckp:
                os.makedirs(save_ckp, exist_ok=True)
                print(f"[AWQ] saving quantized model to {save_ckp}")
                final_model.save_quantized(save_ckp)
                tokenizer.save_pretrained(save_ckp)

        final_model.eval()                    
        final_model.to("cuda:0")          # 全模型搬到单卡
        torch.cuda.empty_cache()          # 释放其他卡上残留显存
        super().__init__(final_model, save_ckp=None, load_ckp=None)
        self.tokenizer = tokenizer             # 若外部想用



    def ft_params(self):
        return []                

    def reset(self):
        pass

    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        output = self.model(input_ids=p_ids)
        pkv      = output.past_key_values if "past_key_values" in output else None

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        for tok, label in zip(torch.chunk(g_ids[:, :-1], turns, -1),
                              torch.chunk(g_ids[:,  1:], turns, -1)):
            kw = {"past_key_values": pkv, "use_cache": True} if pkv is not None else {"use_cache": True}
            out   = self.model(input_ids=tok, **kw)
            logits, pkv = out.logits, getattr(out, "past_key_values", pkv)

            y   = label.view(-1).item()
            top1= logits.argmax(-1).view(-1).item()
            top5= logits.topk(k=5, dim=-1).indices.view(-1).tolist()
            acc1 += (top1 == y)
            acc5 += (y in top5)

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
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

        ppl = torch.exp(loss).item()

        return ppl