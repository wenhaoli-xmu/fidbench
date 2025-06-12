import os
import shutil
from ..modifier import Modifier
import torch
import torch.nn.functional as F
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPTQMethod(Modifier):

    def __init__(self,
                 model, 
                 tokenizer,
                 save_ckp=None,
                 load_ckp=None,
                 config: dict | None = None):

        # 读取 config (fidbench 自动注入的字段)
        self.get_conf(config)
        conf = self.conf
        if load_ckp and os.path.isdir(load_ckp) and os.listdir(load_ckp):
            print(f"[GPTQ] load quantized model from {load_ckp}")
            final_model = AutoModelForCausalLM.from_pretrained(
                load_ckp, device_map="auto", trust_remote_code=True)
        else: 
            bits          = conf.get("bits",               4)
            group_size    = conf.get("group_size",       128)
            nsamples      = conf.get("num_samples",      128)
            model_seqlen  = conf.get("model_seqlen",    2048)
            damp          = conf.get("damp_percent",   0.01)
            desc_act      = conf.get("desc_act",       False)  
            sym           = conf.get("sym",             True)
            true_seq      = conf.get("true_sequential", True)
            disable_ex    = conf.get("disable_exllama", True)

            quantizer = GPTQQuantizer(
                bits              = bits,
                group_size        = group_size,
                dataset           = "wikitext2",       
                nsamples          = nsamples,
                model_seqlen      = model_seqlen,
                damp_percent      = damp,
                desc_act          = desc_act,
                sym               = sym,
                true_sequential   = true_seq,
                disable_exllama   = disable_ex,
            )
      
                
            print("[GPTQ] start quantization …")
            final_model = quantizer.quantize_model(model, tokenizer=tokenizer)
            if save_ckp:
                os.makedirs(save_ckp, exist_ok=True)
                print(f"[GPTQ] saving quantized model to {save_ckp}")
                final_model.save_pretrained(save_ckp, safe_serialization=True)
                tokenizer.save_pretrained(save_ckp)

        final_model.eval()                                      
        super().__init__(final_model, save_ckp=None, load_ckp=None)

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
        pkv      = output.past_key_values      

        acc1, acc5 = 0, 0
        turns = g_ids.shape[-1] - 1

        for tok, label in zip(
                torch.chunk(g_ids[:, :-1], turns, dim=-1), 
                torch.chunk(g_ids[:, 1:], turns, dim=-1)):

            output = self.model(input_ids=tok, past_key_values=pkv)
            logits, pkv = output.logits, output.past_key_values

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
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

        ppl = torch.exp(loss).item()

        return ppl