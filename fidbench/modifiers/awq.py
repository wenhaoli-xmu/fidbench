import os
import shutil
from ..modifier import Modifier
import torch
import torch.nn.functional as F           
from transformers import AutoModelForCausalLM


class AWQMethod(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):

        model_path = model.config._name_or_path
        del model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map='auto')
        super().__init__(model, save_ckp, load_ckp)

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
            shift_logits.view(-1, shift_logits.size(-1)).to(shift_labels.device),
            shift_labels.view(-1),
            reduction='mean'
        )

        ppl = torch.exp(loss).item()

        return ppl