import torch
import torch.nn.functional as F

try:
    from medusa.model.medusa_model_legacy import MedusaModel
    from medusa.model.kv_cache import KVCache
except ImportError:
    raise RuntimeError("Please install Medusa, download the training data, train it, and specify the correct checkpoint")

from ..modifier import Modifier


class Medusa(Modifier):
    def reset(self):
        ...


    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        del model
        model = MedusaModel.from_pretrained(self.conf['checkpoint'])
        super().__init__(model, save_ckp, load_ckp)


    def ft_params(self):
        return []
    

    @torch.no_grad()
    def compute_accuracy(self, p_ids, g_ids):

        assert p_ids.shape[0] == 1, 'only support batch size 1'
        assert p_ids.ndim == 2 and g_ids.ndim == 2

        device = next(iter(self.model.parameters())).device
        dtype = next(iter(self.model.parameters())).dtype
        p_ids, g_ids = p_ids.to(device), g_ids.to(device)

        # ============================================================================================
        config = self.model.config
        # Initializing the batch size to 1, this can be modified if different batch sizes are required
        batch_size = 1
        # Initializing a tensor to store past keys and values for all layers
        past_key_values_data = torch.zeros(
            config.num_hidden_layers * 2,
            batch_size,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            device=device,
            dtype=dtype)
        current_length_data = torch.zeros(
            config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
        )
        past_key_values = [] * config.num_hidden_layers
        for i in range(config.num_hidden_layers):
            past_key_values.append(
                [
                    KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        kv_cache = past_key_values
        # ============================================================================================

        self.model(input_ids=p_ids, past_key_values=kv_cache, output_orig=False)
        num_parallel = len(self.model.medusa_head) + 1

        acc1, acc5 = 0, 0

        turns = g_ids.shape[1] - num_parallel

        for i in range(turns):

            tok = g_ids[:, i:i+1]
            label = g_ids[:, i+1:i+1+num_parallel]
            
            medusa_logits, _, logits = self.model(
                input_ids=tok,
                past_key_values=kv_cache,
                output_orig=True)
            
            # [num_parallel,]
            next_1 = torch.cat([
                logits.argmax(dim=-1).ravel(),
                medusa_logits.argmax(dim=-1).ravel()])

            # [num_parallel, 5]
            next_5 = torch.cat([
                logits.topk(k=5, dim=-1).indices.flatten(1),
                medusa_logits.topk(k=5, dim=-1).indices.flatten(1)])
            
            acc1 += (next_1 == label.ravel()).sum().item()
            acc5 += torch.sum(next_5 == label.T).item()

        acc1 /= turns * num_parallel
        acc5 /= turns * num_parallel

        return acc1, acc5

    @torch.no_grad()
    def compute_ppl(self, input_ids):
        raise NotImplementedError
