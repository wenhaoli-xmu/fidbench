import json, os
import torch
import json
import numpy as np
import random
import argparse
from fidbench.misc import get_model_and_tokenizer

from pygments.console import colorize
from utils import process_doc as post_process
import transformers
transformers.logging.set_verbosity_error()


def log_exceed(l, max_l):
    msg = colorize('red', "[WARNING]: ") + f'Expect input token length plus `max_gen` to be less equal than {max_l}, but got {l}.'
    print(msg, flush=True)


def log_warning(msg):
    msg = colorize('red', "[WARNING]: ") + msg
    print(msg, flush=True)


def log_info(msg):
    msg = colorize('green', "[INFO]: ") + msg
    print(msg, flush=True)


default_template = """{{ bos_token | default('') }}
{%- for message in messages %}
    {%- if message.role == "user" %}
<|start_header_id|>user<|end_header_id|>

{{ message.content.strip() }}<|eot_id|>
    {%- elif message.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>

{{ message.content.strip() }}<|eot_id|>
    {%- elif message.role == "system" %}
<|start_header_id|>system<|end_header_id|>

{{ message.content.strip() }}<|eot_id|>
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{%- endif %}"""


# modified: 将参数max_length去掉
def pred( 
        tokenizer, 
        model,
        data_path, 
        out_path,
        max_gen,
        model_max_length,
        **generation_kwargs):

    with open(data_path, 'r') as f:
        for line in f:
            chats = json.loads(line)['conversations']

            if tokenizer.chat_template is None:
                log_warning('No chat template provided in `tokenizer_config.json`, use default chat template.')
                tokenizer.chat_template = default_template
            
            input_ids = tokenizer.apply_chat_template(chats, return_tensors='pt', add_generation_prompt=True)

            max_possible_length = input_ids.shape[-1] + max_gen
            if max_possible_length >= model_max_length:
                log_exceed(max_possible_length, model_max_length)

            # work around for tokenizer without pad token
            if tokenizer.pad_token_id == None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            output_ids = model.generate(
                input_ids.cuda(), 
                max_new_tokens=max_gen,
                **generation_kwargs)

            output_ids = output_ids.ravel().tolist()
            output_ids = output_ids[input_ids.shape[-1]:]
            
            in_and_out = dict(
                p_ids=input_ids.ravel().tolist(),
                g_ids=output_ids)

            with open(out_path, 'a+') as fo:
                fo.write(json.dumps(in_and_out) + '\n')


def compute_accuracy( 
        model,
        ref_path,
        model_max_length):
    

    acc1_list = []
    acc5_list = []

    with open(ref_path, 'r') as f:
        for line in f:
            in_and_out = json.loads(line)
            input_ids = torch.tensor(in_and_out['p_ids']).unsqueeze(0)
            refer_ids = torch.tensor(in_and_out['g_ids']).unsqueeze(0)

            max_possible_length = input_ids.shape[-1] + refer_ids.shape[-1]
            if max_possible_length >= model_max_length:
                log_exceed(max_possible_length, model_max_length)

            acc1, acc5 = model.compute_accuracy(
                p_ids=input_ids,
                g_ids=refer_ids)
            
            acc1_list.append(acc1)
            acc5_list.append(acc5)

    return acc1_list, acc5_list


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--max_gen", type=int, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--force-recompute", action='store_true')
    args = parser.parse_args()
    
    with open(args.env_conf, "r") as f:
        env_conf = json.load(f)

    gen_kwargs = env_conf['generation_kwargs']
    run_name = args.env_conf.replace('.json', '')
    os.makedirs("pred", exist_ok=True)

    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    
    for path in os.listdir('data'):
        if 'jsonl' not in path:
            continue

        data_path = os.path.join('data', path)
        out_path = os.path.join("pred", run_name, path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if args.label is not None:
            ref_path = os.path.join(args.label, path)
            acc1, acc5 = compute_accuracy(
                model,
                ref_path,
                env_conf['model']['max_length'])
            acc1 = sum(acc1) / len(acc1)
            acc5 = sum(acc5) / len(acc5)
            log_info(f"{path.replace('.jsonl', ''):^20}\t{colorize('yellow','Acc@1:')} {acc1:.5f}\t{colorize('yellow','Acc@5:')} {acc5:.5f}")
        else:
            if os.path.exists(out_path) and not args.force_recompute:
                log_info(f"Skip {path} since {out_path} exists.")
                continue
            pred(
                tokenizer,
                model,
                data_path,
                out_path,
                max_gen=args.max_gen,
                model_max_length=env_conf['model']['max_length'],
                **env_conf['generation_kwargs'])
