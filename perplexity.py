import json, os
import torch
import json
import numpy as np
import random
import argparse
from fidbench.misc import get_model_and_tokenizer

from pygments.console import colorize
from utils import process_doc as post_process


def log_exceed(l, max_l):
    msg = colorize('red', "[WARNING]: ") + f'Expect input_ids to be less equal than {max_l}, but got {l}.'
    print(msg, flush=True)
    

def log_info(msg):
    msg = colorize('green', "[INFO]: ") + msg
    print(msg, flush=True)


def compute_ppl( 
        tokenizer, 
        model,
        data_path,
        model_max_length):
    

    ppl_list = []

    with open(data_path, 'r') as f:
        for line in f:
            text = json.loads(line)['text']

            input_ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids
            
            max_possible_length = input_ids.shape[-1]
            if max_possible_length > model_max_length:
                log_exceed(max_possible_length, model_max_length)

            ppl = model.compute_ppl(
                input_ids=input_ids)
            ppl_list.append(ppl)

    return sum(ppl_list) / len(ppl_list)


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
    parser.add_argument("--data", type=str, default='wt2.jsonl')
    args = parser.parse_args()
    
    with open(args.env_conf, "r") as f:
        env_conf = json.load(f)

    gen_kwargs = env_conf['generation_kwargs']
    run_name = args.env_conf.replace('.json', '')
    os.makedirs("pred", exist_ok=True)

    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

    ppl = compute_ppl(
        tokenizer,
        model,
        args.data,
        env_conf['model']['max_length'])

    log_info(f"{args.data.replace('.jsonl', ''):^20}\t{colorize('yellow','PPL:')} {ppl:.5f}")