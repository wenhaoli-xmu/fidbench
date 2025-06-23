import os, json
from pygments.console import colorize
import argparse
import random
from convert import process_doc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--context', type=int, required=True)
parser.add_argument('--max-worker', type=int, default=32)
parser.add_argument('--sample', type=int, default=None)
parser.add_argument('--error', type=int, default=4096)
parser.add_argument('--tokenizer', type=str, default="Qwen/Qwen2.5-14B-Instruct")
args = parser.parse_args()

DATA_PATH = [
    "/mnt/petrelfs/share_data/liwenhao/share-gpt/sg_90k_part1.json",
    "/mnt/petrelfs/share_data/liwenhao/share-gpt/sg_90k_part2.json"
]

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


def process_chat(chat):
    input_ids = tokenizer(chat['conversations'][0]['value']).input_ids
    if len(input_ids) > args.context - args.error and len(input_ids) < args.context + args.error:
        return chat['conversations'][0]['value']
    return None

texts = []

for data_path in DATA_PATH:
    with open(data_path, 'r') as f:
        data = json.load(f)

        if args.sample is not None:
            data = random.sample(data, k=args.sample)

        with ThreadPoolExecutor(max_workers=args.max_worker) as executor:
            future_to_chat = {executor.submit(process_chat, chat): chat for chat in data}
            for future in tqdm(as_completed(future_to_chat), total=len(data)):
                result = future.result()
                if result:
                    text = result
                    texts.append(text)


os.makedirs('data-long', exist_ok=True)

if len(texts) > 0:
    with open(f'data-long/long{args.context // 1024}.jsonl', 'w') as f:
        for text in texts:
            f.write(json.dumps({"text": process_doc(text)}) + '\n')