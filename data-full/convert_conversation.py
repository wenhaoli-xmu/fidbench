import json
import os

listdir = os.listdir(os.path.dirname(__file__))

for file in listdir:
    if file.endswith('.jsonl'):
        with open(os.path.join(os.path.dirname(__file__), file), 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        data = [
            {"conversations": [{"role": "user", "content": x['text']}]}
            for x in data
        ]

        with open(os.path.join(os.path.dirname(__file__), file), 'w') as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')