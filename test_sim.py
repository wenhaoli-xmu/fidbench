import argparse
import os
from rouge_score.rouge_scorer import RougeScorer
import json
import numpy as np
import tqdm


def get_file_list(dir):
    assert os.path.exists(dir), f"{dir} dost not exists!"
    lst = list(filter(lambda x: isinstance(x, str) and x.endswith(".txt"), os.listdir(dir)))
    return lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir1', type=str)
    parser.add_argument('dir2', type=str)
    parser.add_argument('--metric', type=str, default='rougeL')
    args = parser.parse_args()

    lst1 = get_file_list(args.dir1)
    lst2 = get_file_list(args.dir2)
    lst = set(lst1).intersection(set(lst2))

    scorer = RougeScorer([args.metric], use_stemmer=True)
    result = {}

    for filename in tqdm.tqdm(lst):
        path1 = os.path.join(args.dir1, filename)
        path2 = os.path.join(args.dir2, filename) 

        with open(path1, 'r') as f1:
            with open(path2, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()

                if len(lines1) != len(lines2):
                    continue

                scores = []

                for line1, line2 in zip(lines1, lines2):

                    if args.dir1 == args.dir2:
                        assert line1 == line2

                    score1 = scorer.score(line1, line2)[args.metric].precision
                    score2 = scorer.score(line2, line1)[args.metric].precision
                    scores.append((score1 + score2) / 2)                      

                average = np.mean(scores)

                result[filename] = average
    
    
    for key, value in result.items():
        print(f"{key:30} {value}")
    print(f"average: {np.mean(list(result.values()))}")
