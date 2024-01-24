import json
from tqdm import tqdm
import pandas as pd
import argparse

def main(dpath, dtype):
    with open(f"{dpath}/selection_model_{dtype}_dataset.json") as f:
        data = json.load(f)

    # 20
    questions = []
    passages = []
    labels = []
    answer = []

    for i in tqdm(range(len(data))):
        for j in range(len(data[i]['passages'])):
            questions.append(data[i]['question'])
            passages.append(data[i]['passages'][j])
            labels.append(data[i]['labels'][j])
            answer.append(data[i]['answer'])


    df_top20 = pd.DataFrame()

    df_top20['questions'] = questions
    df_top20['passages'] = passages
    df_top20['labels'] = labels
    df_top20['answer'] = answer

    df_top20.to_parquet(f'{dpath}/{dtype}_top20_dataset.parquet')

def parse_args():
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument('-fpath', '--fpath', type=str, help='fpath', required=True)
    parser.add_argument('-dtype', '--dtype', type=str, help='dtype', required=True)

    args = parser.parse_args()
    return args.fpath, args.dtype

if __name__ == '__main__':
    main(*parse_args())
