import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
import gc
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import argparse
import torch.nn.functional as F


def main(model_path, dpath, spath):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(example):
        context = "[CLS] " + example['passages'] + " [SEP] " + example['questions'] +" [SEP]"
        tokenized_example = tokenizer(context, truncation=True, padding=True, add_special_tokens=False)
        tokenized_example['label'] = example['labels']

        return tokenized_example
    
    test = pd.read_parquet(f"{dpath}")
    test_dataset = Dataset.from_pandas(test)

    tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=['questions','passages', 'labels'])
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=128, drop_last=False, shuffle=False, collate_fn=data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()

    model.eval()

    preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            output = model(**batch)
            probabilities = F.softmax(output.logits, dim=-1)
            selected_probs = probabilities[:, 1]

            preds.extend(selected_probs.detach().cpu().tolist())
            
    test['pred'] = preds

    print(roc_auc_score(test['pred'], test['labels']))

    test.to_csv(f'{spath}', index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument('-model_path', '--model_path', type=str, help='model path', required=True)
    parser.add_argument('-dpath', '--dpath', type=str, help='dataset path', required=True)
    parser.add_argument('-spath', '--spath', type=str, help='dataset save path', required=True)

    args = parser.parse_args()
    return args.model_path, args.dpath, args.spath


if __name__ == '__main__':
    main(*parse_args())
