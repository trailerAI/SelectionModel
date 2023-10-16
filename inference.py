import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
import gc
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def inference():
    model_path = "/home/jisukim/playground-NLP/DPR/selected_model/output_top5/checkpoint-274500"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(example):
        context = "[CLS] " + example['passages'] + " [SEP] " + example['questions'] +" [SEP]"
        tokenized_example = tokenizer(context, truncation=True, padding=True)
        tokenized_example['label'] = example['labels']

        return tokenized_example
    
    test = pd.read_parquet("/home/jisukim/playground-NLP/DPR/selected_model/datasets/test_top5_dataset.parquet")
    test_dataset = Dataset.from_pandas(test)

    tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=['questions','passages', 'labels'])
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=128, drop_last=False, shuffle=False, collate_fn=data_collator)
    device = torch.device('cuda:0')

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    model.eval()

    preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            pred_label = output['logits'].detach().cpu().tolist()
            max_indices = np.argmax(np.array(pred_label), axis=1)
            preds.append(max_indices.tolist())
            
    preds = sum(preds, [])

    test['pred'] = preds

    acc = []
    macro_f1 = []
    micro_f1 = []

    for i in tqdm(range(int(len(test)/5))):
        acc.append(accuracy_score(test.iloc[i*5:(i+1)*5]['pred'].values, test.iloc[i*5:(i+1)*5]['labels'].values))
        macro_f1.append(f1_score(test.iloc[i*5:(i+1)*5]['pred'].values, test.iloc[i*5:(i+1)*5]['labels'].values, average='macro'))
        micro_f1.append(f1_score(test.iloc[i*5:(i+1)*5]['pred'].values, test.iloc[i*5:(i+1)*5]['labels'].values, average='micro'))

    print(np.mean(acc))
    print(np.mean(macro_f1))
    print(np.mean(micro_f1))

    test[['pred', 'labels','questions']].to_csv('test_result_top5.csv', index=False)

if __name__ == "__main__":
    inference()