import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
import gc
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, AutoModel, EarlyStoppingCallback, AutoModelForSequenceClassification
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import argparse
import torch.nn.functional as F
import ctypes
import gc
libc = ctypes.CDLL("libc.so.6")


def main(model_path, dpath, spath):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(example):
        context = "[CLS] " + example['passages'] + " [SEP] " + example['questions'] + " [SEP]"
        tokenized_example = tokenizer(context, truncation=True, padding=True, add_special_tokens=False)
        return tokenized_example

    test = pd.read_parquet(f"{dpath}")
    test_dataset = Dataset.from_pandas(test)

    # Preprocess the dataset
    tokenized_dataset = test_dataset.map(preprocess, remove_columns=['questions', 'passages', 'labels', 'answer'])

    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    model.eval()

    training_args = TrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=1024,
        do_predict=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    output = trainer.predict(tokenized_dataset)

    predictions = output.predictions

    softmax_predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    prob = softmax_predictions[:, 1]
    pred = np.argmax(predictions, axis=1)

    test['preds'] = pred
    test['prob'] = prob

    print("roc_auc_score: ", roc_auc_score(test['labels'], test['prob']))
    print(classification_report(test['labels'], test['preds'], digits=4))
    print("accuracy_score: ", accuracy_score(test['labels'], test['preds']))

    test_prob_np = test.to_numpy()

    _ = gc.collect()
    libc.malloc_trim(0)

    ### 데이터셋 재정렬
    sorted_arrays = []

    for i in tqdm(range(0, len(test_prob_np), 20)):
        array_to_sort = test_prob_np[i:i+20]
        sorted_array = array_to_sort[array_to_sort[:, 5].argsort()[::-1]]
        sorted_arrays.append(sorted_array)

    # 정렬된 데이터프레임들을 다시 병합
    final_array = np.vstack(sorted_arrays)
    final_df = pd.DataFrame(final_array, columns=test.columns)

    final_df.to_csv(f'{spath}', index=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument('-model_path', '--model_path', type=str, help='model path', required=True)
    parser.add_argument('-dpath', '--dpath', type=str, help='dataset path', required=True)
    parser.add_argument('-spath', '--spath', type=str, help='dataset save path', required=True)

    args = parser.parse_args()
    return args.model_path, args.dpath, args.spath


if __name__ == '__main__':
    main(*parse_args())
