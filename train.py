from typing import Optional, Union
import pandas as pd
import numpy as np
# from colorama import Fore, Back, Style
from tqdm.notebook import tqdm
import torch
from datasets import Dataset
import gc
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, AutoModel, EarlyStoppingCallback, AutoModelForSequenceClassification
import wandb
import os
import json
from sklearn.metrics import f1_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def compute_metrics(p):
    preds = p.predictions.tolist()
    input_array = np.array(preds)
    max_indices = np.argmax(input_array, axis=1)
    max_index = max_indices.tolist()

    labels = p.label_ids.tolist()

    f1_macro = f1_score(labels, max_index, average='macro')
    f1_micro = f1_score(labels, max_index, average='micro')
    acc = accuracy_score(labels, max_index)
    return {"f1-macro": f1_macro,
            "f1-micro": f1_micro,
            "acc":acc}



def main():
    model_path = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(example):
        context = "[CLS] " + example['passages'] + " [SEP] " + example['questions'] +" [SEP]"
        tokenized_example = tokenizer(context, truncation=True, padding=True)
        tokenized_example['label'] = example['labels']

        return tokenized_example


    train = pd.read_parquet("/home/jisukim/playground-NLP/DPR/selected_model/datasets/train_top10_dataset.parquet")
    valid = pd.read_parquet("/home/jisukim/playground-NLP/DPR/selected_model/datasets/valid_top10_dataset.parquet")


    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid)


    tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=['questions','passages', 'labels'])
    tokenized_valid_dataset = valid_dataset.map(preprocess, remove_columns=['questions','passages', 'labels'])

    device = torch.device('cuda:0')

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)


    training_args = TrainingArguments(
        output_dir='./output_top10',
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        evaluation_strategy="steps",
        warmup_ratio=0.8,
        learning_rate=2e-6,
        eval_steps=500,
        logging_steps=500,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        num_train_epochs=5,
        report_to=['wandb'],
        seed=42,
        metric_for_best_model='acc',
        save_strategy='steps'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics = compute_metrics,
    )


    trainer.train()

if __name__ == "__main__":
    main()