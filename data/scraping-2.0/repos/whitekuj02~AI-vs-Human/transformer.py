from ignite.utils import manual_seed
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import AdamW

from modules.dataloader import Datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss
from ignite.handlers import EarlyStopping

from torch.nn import CrossEntropyLoss
from torch import cat

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from ignite.metrics import Accuracy

import random
import os
import openai
import re

import argparse

def get_argument():
    parser = argparse.ArgumentParser(description="Ai VS Human")
    parser.add_argument("-ex", "--export_data", action="store_true", help="export naver review")
    parser.add_argument("-aug_gpt", "--augmentationByGPT", action="store_true", help="augmentation gpt")
    parser.add_argument("-val", "--validation", action="store_true", help="validation test")
    parser.add_argument("-ep", "--epochs", type=int, default=50, help="epoch")
    return parser.parse_args()

args = get_argument()

ex_data = args.export_data
aug_gpt = args.augmentationByGPT
val = args.validation
epochs = args.epochs

def set_seed(seed=42):
    np.random.seed(seed)  # 이 부분이 pandas의 sample 함수에도 영향을 줍니다.
    torch.manual_seed(seed)
    manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
cuda:0

train_data = pd.read_csv('./datasets/train.csv')

text = []
label = []
for i in range(len(train_data)):
    R = train_data.iloc[i,1:5].astype(str)
    l = train_data.iloc[i,5]
    one_hot_label = [0,0,0,0]
    one_hot_label[l-1] = 1
    for d,a in zip(R,one_hot_label):
        text.append(d.strip('"'))
        label.append(a)
## 외부 데이터 셋 (네이버 영화 리뷰) ##

if ex_data:
    naver_train_data = pd.read_csv('./datasets/ratings_train.txt', sep="\t", encoding="utf-8")

    naver_train_data.drop("id", axis=1, inplace=True)
    naver_train_data.dropna(subset=["document"], inplace=True)
    naver_train_data["label"] = 0

    # 랜덤하게 1000개의 인덱스를 선택
    # sample_indices = random.sample(range(len(naver_train_data)), 1000)
    sample_indices = range(500)

    for idx in sample_indices:
        text.append(naver_train_data["document"].iloc[idx].strip('"'))
        label.append(naver_train_data["label"].iloc[idx])

## GPT-3.5 로 AI 데이터 생성 augmentation.py 사용

if aug_gpt:
    df_aug = pd.read_csv("./datasets/aug.csv")
    for i in range(len(df_aug)):
        text.append(df_aug["text"].iloc[i].strip('"'))
        label.append(df_aug["label"].iloc[i])

print("train data len :" + str(len(text)))

# 정규화
for idx, t in enumerate(text):
    t = re.sub(r'[^가-힣a-zA-Z0-9 ]', '', t)
    text[idx] = t


## trainning 시작
train_data_pre = pd.DataFrame({
    "text" : text,
    "label" : label
})

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

tokenized_datasets = []
for text,label in zip(train_data_pre["text"],train_data_pre["label"]):
    token = tokenizer(text, padding="max_length", truncation=True)
    token["labels"] = label
    tokenized_datasets.append(token)

# tokenized_datasets 예시: 
# [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}, {"input_ids": [4, 5], "attention_mask": [1, 1]}]

input_ids = [example["input_ids"] for example in tokenized_datasets]
attention_mask = [example["attention_mask"] for example in tokenized_datasets]
labels = [example["labels"] for example in tokenized_datasets]

# 딕셔너리 형태로 정리
data_dict = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels" : labels
}

# 이제 Dataset.from_dict() 사용 가능
tokenized_datasets = Dataset.from_dict(data_dict)

# validation 사용
if val:
    train_size = 0.8
    split = tokenized_datasets.train_test_split(test_size=1-train_size)

    train_dataset = split["train"]
    val_dataset = split["test"]

    # 각 데이터셋의 형식을 torch로 설정
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # 데이터 로더를 생성
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.to(device)

    # # 모든 파라미터의 그래디언트 계산 비활성화
    # for param in model.parameters():
    #     param.requires_grad = False

    # # 분류기의 파라미터만 그래디언트 계산 활성화
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=5e-4)
    num_training_steps = epochs * len(train_dataloader)

    milestones_values = [
            (0, 5e-5),
            (num_training_steps, 0.0),
        ]
    lr_scheduler = PiecewiseLinear(
            optimizer, param_name="lr", milestones_values=milestones_values
        )
    
    def train_step(engine, batch):
        model.train()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    def validate_step(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            return (outputs.logits, batch["labels"])

    trainer = Engine(train_step)
    evaluator = Engine(validate_step)

    # Loss를 계산하기 위한 metric 추가
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    def output_transform(output):
        y_pred, y = output
        return y_pred.argmax(dim=1), y 
    
    # Accuracy를 계산하기 위한 metric 추가
    Accuracy(output_transform=output_transform).attach(evaluator, "accuracy")

    # ProgressBar 추가
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    # Learning rate scheduler 추가
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # 매 epoch 후 검증 데이터셋을 사용하여 성능 평가
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        pbar.log_message(f"Validation Results - Epoch: {engine.state.epoch}, Accuracy: {avg_accuracy:.2f}")

    # EarlyStopping 추가
    def score_function(engine):
        return engine.state.metrics["accuracy"]  # We want to maximize accuracy

    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_dataloader, max_epochs=epochs)

else:
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=8)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.to(device)

    # # 모든 파라미터의 그래디언트 계산 비활성화
    # for param in model.parameters():
    #     param.requires_grad = False

    # # 분류기의 파라미터만 그래디언트 계산 활성화
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=5e-4)
    num_training_steps = epochs * len(train_dataloader)

    milestones_values = [
            (0, 5e-5),
            (num_training_steps, 0.0),
        ]
    lr_scheduler = PiecewiseLinear(
            optimizer, param_name="lr", milestones_values=milestones_values
        )
    
    def train_step(engine, batch):
        model.train()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    pbar = ProgressBar()

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    trainer.run(train_dataloader, max_epochs=epochs)


model.eval()

preds = []
test_df = pd.read_csv('./datasets/test.csv')

with torch.no_grad():
    for idx in tqdm(range(len(test_df))):
        row = test_df.iloc[idx]
        logits_for_label1 = []  # 각 문장의 라벨 1에 대한 로짓값만 저장
        
        for i in range(1, 5):
            prompt = row[f"sentence{i}"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            logits_for_label1.append(logits[1])  # 라벨 1에 대한 로짓값 저장

        # 확률이 가장 높은 두 개의 인덱스를 얻기 위해 np.argsort를 사용
        top2_indices = np.argsort(logits_for_label1)[-2:]
        
        # 결과는 1부터 4의 라벨을 가지므로 1을 더해줍니다.
        top2_labels = [i + 1 for i in top2_indices]
        
        # 두 라벨을 문자열로 연결
        combined_label = str(top2_labels[0]) + str(top2_labels[1])
        preds.append(combined_label)

preds = [str(pred) for pred in preds]
preds[:5]

submit = pd.read_csv('./datasets/sample_submission.csv')
submit['label'] = preds
submit.head()

submit.to_csv('./result/baseline_submit.csv', index=False)