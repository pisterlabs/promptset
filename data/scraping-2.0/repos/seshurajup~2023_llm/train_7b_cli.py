# Standard libraries
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Third-party libraries
from datasets import load_dataset
from IPython.display import Markdown, display
import torch
import transformers
import wandb

# Transformers specific
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

# Langchain specific
from langchain.prompts import PromptTemplate

# 4bit training specific
from peft import LoraConfig
import trl
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ['TRANSFORMERS_CACHE'] = './hug_models/'

# Command-line arguments
parser = argparse.ArgumentParser(description='Command-line interface for the model')
parser.add_argument('--hug', type=str, default="Salesforce/xgen-7b-4k-base", help='ID of the model')
parser.add_argument('--seed', type=int, default=np.random.randint(0,10_000_000), help='Random seed')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA parameter r')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA parameter alpha')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
parser.add_argument('--train_batch_size', type=int, default=2048, help='Training Batch Size')
parser.add_argument('--eval_batch_size', type=int, default=2048, help='Eval Batch Size')
args = parser.parse_args()

api_key = "894fc801a85dc79ac36d24f3f2ac70cebe580ecf"
seed = np.random.randint(0,10_000_000)
trl.set_seed(seed)

hug = args.hug
model_name = hug.split("/")[-1]
max_seq_length=args.max_seq_length
train_batch_size=args.train_batch_size
eval_batch_size=args.eval_batch_size
lora_r=args.lora_r
lora_alpha=args.lora_alpha
lora_dropout=args.lora_dropout
epochs = args.epochs
output_name = f"{model_name}-s-{seed}-epochs-{epochs}-lora_r-{lora_r}-max_seq-{max_seq_length}"
print("*"*10+hug+"*"*10)
config = {}
config['hug'] = hug
config['max_seq_length'] = max_seq_length
config['train_batch_size'] = train_batch_size
config['eval_batch_size'] = eval_batch_size
config['lora_r'] = lora_r
config['lora_alpha'] = lora_alpha
config['lora_dropout'] = lora_dropout
config['epochs'] = epochs
config['output_name'] = output_name
config['seed'] = seed
wandb.login(key=api_key)
wandb.init(project='llm', name=output_name, config=config)

train_test = pd.read_csv("/datadrive1/input/kaggle-llm-science-exam/train.csv")
test = train_test[41:93+1].reset_index()
train_1k = pd.read_csv("/datadrive1/input/wikipedia-stem-1k/stem_1k_v1.csv")
train = pd.concat([train_test[0:41], train_test[94:], train_1k]).reset_index()
train.to_csv("/datadrive1/input/train.csv")
test.to_csv("/datadrive1/input/test.csv")
print("Training Samples", len(train), "Testing Samples", len(test))

data = load_dataset("csv", data_files={'train': "/datadrive1/input/train.csv",'test': "/datadrive1/input/test.csv"})
data

template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
E) {e}\n

Answer: {answer}"""

prompt = PromptTemplate(template=template, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e', 'answer'])

sample = data['train'][0]
display(Markdown(prompt.format(prompt=sample['prompt'], 
                               a=sample['A'], 
                               b=sample['B'], 
                               c=sample['C'], 
                               d=sample['D'], 
                               e=sample['E'], 
                               answer=sample['answer'])))
def format_text(example):
    text = prompt.format(prompt=example['prompt'], 
                         a=example['A'], 
                         b=example['B'], 
                         c=example['C'], 
                         d=example['D'], 
                         e=example['E'], 
                         answer=example['answer'])
    return {"text": text}

data = data.map(format_text)
data

def plot_sequence_lengths(data, split='train', max_length=2048):
    sequence_lengths = []
    keep_indices = []

    # Loop over the dataset and get the lengths of text sequences
    for i, example in enumerate(data[split]):
        sequence_lengths.append(len(example['text']))
        if sequence_lengths[i] < max_length:
            keep_indices.append(i)

    return keep_indices

keep_indices_train = plot_sequence_lengths(data, max_length=max_seq_length)
data['train'] = data['train'].select(keep_indices_train)
tokenizer = AutoTokenizer.from_pretrained(hug, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    hug,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

from peft import LoraConfig, get_peft_model
#prepare_for_int8_training
from peft import prepare_model_for_kbit_training
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)
target_modules=None
if 'falcon' in hug.lower():
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
if target is not None:
    peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    task_type="CAUSAL_LM"
    )
model = get_peft_model(model, peft_config)
training_args = TrainingArguments(
    output_dir=f"/datadrive1/train_models/{output_name}", 
    per_device_train_batch_size=144,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    logging_strategy="steps",
    #max_steps=1000,
    optim="paged_adamw_8bit",
    fp16=True,
    run_name="baseline-falcon-sft",
    warmup_ratio=0.1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    load_best_model_at_end = True
)

trainer = SFTTrainer(
    model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    #packing=True,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                  response_template="Answer:")
)

trainer.train()
trainer.save_model(f'./final_models/{output_name}/')

