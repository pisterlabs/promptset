# use one GPU or CPU. Data parallel problem
# python train_GPT.py  --tokenizer ./tokenizer/   --train_data  ../MT_Dataset/hy_mix/train.txt  --valid_data ../MT_Dataset/hy_mix/valid.txt  --output_dir ./output

import torch

from transformers import OpenAIGPTConfig
   
from transformers import BertTokenizerFast

from transformers import OpenAIGPTLMHeadModel

from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments

from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tokenizer', default='tokenizer/',
                        help='path to vocab directory')
                                         
parser.add_argument('--train_data', default='train.txt',
                        help='path to training data')    
                                         
parser.add_argument('--valid_data', default='valid.txt',
                        help='path to validation data')  
                      
parser.add_argument('--output_dir', default='./',
                        help='output directory') 
                      
parser.add_argument('--n_position', type=int, default=256,
	                    help='the number of position')                                     

parser.add_argument('--n_embd', type=int, default=180,
	                    help='the number of embedding')  

parser.add_argument('--n_layer', type=int, default=12,
	                    help='the number of layer')  


parser.add_argument('--n_head', type=int, default=6, 
	                    help='the number of head. It is divisible by n_embd')  


parser.add_argument('--epochs', type=int, default=1000,
	                    help='epochs')  

parser.add_argument('--train_batch', type=int, default=256,
	                    help='train batch size')  

parser.add_argument('--valid_batch', type=int, default=256,
	                    help='valid batch size') 
args = parser.parse_args()


tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, max_len=512, do_lower_case=False)

config = OpenAIGPTConfig(
    vocab_size = 120,     
    n_positions = args.n_position,
    n_embd = args.n_embd,
    n_layer = args.n_layer,
    n_head = args.n_head,
)

model = OpenAIGPTLMHeadModel(config=config)

datasets = load_dataset("text", data_files={"train": args.train_data, "validation": args.valid_data})

def tokenize_function(examples):
    return tokenizer(examples["text"])
    
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print("example: ", lm_datasets["train"][0])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.1
)
    
    
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.train_batch,
    per_device_eval_batch_size=args.valid_batch,
    save_steps=5000,    # save checkpoints every 5000 steps
    save_total_limit=80,  # Up to 80 checkpoints can be stored
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator
)
trainer.train()
trainer.save_model(args.output_dir)