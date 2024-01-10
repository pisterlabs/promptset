import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers import OpenAIGPTConfig, OpenAIGPTModel
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace
from torch.optim import Adam
import sys
import os
import time
from transformers import PreTrainedTokenizerFast
import pickle
import subprocess as sp
import os
import logging
import random
from sklearn.metrics import matthews_corrcoef, accuracy_score
import numpy as np
import argparse
from scipy import stats

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

TRAIN_DF_NAME = "train.csv"
VALID_DF_NAME = "valid.csv"
TEST_DF_NAME = "test.csv"

TASK_REGRESSION = "REGRESSION"
TASK_CLASSIFICATION = "CLASSIFICATION"

TOKEZNIER_BPE = "BPE"
TOKEZNIER_WPC = "WPC"
TOKEZNIER_UNI = "UNI"
TOKEZNIER_WORDS = "WORDS"
TOKEZNIER_PAIRS = "PAIRS"

UNK_TOKEN = "<UNK>"  # token for unknown words
SPL_TOKENS = [UNK_TOKEN]  # special tokens


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)

def add_arguments(parser):
    parser.add_argument("-t", "--tokenizer-type", type=str, choices=[TOKEZNIER_BPE, TOKEZNIER_WPC, TOKEZNIER_UNI, TOKEZNIER_WORDS, TOKEZNIER_PAIRS], default='WRL', help=f'which tokenizer to train options: ["{TOKEZNIER_BPE}", "{TOKEZNIER_WPC}", "{TOKEZNIER_UNI}", "{TOKEZNIER_WORDS}", "{TOKEZNIER_PAIRS}"]')
    parser.add_argument("-s", "--vocab-size", type=int, default=100, help=f'vocabulary size for the trained tokenziers: "{TOKEZNIER_BPE}", "{TOKEZNIER_WPC}" and "{TOKEZNIER_UNI}"')
    parser.add_argument("-r", "--results-path", type=str, default='.', help='path to save model, tokneizer and results csv')
    parser.add_argument("-l", "--layers-num", type=int, default=2, help='numbers of BERT layers')
    parser.add_argument("-a", "--attention-heads-num", type=int, default=2, help='numbers of BERT attention heads')
    parser.add_argument("-z", "--hidden-size", type=int, default=128, help='hidden size')
    parser.add_argument("-d", "--data-path", type=str, help='path to folder containing three files: train.csv, valid.csv and test.csv')
    parser.add_argument("-e", "--epochs", type=int, default=30, help='number of training epochs')
    parser.add_argument("-p", "--print-training-loss", type=int, default=1000, help='number of iteration before printing a log')
    parser.add_argument("-y", "--task-type", type=str, choices=[TASK_REGRESSION, TASK_CLASSIFICATION], required=True, help=f'task type: ["{TASK_REGRESSION}" or "{TASK_CLASSIFICATION}"]')
    parser.add_argument("-m", "--max-length", type=int, default=512, help=f'max tokens per seqeunce')

def prepare_tokenizer_trainer(alg, voc_size):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == TOKEZNIER_BPE:
        tokenizer = Tokenizer(BPE(unk_token = UNK_TOKEN))
        trainer = BpeTrainer(special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_UNI:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token= UNK_TOKEN, special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_WPC:
        tokenizer = Tokenizer(WordPiece(unk_token = UNK_TOKEN, max_input_chars_per_word=10000))
        trainer = WordPieceTrainer(special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_WORDS:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    elif alg == TOKEZNIER_PAIRS:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    else:
        exit(f'unknown tokenizer type, please use one of the following: ["{TOKEZNIER_BPE}", "{TOKEZNIER_WPC}", "{TOKEZNIER_UNI}", "{TOKEZNIER_WORDS}", "{TOKEZNIER_PAIRS}"]')
    
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer
    
def train_tokenizer(iterator, alg, vocab_size):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, vocab_size)
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer

def batch_iterator(dataset):
    batch_size = 10000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def train_biological_tokenizer(data_path, task_type, tokenizer_type, vocab_size, results_path, max_length):
    """
    Reads the data from folder, trains the tokenizer, encode the sequences and returns list of data for BERT training
    """
    df_train = pd.read_csv(os.path.join(data_path, TRAIN_DF_NAME))
    df_valid = pd.read_csv(os.path.join(data_path, VALID_DF_NAME))
    df_test = pd.read_csv(os.path.join(data_path, TEST_DF_NAME))

    if args.task_type == TASK_REGRESSION:
        logger.info(f'starting a REGRESSION task!')
        y_train = df_train['label'].astype(float).tolist()
        y_valid = df_valid['label'].astype(float).tolist()
        y_test = df_test['label'].astype(float).tolist()
        
        num_of_classes = 1
    elif args.task_type == TASK_CLASSIFICATION:
        logger.info(f'starting a CLASSIFICATION task!')
        df_train['label_numeric'] = pd.factorize(df_train['label'], sort=True)[0]
        df_valid['label_numeric'] = pd.factorize(df_valid['label'], sort=True)[0]
        df_test['label_numeric'] = pd.factorize(df_test['label'], sort=True)[0]
        y_train = df_train['label_numeric'].astype(int).tolist()
        y_valid = df_valid['label_numeric'].astype(int).tolist()
        y_test = df_test['label_numeric'].astype(int).tolist()
        
        num_of_classes = len(list(set(y_train))) # counts the number different classes
    else:
        exit(f'unknown type of task, got {task_type}. Aviable options are: {TASK_REGRESSION} for regression or {TASK_CLASSIFICATION} for classification')
    
    
    X_train = df_train['seq'].astype(str).tolist()
    X_valid = df_valid['seq'].astype(str).tolist()
    X_test = df_test['seq'].astype(str).tolist()

    if 'WORDS' == tokenizer_type:
        X_train = [' '.join([*aminos]) for aminos in X_train]
        X_valid = [' '.join([*aminos]) for aminos in X_valid]
        X_test = [' '.join([*aminos]) for aminos in X_test]
    elif 'PAIRS' == tokenizer_type:
        def create_pairs(sequences):
            results = []
            for amino in sequences:
                amino_spaces = [*amino]
                if len(amino_spaces[::2]) == len(amino_spaces[1::2]):
                    pairs = [i+j for i,j in zip(amino_spaces[::2], amino_spaces[1::2])]
                elif len(amino_spaces[::2]) < len(amino_spaces[1::2]):
                    lst = amino_spaces[::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(lst, amino_spaces[1::2])] #add an element to the first list
                else:
                    lst = amino_spaces[1::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(amino_spaces[::2], lst)] #add an element to the second list
                results.append(' '.join(pairs))
            return results.copy()
        X_train = create_pairs(X_train)
        X_valid = create_pairs(X_valid)
        X_test = create_pairs(X_test)
    
    logger.info(f'starting to train {tokenizer_type} tokenizer...')
    tokenizer = train_tokenizer(batch_iterator(X_train), tokenizer_type, vocab_size)
    tokenizer.enable_padding(length=max_length)
    logger.info(f'saving tokenizer to {results_path}...')
    tokenizer.save(os.path.join(results_path, "tokenizer.json"))
    
    def encode(X):
        result = []
        for x in X:
            ids = tokenizer.encode(x).ids
            if len(ids) > max_length:
                ids = ids[:max_length]
            result.append(ids)
        return result

    X_train_ids = encode(X_train)
    X_valid_ids = encode(X_valid)
    X_test_ids = encode(X_test)
    
    X_train_ids = [torch.tensor(item).to(device) for item in X_train_ids]
    y_train = [torch.tensor(item).to(device) for item in y_train]
    logger.info(f'loaded train data to device')
    train_dataset = Dataset(X_train_ids, y_train)

    X_valid_ids = [torch.tensor(item).to(device) for item in X_valid_ids]
    y_valid = [torch.tensor(item).to(device) for item in y_valid]
    logger.info(f'loaded valid data to device')
    valid_dataset = Dataset(X_valid_ids, y_valid)

    X_test_ids = [torch.tensor(item).to(device) for item in X_test_ids]
    y_test = [torch.tensor(item).to(device) for item in y_test]
    logger.info(f'loaded test data to device')
    test_dataset = Dataset(X_test_ids, y_test)
    
    return num_of_classes, train_dataset, valid_dataset, test_dataset

class BioBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModel, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # additional layers for the classification / regression task
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
       )

    def forward(self, ids, mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.transformer(
           ids, 
           attention_mask=mask,
           token_type_ids=token_type_ids,
           return_dict=False
        )

        sequence_output = torch.mean(sequence_output, dim=1)
        result = self.head(sequence_output)
        
        return result

def train_model(model, task_type, train_generator, valid_generator, test_generator, epochs, print_training_logs, results_path):
    if task_type == TASK_REGRESSION:
        loss_fn = nn.MSELoss()
    elif task_type == TASK_CLASSIFICATION:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00002)
    
    def calc_metrics_regression(model, generator):
        loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,y in generator:
                outputs = model(x)  
                outputs = outputs.to(torch.float)
                y_pred.append(outputs[0].item())
                y = y.to(torch.float)
                loss += loss_fn(outputs, y).item()
                y_true.append(y.item())
            loss = loss / len(generator)
            spearman = stats.spearmanr(y_pred, y_true)
        return loss, spearman[0], spearman[1]
    
    def calc_metrics_classification(model, generator):
        loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,y in generator:
                outputs = model(x)  
                y_pred.append(torch.argmax(outputs, dim=1).int().item())
                loss += loss_fn(outputs, y).item()
                y_true.append(y.int().item())
            loss = loss / len(generator)
            mcc = matthews_corrcoef(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
        return loss, mcc, accuracy
        
    list_of_rows = []
    for epoch in range(1, epochs + 1):
        logger.info(f'----- starting epoch = {epoch} -----')
        epoch_loss = 0.0
        running_loss = 0.0
        # Training
        start_time = time.time()
        model.train()
        for idx, (x, y) in enumerate(train_generator):
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if idx % print_training_logs == print_training_logs - 1:
                end_time = time.time()
                logger.info('[%d, %5d] time: %.3f loss: %.3f' %
                      (epoch, idx + 1, end_time - start_time, running_loss / print_training_logs))
                running_loss = 0.0
                start_time = time.time()
        
        model.eval()
        if task_type == TASK_REGRESSION:
            val_loss, spearman_val_corr, spearman_val_p = calc_metrics_regression(model, valid_generator)
            test_loss, spearman_test_corr, spearman_test_p = calc_metrics_regression(model, test_generator)
            
            logger.info(f'epoch = {epoch}, val_loss = {val_loss}, spearman_val_corr = {spearman_val_corr}, spearman_val_p = {spearman_val_p}, test_loss = {test_loss}, spearman_test_corr = {spearman_test_corr}, spearman_test_p = {spearman_test_p}')
            list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
        
        elif task_type == TASK_CLASSIFICATION:
            val_loss, val_mcc, accuracy_val = calc_metrics_classification(model, valid_generator)
            test_loss, test_mcc, accuracy_test = calc_metrics_classification(model, test_generator)
            
            logger.info(f'epoch = {epoch}, val_loss = {val_loss}, val_mcc = {val_mcc}, test_loss = {test_loss}, test_mcc = {test_mcc}, accuracy_val = {accuracy_val}, accuracy_test = {accuracy_test}')
            list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'val_mcc': val_mcc, 'test_loss': test_loss, 'test_mcc': test_mcc, 'accuracy_val': accuracy_val,'accuracy_test': accuracy_test})
        
        torch.save(model.state_dict(), os.path.join(results_path, f"checkpoint_{epoch}.pt"))
    
    df_loss = pd.DataFrame(list_of_rows)
    df_loss.to_csv(os.path.join(results_path, f"results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_path, exist_ok=True)
    
    num_classes, train_dataset, valid_dataset, test_dataset = train_biological_tokenizer(args.data_path, args.task_type, args.tokenizer_type, args.vocab_size, args.results_path, args.max_length)
    
    model = BioBERTModel(args.hidden_size, args.layers_num, args.attention_heads_num, num_classes)
    model.to(device)
    logger.info(f'loaded model to device')
    logger.info(f'device is {device}')
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'num of paramters = {total_params}')
    
    g = torch.Generator()
    g.manual_seed(0)
    train_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8, generator=g)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)
    test_generator = torch.utils.data.DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)
    train_model(model, args.task_type, train_generator, valid_generator, test_generator, args.epochs, args.print_training_loss, args.results_path)
