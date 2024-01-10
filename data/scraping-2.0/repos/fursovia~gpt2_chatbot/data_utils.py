import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import OpenAIGPTTokenizer
from pytorch_pretrained_bert import GPT2Tokenizer

PAD_VALUE = 123  # '¿'


def vectorize(text, vocab, max_len=20, crop='post'):

    vectorized = []
    words = text.split()

    for i, word in enumerate(words):

        if word in vocab:
            vectorized.append(vocab[word])
        else:
            vectorized.append(vocab['<UNK>'])

    if crop == 'post':
        vectorized = vectorized[:max_len]
    elif crop == 'pre':
        vectorized = vectorized[-max_len:]
    else:
        raise NotImplementedError(f'{crop} -- no such cropping strategy')

    true_len = len(vectorized)

    if len(vectorized) < max_len:
        vectorized.extend([vocab['<PAD>']] * (max_len - len(words)))
        
    return vectorized, true_len


def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'GPT-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif tokenizer_name == 'GPT':
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    else:
        raise NotImplementedError(f'{tokenizer_name} -- No such tokenizer')

    return tokenizer


def load_vocab(txt_path):

    vocab = dict()
    with open(txt_path, 'r') as file:
        c = 0
        for line in file:
            vocab[line.strip()] = c
            c += 1
            
    return vocab


def crop_or_pad(x, pad_value=-1, max_len=40, crop='post'):
    if len(x) >= max_len:

        if crop == 'post':
            x = x[:max_len]
        elif crop == 'pre':
            x = x[-max_len:]
        else:
            raise NotImplementedError(f'{crop} -- no such cropping strategy')
    else:
        x.extend([pad_value] * (max_len - len(x)))

    return x


class CsvDataset(Dataset):
    # TODO: try keras tokenizer https://keras.io/preprocessing/text/

    def __init__(self, csv_path, vocab=None, max_len=20, tokenizer=None):
        self.data = pd.read_csv(csv_path)
        self.word2idx = vocab
        if self.word2idx is not None:
            self.idx2word = {i: word for word, i in self.word2idx.items()}

        self.context = self.data['context'].values
        self.answer = self.data['answer'].values

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, item):

        # TODO: change to https://torchtext.readthedocs.io/en/latest/examples.html
        if self.tokenizer is None:
            cont, true_cont_len = vectorize(self.context[item], self.word2idx, self.max_len, 'pre')
            ans, true_ans_len = vectorize(self.answer[item], self.word2idx, self.max_len, 'post')

            true_cont_len = torch.tensor(true_cont_len)
            true_ans_len = torch.tensor(true_ans_len)
        else:
            cont = self.tokenizer.encode(self.context[item])
            ans = self.tokenizer.encode(self.answer[item])

            true_cont_len = torch.tensor(len(cont)) if len(cont) <= self.max_len else torch.tensor(self.max_len)
            true_ans_len = torch.tensor(len(ans)) if len(ans) <= self.max_len else torch.tensor(self.max_len)

            cont = crop_or_pad(cont, pad_value=PAD_VALUE, max_len=self.max_len, crop='pre')
            ans = crop_or_pad(ans, pad_value=PAD_VALUE, max_len=self.max_len, crop='post')

        cont = torch.tensor(cont)
        ans = torch.tensor(ans)

        return (cont, true_cont_len), (ans, true_ans_len)

    def __len__(self):
        return self.data.shape[0]
