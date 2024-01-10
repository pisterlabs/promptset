import os
import random
import json

import nltk
import torch
# from apex import amp
from tqdm import tqdm, trange
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam

SPECIAL_TOKENS = ["<long>", "<quotes>", "<others>", "<cyberpunk>", "<text>", "<meta1>", "<meta2>","<pad>"]
LR = 6.25e-5
MAX_LEN = 500
BATCH_SIZE = 13

OUTPUT_DIR = "/home/altsoph/current"
random.seed(0xDEADFEED)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))
model.to(device)
optimizer = OpenAIAdam(model.parameters(), 
                       lr=LR,
                       warmup=0.002,
                       max_grad_norm=1,
                       weight_decay=0.01)

TAG_TEXT, TAG_META1, TAG_META2, TAG_PAD = tokenizer.convert_tokens_to_ids(("<text>", "<meta1>", "<meta2>", "<pad>"))

def pad(x, padding, padding_length):
    return x + [padding] * (padding_length - len(x))

dataset = []
for line in open('gpt1_trainset_tokens.tsv'):
    chunks = line.strip().split('\t')
    tokens = list(map(int,chunks[2].split(',')))
    if len(tokens)<8: continue
    segments = [TAG_META1, TAG_META2] + [TAG_TEXT for _ in tokens[2:]]
    positions = list(range(len(tokens)))
    lm_targets = [-1, -1, -1] + tokens[3:]
    dataset.append( (len(tokens), tokens, segments, positions, lm_targets) )

model.train()

for epoch in range(10):
    exp_average_loss = None
    nb_tr_steps = 0
    tr_loss = 0

    dataset = list(sorted(dataset,key=lambda x:random.random()))

    tqdm_bar = tqdm(range(0,len(dataset),BATCH_SIZE), desc="Training", mininterval=6.0)
    for batch_num,batch_start in enumerate(tqdm_bar):

        batch_raw = dataset[batch_start:batch_start+BATCH_SIZE]
        pad_size = max(map(lambda x:x[0],batch_raw))

        input_words = []
        input_segments = []
        input_targets = []

        for _,words,segments,_,targets in batch_raw:
            input_words.append( pad(words,TAG_PAD,pad_size) )
            input_segments.append( pad(segments,TAG_PAD,pad_size) )
            input_targets.append( pad(targets,-1,pad_size) )

        input_ids = torch.tensor(input_words, dtype=torch.long)
        token_type_ids = torch.tensor(input_segments, dtype=torch.long)
        lm_labels = torch.tensor(input_targets, dtype=torch.long)

        loss = model(input_ids.to(device), lm_labels=lm_labels.to(device), token_type_ids=token_type_ids.to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tr_loss += loss.item()
        exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
        nb_tr_steps += 1
        tqdm_bar.desc = "Epoch {:02}, batch {:05}/{:05}. Training loss: {:.2e} lr: {:.2e}".format(epoch, batch_num, len(dataset)//BATCH_SIZE, exp_average_loss, optimizer.get_lr()[0])

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    model_to_save.config.to_json_file(os.path.join(OUTPUT_DIR, "config.json"))
    tokenizer.save_vocabulary(OUTPUT_DIR)
