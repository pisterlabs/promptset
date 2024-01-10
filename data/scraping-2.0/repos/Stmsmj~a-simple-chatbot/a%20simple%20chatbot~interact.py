# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import ttkbootstrap as ttk
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import keras
import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import random
import os
import re
import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids).logits
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

# def run():
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

if args.model_checkpoint == "":
    if args.model == 'gpt2':
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()


if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

logger.info("Sample a personality")
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
personality = random.choice(personalities)
logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

history = []
def process(input_sentence):
    global history,personality,tokenizer,model,args
    raw_text = input_sentence
    if not raw_text:
        return 'Prompt should not be empty!'
    
    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text

# if __name__ == "__main__":
#     run()



#!/usr/bin/env python3


# this is for jokes and etc
# jokes separator
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path,'jokes\\jokes.txt'),encoding='utf-8') as jokes_file:
    jokes=jokes_file.read()
changed_str=re.sub(pattern=r'([0-9]+)\.',repl='',string=jokes)
final_jokes=changed_str.split('\n\n')

# fun facts separator
with open(os.path.join(dir_path,'jokes\\fun facts.txt'),encoding='utf-8') as fun_fact_file:
    fun_facts=fun_fact_file.read()
final_fun_facts=fun_facts.split('\n\n')

# quotes separator
with open(os.path.join(dir_path,'jokes\\quotes.txt'),encoding='utf-8') as quotes_file:
    quotes=quotes_file.read()
final_quotes=quotes.split('\n')
for i in range(len(final_quotes)):
    final_quotes[i]=final_quotes[i].lstrip()

pic_model=keras.models.load_model(os.path.join(dir_path,'model.h5'))
pic_dir=None
categories=os.listdir(os.path.join(dir_path,'train'))
sentence_pre=['its a','i dont know but if im right its a ','maybe its a ','i guess its a ','i will go with ','its gonna be a ']

def separating(sentence):
    multipe=1
    final_sentence=''
    separated_sentence=sentence.split(' ')
    for sen in separated_sentence:
        final_sentence=final_sentence+' '+sen
        if len(final_sentence)>multipe*45:
            final_sentence=final_sentence[:-len(sen)-1]
            final_sentence=final_sentence+'\n'+sen
            multipe+=1
    return final_sentence

    

def open_pic():
    pic=filedialog.askopenfile()
    pic_dir=pic.name
    image=cv2.imread(pic_dir)/255
    resized = [cv2.resize(image, (300, 300))]
    prediction=pic_model.predict(np.array(resized))
    pred_str=categories[np.argmax(prediction)]
    whole_sentence='bot: '+random.choice(sentence_pre)+pred_str
    msg_list.insert(tkinter.END, whole_sentence)

def input_processing():
    sentence=entry_field.get()
    entry_field.delete(0, END)
    if sentence=='tell me a joke':
        msg_list.insert(tkinter.END, 'you: '+sentence)
        output=random.choice(final_jokes)
        msg_list.insert(tkinter.END, 'bot: '+output)
    elif sentence=='tell me a fun fact':
        msg_list.insert(tkinter.END, 'you: '+sentence)
        output=random.choice(final_fun_facts)
        msg_list.insert(tkinter.END, 'bot: '+output)
    elif sentence=='tell me a quote':
        msg_list.insert(tkinter.END, 'you: '+sentence)
        output=random.choice(final_quotes)
        msg_list.insert(tkinter.END, 'bot: '+output)
    else:
        msg_list.insert(tkinter.END, 'you: '+sentence)
        output=process(sentence)
        msg_list.insert(tkinter.END, 'bot: '+output)

def changing_personality():
    global personality
    personality = random.choice(personalities)
    text_area.config(state='normal')
    text_area.delete('1.0',END)
    msg_list.delete(0,END)
    text_area.insert(END,separating(tokenizer.decode(chain(*personality))))
    text_area.config(state='disabled')

is_off=True
def dark_mode_toggle():
    global is_off
    if is_off==True:
        style.theme_use(themename='darkly')
        is_off=False
    elif is_off==False:
        style.theme_use(themename='morph')
        is_off=True



top = tkinter.Tk()
style=ttk.Style(theme='morph')
top.title("simple chatbot")
top.resizable(False,False)

messages_frame = ttk.Frame(top)
messages_frame.grid(row=0,column=0)
person_frame=ttk.Frame(top)
person_frame.grid(row=0,column=1,sticky=N)
msg_bar=ttk.Frame(top)
msg_bar.grid(row=1,column=0)
empty=ttk.Frame(top)
empty.grid(row=1,column=1,sticky=tkinter.E)


my_msg = ttk.StringVar()  # For the messages to be sent.
my_msg.set("")
scrollbar = ttk.Scrollbar(messages_frame,orient='vertical')  # To see through previous messages.
scrollbar2 = ttk.Scrollbar(messages_frame,orient='horizontal')
dark_mode=ttk.Checkbutton(bootstyle='dark-round-toggle',master=empty,command=dark_mode_toggle)
dark_mode_label=ttk.Label(master=empty,text='dark mode')
msg_list = tkinter.Listbox(messages_frame, height=30, width=100, yscrollcommand=scrollbar.set)
msg_list.bindtags((msg_list, top, "all"))
entry_field = ttk.Entry(msg_bar, textvariable=my_msg,width=92)
send_button = ttk.Button(msg_bar, text="Send", command=input_processing)
entry_field.bind("<Return>", lambda event: input_processing())
text_area=ttk.Text(person_frame,height=10,width=50)  # text area for showing personality
change_personality=ttk.Button(person_frame,text='change personality',command=changing_personality)
load_img=ttk.Button(person_frame,text='load image',command=open_pic)

scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
scrollbar2.pack(side=tkinter.BOTTOM, fill=tkinter.X)

msg_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
msg_list.pack(side=tkinter.TOP,anchor='w')

msg_list.config(yscrollcommand=scrollbar.set,xscrollcommand=scrollbar2.set)
scrollbar.config(command = msg_list.yview)
scrollbar2.config(command = msg_list.xview)

entry_field.pack(side=tkinter.LEFT)
send_button.pack(side=tkinter.LEFT)

text_area.pack(side=tkinter.TOP)
load_img.pack(side=tkinter.BOTTOM)
change_personality.pack(side=tkinter.BOTTOM)

dark_mode.grid(row=0,column=1)
dark_mode_label.grid(row=0,column=0)

text_area.insert(END,separating(tokenizer.decode(chain(*personality))))
text_area.config(state='disabled')


top.protocol("WM_DELETE_WINDOW", top.destroy)
tkinter.mainloop()  # for start of GUI  Interface
