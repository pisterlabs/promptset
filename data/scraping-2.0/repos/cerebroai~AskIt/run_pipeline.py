from flask import Flask, jsonify
from flask import redirect, render_template, jsonify, request, make_response, abort, g, session, Response
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
import socket 
import logging
import random
import argparse
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import configparser
import torch
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from model import download_model_folder, load_model
from decoder import generate_response
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model
from interact import sample_sequence


app_flask = Flask(__name__)
CORS(app_flask, support_credentials=True)

UPLOAD_FOLDER = './uploads'
CLOSENESS = 0.01
app_flask.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
hostname = socket.gethostname()    
IPAddr = socket.gethostbyname(hostname)  

model_sentence_transformers = SentenceTransformer('bert-base-nli-mean-tokens')


# ConvAI part
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
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
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
# logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
print(personality)
l = chain(*personality)
custom_personality_text = ['I am student from India', 'i work in the field of Computer science', 'i like playing cricket', 'In my free time I like conducting talks for students to learn', 
                            'I am a huge fan of IPL', 'I like talking, its fun to talk!' ]
custom_personality = list(map(tokenizer.encode, custom_personality_text))
print(custom_personality)
logger.info("Selected personality: %s", tokenizer.decode(chain(*custom_personality)))


# Reddit dialogue bot part
config = configparser.ConfigParser(allow_no_value=True)
with open("chatbot.cfg") as f:
    config.read_file(f)
target_dir = download_model_folder(config)
model_reddit, tokenizer_reddit = load_model(target_dir, config)

def cosine_similarity(l1, l2):
    cosine = torch.mm(l1.unsqueeze(0), l2.unsqueeze(0).transpose(0, 1))
    n1, n2 = l1.norm(), l2.norm()
    value = cosine/(n1*n2)
    return value


@app_flask.route('/get_response', methods=['GET', 'POST'])
def get_response():
    print(request.json)
    score = request.json['score']
    history = request.json['history']
    # if score < 0.5:
    #     bot_message = generate_response(model_reddit, tokenizer_reddit, ' {} '.format(tokenizer_reddit.eos_token).join(history), config)
    # else:
    out_ids = sample_sequence(custom_personality, list(map(tokenizer.encode, history)), tokenizer, model, args)
    bot_message = tokenizer.decode(out_ids, skip_special_tokens=True)
    score += CLOSENESS 
    return jsonify(response = bot_message, score = score)


if __name__ == "__main__":
    app_flask.run(host='localhost', port=5000)

