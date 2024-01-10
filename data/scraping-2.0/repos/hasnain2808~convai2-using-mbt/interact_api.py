# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model


import flask
from flask import request, jsonify
from itertools import chain


app = flask.Flask(__name__)


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
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
        instance, _ = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
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



class Args:
    dataset_path = ''
    dataset_cache = './dataset_cache'
    model = 'gpt'
    model_checkpoint = 'runs/Apr05_01-29-21_ip-172-31-67-62_openai-gpt/'
    max_history = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 20
    min_length=1
    seed=42
    temperature = 0.7
    top_k = 0
    top_p = 0.9
    no_sample = "store_true"
args=Args()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

if args.model_checkpoint == "":
    args.model_checkpoint = download_pretrained_model()

random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

logger.info("Get pretrained model and tokenizer")
tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

personalities = ''
personality = ''
@app.route('/gen_personality', methods=['GET'])
def gen_personality():
    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = random.choice(personalities)
    pers = tokenizer.decode(chain(*personality))
    logger.info("Selected personality: %s", pers)
    return pers
history = []


books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]



@app.route('/', methods=['GET'])
def home():
    # parser = ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    # parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    # parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    # parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    # parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    # parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    # parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    # parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    # parser.add_argument("--seed", type=int, default=42, help="Seed")
    # parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    # parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    # parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    # args = parser.parse_args()
    global history
    if 'input_dialogue' in request.args:
        raw_text = str(request.args['input_dialogue'])
    print(raw_text)
    raw_text = ' '.join(raw_text.split('+'))
    # while True:
    # raw_text = input(">>> ")
    # while not raw_text:
    #     print('Prompt should not be empty!')
    #     raw_text = input(">>> ")
    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    print(tokenizer.decode(chain(*history), skip_special_tokens=True))
    print(out_text)

    return out_text

app.run(host='0.0.0.0', port=80)
# if __name__ == "__main__":
#     run()
