# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from itertools import chain
from pprint import pformat
import warnings
import sklearn

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model
from emote_text import EMOTION_TO_EMOTE, txtemote_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

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

    for i in range(args["max_length"]):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args["device"]).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args["device"]).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args["temperature"]
        logits = top_filtering(logits, top_k=args["top_k"], top_p=args["top_p"])
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1]
        if i < args["min_length"] and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def select_persona(args, tokenizer):
    dataset = get_dataset(tokenizer, args["dataset_path"], args["dataset_cache"])
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    if args["persona"]:
        ind_ps = [i for i,v in enumerate(personalities) if args["persona"] in tokenizer.decode(chain(*v))]
        if ind_ps:
            personality = personalities[ind_ps[0]]
        else:
            personality = random.choice(personalities)
    else:
        personality = random.choice(personalities)
    
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    return {"tokenized": personality, "string": tokenizer.decode(chain(*personality))}

def model_tokenizer(args):
    if torch.cuda.is_available():
        args["device"] = "cuda"
    else:
        args["device"] = "cpu"
    
    if args["model_checkpoint"] == "":
        if args["model"] == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args["model_checkpoint"] = download_pretrained_model()
	
    if args["seed"] != 0:
    	random.seed(args["seed"])
    	torch.random.manual_seed(args["seed"])
    	torch.cuda.manual_seed(args["seed"])

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args["model"] == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args["model_checkpoint"])
    model = model_class.from_pretrained(args["model_checkpoint"])
    model.to(args["device"])
    add_special_tokens_(model, tokenizer)
    
    logger.info("Get text to emote model")
    emote_clf = txtemote_model(args["txtemotion_dataset_path"])
    return model, emote_clf, tokenizer

def get_resp(input_text, model, emote_clf, tokenizer, personality, history, args):
    output = dict()
    
    output["emotion"] = "".join(emote_clf.predict([input_text]))
    output["emoji"] = EMOTION_TO_EMOTE[output["emotion"]]
        
    history.append(tokenizer.encode(input_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    while len(history) > 2*max_history:
        history.pop(0)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        
    output["text"] = out_text + " " + EMOTION_TO_EMOTE["".join(emote_clf.predict([out_text]))]
    return output
