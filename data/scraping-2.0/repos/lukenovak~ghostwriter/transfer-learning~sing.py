import random
from argparse import ArgumentParser
import itertools
import warnings
import pdb
import torch

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
import utils

def filter_tokens(filter_data, filter1=0., filter2=0.9, t=-float('Inf'), f=-float('Inf')):
    filter1 = min(filter1, filter_data.size(-1))
    if filter1 > 0:
        bad_idx = filter_data < torch.topk(filter_data, filter1)[0][..., -1, None]
        filter_data[bad_idx] = f
    if filter2 > 0.0:
        sorted_logits, sorted_indices = torch.sort(filter_data, descending=True)
        prob_sums = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        s_bad_idx = prob_sums > filter2
        s_bad_idx[..., 1:] = s_bad_idx[..., :-1].clone()
        s_bad_idx[..., 0] = 0
        bad_idx = sorted_indices[s_bad_idx]
        filter_data[bad_idx] = f
    bad_idx = filter_data < t
    filter_data[bad_idx] = f

    return filter_data


def sample_sequence(feature, background, tokenizer, model, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(20): # 20 tokens max output per line
        accumulator = build_input_from_segments(feature, background, current_output, tokenizer, with_eos=False)
        inlab = torch.tensor(accumulator["input_ids"], device="cpu").unsqueeze(0)
        toklab = torch.tensor(accumulator["token_type_ids"], device="cpu").unsqueeze(0)
        m = model(inlab, token_type_ids=toklab)
        if isinstance(m, tuple):
            m = m[0]
        m = m[0, -1, :] / 0.7 # temperature value
        m = filter_tokens(m, filter1=0, filter2=0.9) # 0 means no filtering 
        probs = torch.nn.functional.softmax(m, dim=-1)
        back = torch.multinomial(probs, 1)
        if i < 1 and back.item() in special_tokens_ids:
            while back.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    break
                back = torch.multinomial(probs, num_samples=1)
        if back.item() in special_tokens_ids:
            break
        current_output.append(back.item())
    return current_output

def run():
    pretrained_model = utils.download_pretrained_model()
    tokenizer_class, model_class = (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    model = model_class.from_pretrained(pretrained_model)
    model.to("cpu")
    add_special_tokens_(model, tokenizer)
    dataset = utils.get_dataset(tokenizer, "./dataset_cache")
    features = [dialog["feature"] for dataset in dataset.values() for dialog in dataset]
    feature = random.choice(features)
    print("Examples of selected feature:\n", tokenizer.decode(itertools.chain(*feature)))
    background = [tokenizer.encode("tell me about yourself")]
    generated_lyrics = []
    hist_size = 2
    for _ in range(5): # how many lines of lyrics to generate - time grows exponentially with this value
        with torch.no_grad():
            out_ids = sample_sequence(feature, background, tokenizer, model)
        background.append(out_ids)
        background.append(random.choice(background))
        background = background[-5:] # size of history to retain (needs to be odd number since we're using two headed model)
        this_line = tokenizer.decode(out_ids, skip_special_tokens=True)
        generated_lyrics.append(this_line)
    print("\nGenerated lyrics:")
    print("\n".join(generated_lyrics))


if __name__ == "__main__":
    run()
