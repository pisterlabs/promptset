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
import tensorflow as tf


import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


def top_filtering(logits, top_k=0.0, top_p=0.9, threshold=-float("Inf"), filter_value=-float("Inf"), beam=True):
    """Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
        top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
            whose total probability mass is greater than or equal to the threshold top_p.
            In practice, we select the highest probability tokens whose cumulative probability mass exceeds
            the threshold top_p.
        threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1

    if beam == True:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        indices_to_remove = sorted_indices[5:]
        logits[indices_to_remove] = filter_value

    else:
        if top_k > 0:
            # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
            top_k = min(top_k, logits.size(-1))
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

        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + 40,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )
        return outputs[0].tolist()
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        # token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        # logits = model(input_ids, token_type_ids=token_type_ids)

        # beam_output = model.generate(input_ids, max_length=40, num_beams=5)

        # return beam_output[0].tolist()

    #     if isinstance(logits, tuple):  # for gpt2 and maybe others
    #         logits = logits[0]

    #     logits = logits[0, -1, :] / args.temperature

    #     logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)

    #     probs = F.softmax(logits, dim=-1)

    #     prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
    #     if i < args.min_length and prev.item() in special_tokens_ids:
    #         while prev.item() in special_tokens_ids:
    #             if probs.max().item() == 1:
    #                 warnings.warn("Warning: model generating special token with probability 1.")
    #                 break  # avoid infinitely looping over special token
    #             prev = torch.multinomial(probs, num_samples=1)

    #     if prev.item() in special_tokens_ids:
    #         break
    #     current_output.append(prev.item())

    # return current_output


## ADDED CODE
def get_data():
    datapath = (
        "/raid/cs18resch11003/aditi/transfer-learning-conv-ai/ParlAI/data/ConvAI2/test_self_original_no_cands.txt"
    )
    data = open(datapath, "r")
    lines = data.readlines()

    personality = []
    persona = []
    for idx, line in enumerate(lines):
        if "your persona:" in line:
            line = line.strip()
            line = line.replace("your persona:", "")
            line = line[3:]
            persona.append(line)
        else:
            if len(persona) != 0:
                personality.append(persona)
                persona = []

    conversations = []
    conversation = []
    for idx, line in enumerate(lines):
        if "your persona:" not in line:
            line = line.strip()
            line = line[2:]
            conversation.append(line)
        else:
            if len(conversation) != 0:
                conversations.append(conversation)
                conversation = []
    conversations.append(conversation)

    prompts = []
    replies = []

    for conversation in conversations:
        temp_prompt = []
        temp_replies = []
        for turn in conversation:
            utterance1, utterance2 = turn.split("\t")
            utterance1 = utterance1.strip()
            utterance2 = utterance2.strip()

            temp_prompt.append(utterance1)
            temp_replies.append(utterance2)
        prompts.append(temp_prompt)
        replies.append(temp_replies)

    assert len(personality) == (len(prompts) + len(replies)) / 2

    return personality, prompts, replies


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3."
    )
    parser.add_argument(
        "--dataset_cache", type=str, default="./dataset_cache", help="Path or url of the dataset cache"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-gpt",
        help="Model type (openai-gpt or gpt2)",
        choices=["openai-gpt", "gpt2"],
    )  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument(
        "--max_history", type=int, default=2, help="Number of previous utterances to keep in history"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )

    parser.add_argument("--no_sample", action="store_true", help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument(
        "--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == "gpt2":
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()

    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (
        (GPT2Tokenizer, GPT2LMHeadModel) if args.model == "gpt2" else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    ####################

    # logger.info("Sample a personality")

    # dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    # personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]

    # personality = random.choice(personalities)
    # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    ## ADDED CODE

    personality, prompts, replies = get_data()

    tokens_personality = []

    for persona in personality:
        temp_p = []
        for p in persona:
            temp_p.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p)))
        tokens_personality.append(temp_p)

    history = []
    count = 0
    for idx, (tp, p, r) in enumerate(zip(tokens_personality, prompts, replies)):
        print(f"Personality: {personality[idx]}")
        for _p, _r in zip(p, r):
            print(f"Prompt: {_p}")
            print(f"Original: {_r}")

            history.append(tokenizer.encode(_p))
            with torch.no_grad():
                out_ids = sample_sequence(tp, history, tokenizer, model, args)
            history.append(out_ids)

            history = history[-(2 * args.max_history + 1) :]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(f"Generated: {out_text}\n")

            count += 1

            if count == 50:
                exit(0)

    # while True:
    #     raw_text = input(">>> ")
    #     while not raw_text:
    #         print("Prompt should not be empty!")
    #         raw_text = input(">>> ")

    #     history.append(tokenizer.encode(raw_text))

    #     with torch.no_grad():
    #         out_ids = sample_sequence(personality, history, tokenizer, model, args)

    #     history.append(out_ids)

    #     history = history[-(2 * args.max_history + 1) :]
    #     out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    #     print(out_text)


if __name__ == "__main__":
    run()
