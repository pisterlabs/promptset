import torch
import openai
import numpy as np
from multiprocessing.pool import ThreadPool

# Get the log likelihood of each text under the base_model
def get_ll(args, config, text):
    DEVICE = args.DEVICE
    base_model = config["base_model"]
    base_tokenizer = config["base_tokenizer"]

    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(args, config, texts):
    GPT2_TOKENIZER = config["GPT2_TOKENIZER"]

    if not args.openai_model:
        return [get_ll(args, config, text) for text in texts]
    else:
        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        config["API_TOKEN_COUNTER"] += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)