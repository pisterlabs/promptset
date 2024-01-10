import os
import time

import openai
import transformers
import torch
from tqdm import tqdm

from . import ezlog

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.cache")
_CACHE_ENCODING = "utf-8"

LOCAL_MODELS = ['gpt2', 'EleutherAI', 'EleutherAI/gpt-neo-2.7B'] + transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST

# the cache file is just a list of (query params dictionary encoded as a string but without n, result list)
# multiple queries with the same params (except for n) are merged into a single big list
class Cache:
    def __init__(self, filename):
        self.filename = filename.replace('/', '_')
        self._cache = None

    def _load_cache(self):
        """for lazy loading"""
        assert self._cache is None, "gpt cache already loaded"

        if not os.path.exists(_CACHE_PATH):
            ezlog.warn("Creating cache path")
            os.makedirs(_CACHE_PATH)

        self._cache = {}

        if os.path.exists(self.filename):
            time0 = time.perf_counter()
            with open(self.filename, "r", encoding=_CACHE_ENCODING) as f:
                for k, v in [eval(line) for line in f.readlines()]:
                    if k not in self._cache:
                        self._cache[k] = v
                    else:
                        self._cache[k].extend(v)
            ezlog.info(f"Loaded cache `{self.filename}` in {time.perf_counter() - time0:.1f}s")
        else:
            ezlog.warn("No gpt cache yet")

    def defrag(self):
        if self._cache is None:
            self._load_cache()

        if self._cache:
            with open(self.filename, "w", encoding=_CACHE_ENCODING) as f:
                # f.write("\n".join([str((helper(k), v)) for k, v in self._cache.items()]+[""]))
                f.write("\n".join([str((k, v)) for k, v in self._cache.items()]+[""]))
            ezlog.info("Defragged cache")
        else:
            ezlog.warn("No cache to defrag")


    def get(self, item):
        if self._cache is None:
            self._load_cache()

        return self._cache.get(item, []).copy()  # no monkey business changing cache

    def extend(self, key, values):
        if self._cache is None:
            self._load_cache()

        v = self._cache.setdefault(key, [])
        v.extend(values)

        with open(self.filename, "a", encoding=_CACHE_ENCODING) as f:
            f.write(str((key, values)) + "\n")

        return v.copy()  # no monkey business changing cache


BATCH_SIZES = {
    "davinci": 32,
    "davinci-codex": 128, 
    "cushman-codex": 128,
}

for model in LOCAL_MODELS:
    BATCH_SIZES[model] = 20 # doesn't really matter since we are doing it sequentially anyways

CACHES = {cache: Cache(os.path.join(_CACHE_PATH, cache + ".cache")) for cache in BATCH_SIZES}


def load_tokenizer_and_model(args):
    if ('EleutherAI' in args.engine or '2700' in args.engine):
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif 'gpt' in args.engine: # Should handle GPT-2 and GPT-Neo
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.engine)
    elif args.engine in {'codebert'}:
        tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    else:
        raise NotImplementedError()
    
    if args.load:
        if 'EleutherAI' in args.engine:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.load)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Loaded model from {args.load}")
    else:
        if "EleutherAI" in args.engine:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.engine)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.engine)
    return tokenizer, model


def call_gpt_for_results(model, tokenizer, prompt_text, sample_sol, args, cached, 
                         cur_cache, key, max_batch):
    new = []
    n = args.num_samples
    n -= len(cached)
    while n > 0:
        m = min(n, max_batch)
        if args.engine in transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + ['EleutherAI', 'EleutherAI/gpt-neo-2.7B']:
            res = get_disk_gpt_res(model, tokenizer, prompt_text, args, m, sample_sol)
        else:
            res_openai = get_openai_gpt_res(prompt_text, m, args)
            res = [c["text"] for c in res_openai["choices"]] 
        new += res
        n -= m
    # Save the generated sol
    all_res = cur_cache.extend(key, new)
    return all_res
    
    
def get_openai_gpt_res(prompt, m, args, max_retries=10):
    try_number = 0
    prompt_temp = prompt.split('ANSWER:\n')[0] # remove the answer part of the prompt
    prompt_new = '"""' + prompt_temp + '"""' # make prompt a docstring for codex
    
    while True:
        try:
            res = openai.Completion.create(
                engine=args.engine,
                prompt=prompt_new,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                n=m,
                stop=None
            )
            break
        except (openai.error.RateLimitError, openai.error.APIError):
            if try_number == max_retries:
                print("Rate limit error: Giving up!")
                raise
            sleep_secs = 10 * (2 ** try_number)
            try_number += 1
            print(f"Rate limit error #{try_number}: Sleeping for {sleep_secs} seconds...")
            time.sleep(sleep_secs)
    return res

def get_disk_gpt_res(model, tokenizer, prompt, args, m, sample_sol):
     with torch.no_grad():
        input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(0).cuda()
        output_strs = []
        for i in tqdm(range(m)):
            try:
                if input_ids.shape[-1] > tokenizer.model_max_length:
                    print(f"Problem text has token length {input_ids.shape[-1]} > {tokenizer.model_max_length} tokens, so cannot do generation")
                    output_str = ""
                else:
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        early_stopping=True,
                        temperature=args.temperature,  
                        max_length=tokenizer.model_max_length,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    output_str = tokenizer.decode(output_ids[0])
                if args.peeking == 1.0:
                    output_str = sample_sol
                elif len(output_str):
                    output_str = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
            except Exception as e:
                if isinstance(e, UnboundLocalError) and str(e) == "local variable 'next_tokens' referenced before assignment":
                    # See https://github.com/huggingface/transformers/issues/5118
                    if args.debug:
                        print("Problem text was > 1024 tokens, so cannot do generation")
                        print(e)
                else:
                    print("Unexpected exception in generating solution")
                    print(e)
        
                # Default to empty string on errors
                output_str = ""
            output_strs.append(output_str)
        return output_strs