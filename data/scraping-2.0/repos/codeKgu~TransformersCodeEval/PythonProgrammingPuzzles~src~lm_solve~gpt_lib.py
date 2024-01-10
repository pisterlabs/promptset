import os
import json
import openai
import ezlog
import time
import datetime

import torch
import transformers
from tqdm import tqdm


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.cache")
_CACHE_ENCODING = "utf-8"
OPEN_AI_ENGINE_SUFFIX = os.environ.get('OPEN_AI_ENGINE_SUFFIX', '') # add extension such as -msft to engine names

# the cache file is just a list of (query params dictionary encoded as a string but without n, result list)
# multiple queries with the same params (except for n) are merged into a single big list
class Cache:
    def __init__(self, filename):
        print(filename)
        self.filename = filename
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
        
        # def helper(k):  # remove max_batch
        #     k2 = eval(k)
        #     del k2["max_batch"]
        #     return str(k2)

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

LOCAL_MODELS = ['gpt2', 'EleutherAI', 'EleutherAI/gpt-neo-2.7B'] + transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST


BATCH_SIZES = {
    "davinci": 32,
    "davinci-codex": 128, 
    "cushman-codex": 128,
}

for model in LOCAL_MODELS:
    BATCH_SIZES[model] = 20 # doesn't really matter since we are doing it sequentially anyways

CACHES = {cache: Cache(os.path.join(_CACHE_PATH, cache.replace('/','_') + ".cache")) for cache in BATCH_SIZES}

def query(prompt, n=10, max_tokens=150, temp=1.0, stop=None, notes=None, cache_only=False, verbose=True,
          max_retries=10, engine="cushman-codex", model=None, tokenizer=None):
    """Query gpt

    :param prompt: Up to 2048 tokens (about 3-4k chars)
    :param n: number of answers, None returns all cached answers
    :param max_tokens:
    :param temp: 0.9 seems to work well
    :param stop: string to stop at or '' if not to stop
    :param notes: notes you want to save or change in case you want to run the same query more than once!
    :return: list of answers and then the response items
    """
    global BATCH_SIZES
    global CACHES
    cur_cache = CACHES[engine]
    max_batch = BATCH_SIZES[engine]         
    engine += OPEN_AI_ENGINE_SUFFIX # add tail to engine name

    if temp == 0 and n > 1:
        ezlog.debug("Temp 0: no point in running more than one query")
        n = 1

    key = str(dict(prompt=prompt, max_tokens=max_tokens, temp=temp, stop=stop, rep=notes))

    cached = cur_cache.get(key)

    if n is None:
        return cached

    if len(cached) >= n:
        return cached[:n]

    assert not cache_only, f'Entry not found in cache with prompt "{json.dumps(prompt)}"'
    if verbose:
        print("/" * 100)
        print(f"Querying GPT {engine} with prompt:")
        print(prompt)
        s = stop and stop.replace('\n', '\\n')
        print(f"/// n={n} ({n - len(cached)} new) max_tokens={max_tokens} temp={temp} max_batch={max_batch} stop={s}")
        print("/" * 100)

    time0 = time.perf_counter()

    new = []
    n -= len(cached)

    while n > 0:
        m = min(n, max_batch)

        if engine in LOCAL_MODELS:
            res = get_disk_gpt_res(model, tokenizer, prompt, max_tokens, temp, m)
        else:
            res_openai = get_openai_gpt_res(prompt, engine, max_tokens, temp, m, stop, max_retries)
            res = [c["text"] for c in res_openai["choices"]]
        new += res
        n -= m

    return cur_cache.extend(key, new)

def index_default(line, char):
    """Returns the index of a character in a line, or the length of the string
    if the character does not appear.
    """
    try:
        retval = line.index(char)
    except ValueError:
        retval = len(line)
    return retval

def split_line(line, pattern1, pattern2):
    """Splits a line at either pattern1 or pattern2, depending on which appears 
    first in the line.
    """
    if index_default(line, pattern1) < index_default(line, pattern2):
        return line.split(pattern1)
    else:
        return line.split(pattern2)
    
def get_disk_gpt_res(model, tokenizer, prompt, max_tokens, temp, m):
    with torch.no_grad():
        input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(0).cuda()
        output_strs = []
        for i in tqdm(range(m)):
            try:
                if input_ids.shape[-1] > tokenizer.model_max_length:
                    print(f"Problem text {i} has token length {input_ids.shape[-1]} > {tokenizer.model_max_length} tokens, so cannot do generation")
                    output_str = ""
                else:
                    max_len = min(input_ids.shape[-1] + max_tokens, tokenizer.model_max_length)
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        early_stopping=True,
                        temperature=temp,  
                        max_length=max_len,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    output_str = tokenizer.decode(output_ids[0])

                # hacky way to get the output, assumes stops at next \nassert or \ndef
                last_line_prompt = prompt.split('\n')[-1]
                temp_output_str = output_str.split(last_line_prompt)[1]
                output_str = split_line(temp_output_str, "\ndef", "\nassert")[0]
                # print(output_str)
                # output_str = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
            except Exception as e:
                print("Unexpected exception in generating solution")
                print(e)
            
                    # Default to empty string on errors
                output_str = ""
            output_strs.append(output_str)
        return output_strs

def get_openai_gpt_res(prompt, engine, max_tokens, temp, m, stop, max_retries):
    try_number = 0
    
    while True:
        try:
            res = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                n=m,
                stop=stop or None
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

