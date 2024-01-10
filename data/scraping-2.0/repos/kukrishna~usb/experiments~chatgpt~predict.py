import jsonlines
import openai
import argparse
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
import tiktoken

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class GPTInference():
    def __init__(self, model_name):
        openai.api_key = "YOUR_KEY_HERE"
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, prompt, max_tokens=None):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
                )
        return response['choices'][0]['message']['content']


if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--max-pred", type=int, required=False)
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name
    max_pred = args.max_pred

    if max_pred is None:
        max_pred = 999999999

    gpt = GPTInference(model_name=model_name)

    input_dps = list(jsonlines.open(input_file))

    if "gpt-3.5-turbo" in model_name:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    else:
        raise NotImplementedError

    _cnt = 0

    with jsonlines.open(output_file, "w") as w:
        for dp in tqdm(input_dps):
            inp_str = dp["input_string"]
            # https://github.com/openai/openai-python/issues/304
            # openai adds some tokens on their side. the error it threw suggests it's 8
            to_generate_len = 4096 - len(tokenizer.encode(inp_str)) - 8
            response = gpt.get_response(inp_str, max_tokens=to_generate_len)
            dp["prediction"] = response
            w.write(dp)

            _cnt+=1
            if _cnt==max_pred:
                break


