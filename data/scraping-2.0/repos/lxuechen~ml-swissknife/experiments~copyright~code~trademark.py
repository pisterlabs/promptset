import logging
import os
import time

import fire
import openai
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml_swissknife import utils

openai.api_key = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_cache_dir = "/u/scr/nlp/data/lxuechen-data/hfcache"

codegen_model_names = [
    f"Salesforce/codegen-{model_size}-{data}"
    # Things don't always work for 16B on 1 A100.
    # for model_size in ("350M", "2B", "6B", "16B")
    for model_size in ("350M", "2B", "6B",)
    for data in ("nl", "multi", "mono")
]

codex_model_names = [
    "openai/code-davinci-001",
    "openai/code-davinci-002",
    "openai/code-cushman-001",
]


def main(
    prompt="MODULE_AUTHOR(",  # Linux kernel source.
    load_in_8bit=True,
    top_p=1.,
    temperature=0.7,
    max_tokens=50,
    output_path=None,
    num_return_sequences=None,
    only_codex=False,
    only_codegen=False,
):
    record = dict()

    if only_codex:
        #  python trademark.py --output_path "extraction_codex.json"   --only_codex True
        model_names = codex_model_names
    elif only_codegen:
        #  python trademark.py --output_path "extraction_codegen.json"   --only_codegen True
        model_names = codegen_model_names
    else:
        model_names = codex_model_names + codegen_model_names

    for model_name in model_names:
        if 'openai' in model_name:
            this_num_return_sequences = 1000 if num_return_sequences is None else num_return_sequences
            rate_limit_micro_batch_size = 10  # Annoying rate limit on tokens.
            real_model_name = model_name.split('/')[1]

            outputs = []
            for _ in tqdm.tqdm(range(this_num_return_sequences // rate_limit_micro_batch_size)):

                while True:
                    try:
                        raw_outputs = openai.Completion.create(
                            model=real_model_name,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=rate_limit_micro_batch_size,
                        )
                        outputs.extend(
                            [prompt + choice["text"] for choice in raw_outputs["choices"]]
                        )
                        break
                    except openai.error.RateLimitError as e:
                        logging.warning('Hit request rate limit; retrying...')
                        time.sleep(10)  # Annoying rate limit on requests.

            record[model_name] = outputs
        else:
            this_num_return_sequences = 100 if num_return_sequences is None else num_return_sequences
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=hf_cache_dir,
                device_map='auto',
                # 8-bit inference example
                # https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing
                load_in_8bit=load_in_8bit,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            outputs = []
            for _ in tqdm.tqdm(range(this_num_return_sequences), desc=f"{model_name}"):  # micro batch size of 1.
                samples = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=input_ids.size(1) + max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = tokenizer.batch_decode(
                    samples,
                    truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
                )
                outputs.extend(output)
            record[model_name] = outputs

    if output_path is not None:
        utils.jdump(record, output_path)


if __name__ == "__main__":
    fire.Fire(main)
