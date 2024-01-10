import os
import sys
import json
import logging
import argparse
import unicodedata
from collections import defaultdict

import openai
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(funcName)s() - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def write_txt(file, text, write_log=True):
    if write_log:
        characters = len(text)
        logger.info(f"Writing {characters:,} characters to {file}")

    with open(file, "w", encoding="utf8") as f:
        f.write(text)

    if write_log:
        logger.info(f"Written")
    return


def read_json(file, is_jsonl=False, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        if is_jsonl:
            data = []
            for line in f:
                datum = json.loads(line)
                data.append(datum)
        else:
            data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


class Config:
    def __init__(self, config_file):
        data = read_json(config_file)

        self.task_list = data["task_list"]
        self.prompt_method_list = data["prompt_method_list"]
        self.language_list = data["language_list"]
        self.model = data["model"]
        self.news_dir = os.path.join(*data["news_dir"])
        self.prompt_dir = os.path.join(*data["prompt_dir"])
        self.output_dir = os.path.join(*data["output_dir"])
        return


def read_news_data(news_dir, language_list, write_log=False):
    title_list = ["Russo-Ukrainian_War", "2023_Hawaii_wildfires"]
    title_language_linelist = defaultdict(lambda: defaultdict(lambda: []))

    for title in title_list:
        for language in language_list:
            file = os.path.join(news_dir, title, f"wiki_{language}_intro.txt")

            with open(file, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = unicodedata.normalize("NFKC", line)
                    title_language_linelist[title][language].append(line)

            if write_log:
                lines = len(title_language_linelist[title][language])
                characters = sum(len(_) for _ in title_language_linelist[title][language])
                variable = f"[\"{title}\"][\"{language}\"]"
                logger.info(f"{variable} {lines:,} lines; {characters:,} characters")

    title_language_text = {
        title: {
            language: "\n".join(line_list)
            for language, line_list in language_to_linelist.items()
        }
        for title, language_to_linelist in title_language_linelist.items()
    }

    return title_language_linelist, title_language_text


def read_prompt(prompt_dir, task, method, language):
    file = os.path.join(prompt_dir, f"{task}__{method}__{language}.txt")
    if os.path.exists(file):
        with open(file, "r", encoding="utf8") as f:
            return f.read().strip()
    return ""


def get_specific_tokens_model(model, text_in, tokenizer, out_tokens=500):
    model_candidate_tokens = {
        "gpt-3.5-turbo": {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        },
        "gpt-4": {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }
    }

    in_token_list = tokenizer.encode(text_in)
    in_tokens = len(in_token_list)
    tokens = in_tokens + out_tokens

    for candidate, max_tokens in model_candidate_tokens.get(model, {}).items():
        if max_tokens >= tokens:
            break
    else:
        candidate = ""

    return in_tokens, candidate


def run_gpt(model, tokenizer, text_in, log_prefix):
    in_tokens, specific_tokens_model = get_specific_tokens_model(
        model, text_in, tokenizer, out_tokens=500,
    )
    if not specific_tokens_model:
        logger.info(f"[{log_prefix}] too many tokens, return empty result")
        return ""

    logger.info(f"[{log_prefix}]: prompting {specific_tokens_model} with {in_tokens} tokens...")
    completion = openai.ChatCompletion.create(
        model=specific_tokens_model,
        n=1,
        messages=[
            {"role": "user", "content": text_in},
        ]
    )
    logger.info(f"[{log_prefix}]: received result")
    text_out = completion.choices[0].message.content
    return text_out


def run_news(config):
    _title_language_linelist, title_language_text = read_news_data(config.news_dir, config.language_list)

    tokenizer = tiktoken.encoding_for_model(config.model)

    for task in config.task_list:
        for method in config.prompt_method_list:
            for language in config.language_list:
                log_prefix = f"{task}/{method}/{language}"
                prompt = read_prompt(config.prompt_dir, task, method, language)
                if not prompt:
                    logger.info(f"[{log_prefix}] no prompt, skip")
                    continue

                for title, language_to_text in title_language_text.items():
                    log_prefix = f"{task}/{method}/{language}/{title}"

                    article = language_to_text[language]
                    text_in = prompt.replace("INPUT_PLACEHOLDER", article)

                    text_out = run_gpt(config.model, tokenizer, text_in, log_prefix)

                    separation_line = "#" * 100
                    separation_line = f"\n\n{separation_line}\n# OUTPUT\n{separation_line}\n\n"
                    text_full = f"{text_in}{separation_line}{text_out}"

                    output_file = os.path.join(config.output_dir, f"{task}__{method}__{language}__{title}.txt")
                    write_txt(output_file, text_full)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="lab1_config.json")
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[arg.{key}] {value}")

    openai.api_key = input("OpenAI API Key: ")
    logger.info("OpenAI API has been set up")

    config = Config(arg.config_file)
    os.makedirs(config.output_dir, exist_ok=True)

    run_news(config)
    return


if __name__ == "__main__":
    main()
    sys.exit()
