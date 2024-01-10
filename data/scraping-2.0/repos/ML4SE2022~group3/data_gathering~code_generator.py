import json
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import openai

from langs_util import LANG_EXTS

__CODE_DELIM = '```'
__SCRIPT_PARENT = Path(__file__).parent
__INP_DEFAULT = __SCRIPT_PARENT / 'resources' / 'conala-corpus-v1.1'
__DISALLOWED_DEFAULT = ['numpy', 'pandas', 'dataframe', 'flask', *LANG_EXTS]


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f'Maximum number of retries ({max_retries}) exceeded.'
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', nargs='+',
                        type=Path,
                        default=[
                            __INP_DEFAULT / 'conala-train.json',
                            __INP_DEFAULT / 'conala-test.json',
                        ],
                        help='File location(s) for the JSON input, which is expected to be an array on objects.')
    parser.add_argument('-k', '--key',
                        default='intent',
                        help='Key for the JSON objects forming the input whose value contains the prompt.')
    parser.add_argument('-l', '--langs', nargs='+',
                        default=['Java', 'Python'],
                        help=f'Languages for which to generate code for starting from the prompt; supported: {", ".join(LANG_EXTS.keys())}.')
    parser.add_argument('-d', '--disallowed', nargs='+',
                        default=__DISALLOWED_DEFAULT,
                        help='Terms for which promps containing them will be discarded.')
    parser.add_argument('-t', '--tokens',
                        type=int,
                        default=512,
                        help='Number of maximum tokens to be output by generating model; a higher number may incurr higher cost.')
    parser.add_argument('-m', '--model',
                        default='code-davinci-002',
                        help='OpenAI model to be used; ensure the used API key has access to it.')
    parser.add_argument('-o', '--output',
                        type=Path,
                        default=__SCRIPT_PARENT.parent / 'data',
                        help='Output directory.')

    args = parser.parse_args()
    langs: list[str] = [lang for lang in args.langs if lang in LANG_EXTS]

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise RuntimeError('Environment variable "OPENAI_API_KEY" is not set.')
    openai.api_key = openai_api_key

    disallowed_terms: list[str] = [d.casefold() for d in args.disallowed]
    start_time = time.time()

    for in_path in args.input:
        ds = json.loads(in_path.read_text(encoding='utf-8'))
        prompts: list[str] = list(set(e[args.key] for e in ds))
        prompts = [
            prompt for prompt in prompts
            if not any(x in prompt.casefold() for x in disallowed_terms)
        ]
        ds_path = args.output / f'{in_path.stem}_{args.model}_{int(start_time)}'
        prompts_path = ds_path / 'prompts'
        prompts_path.mkdir(parents=True, exist_ok=True)

        for lang in langs:
            lang_path = ds_path / lang
            lang_path.mkdir(exist_ok=True)

        for i, prompt in enumerate(prompts):
            lang_to_res: dict[str, str] = {}

            for lang in langs:
                response = completion_with_backoff(
                    model=args.model,
                    prompt=f'### {prompt}\n### {lang}\n',
                    temperature=0,
                    max_tokens=args.tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=['###']
                )
                stripped_text = response.choices[0].text.strip()
                lines: list[str] = stripped_text.splitlines()
                if lines and __CODE_DELIM in lines[0]:
                    del lines[0]
                if lines and __CODE_DELIM in lines[-1]:
                    del lines[-1]
                lines = [line.removeprefix('>>> ').rstrip() for line in lines]
                if len(lines) < 2:
                    break
                # lines.append('')
                lang_to_res[lang] = "\n".join(lines)

            if len(lang_to_res) != len(langs):
                continue
            prompt_path = prompts_path / f'{i}.txt'
            prompt_path.write_text(prompt, encoding='utf-8', newline='\n')

            for lang, res in lang_to_res.items():
                res_path = ds_path / lang / f'{i}.{LANG_EXTS[lang]}'
                res_path.write_text(res, encoding='utf-8', newline='\n')


if __name__ == "__main__":
    main()
