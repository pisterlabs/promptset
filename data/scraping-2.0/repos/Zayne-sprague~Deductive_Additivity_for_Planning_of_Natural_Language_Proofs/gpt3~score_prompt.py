import openai
import os
import json
import sys
import argparse
from pathlib import Path

from gpt3.utils import gpt3_common_args, gpt3_completion_args, get_gpt3_api_key, set_gpt3_api_key, PROMPTS_FOLDER, OUTPUTS_FOLDER

GPT_3_FOLDER = Path(__file__).parent.resolve()


# TODO - Intentionally not hooked into an automatic CSV printout to avoid people from spamming the heck out of GPT3
#       (easy to add if this becomes cumbersome). You can output json files per run though!

def score(
        prompt,
        context: str = '',
        engine: str = 'davinci',
        top_p: float = 0.9,
):

    full_prompt = f'{context}{prompt}'
    response = openai.Completion.create(
        engine=engine,
        prompt=full_prompt,
        max_tokens=0,
        logprobs=0,
        top_p=top_p,
        echo=True,
        n=1
    )

    choice = response['choices'][0]

    log_probabilities = choice['logprobs']['token_logprobs']

    offset = 0 if context is None else len(context)
    token_offsets = choice['logprobs']['text_offset']

    prompt_starting_idx = 0
    for idx, token_offset in enumerate(token_offsets):
        if token_offset > offset:
            break
        prompt_starting_idx = idx

    prompt_log_probs = log_probabilities[prompt_starting_idx:]
    prompt_log_probs = [x for x in prompt_log_probs if x is not None]

    score = sum(prompt_log_probs) / len(prompt_log_probs)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    gpt3_common_args(parser)

    parser.add_argument('--prompt', '-p', type=str,
                        help='Prompt to pass into the completion endpoint.')

    parser.add_argument('--prompt_file_name', '-pfn', type=str,
                        help='Name of the file in the prompts directory.')

    parser.add_argument('--context', '-c', type=str,
                        help='Context to pass into the completion endpoint. This will be pre-pended to the prompt'
                             'and will NOT be scored')

    parser.add_argument('--json_input_file', '-jif', type=str,
                        help='A json file that includes an array of json objects with each object having a '
                             'context property and a prompt property, the prompt will be scored.')

    parser.add_argument('--json_output_file', '-jof', type=str,
                        help='A json file that will output the scores of the model, will use the same name as the'
                             'json input file if specified (or the prompt file name if specified) as a default'
                             'without either, no output will be generated.')

    args = parser.parse_args()

    api_key = args.api_key
    api_key_file_path = args.api_key_file_path
    if not api_key and api_key_file_path:
        api_key = get_gpt3_api_key(Path(api_key_file_path))

    if not api_key:
        raise Exception("Specify an api key via --api_key or an api key file path --api_key_file_path=./api_key.txt")

    set_gpt3_api_key(api_key)

    silent = args.silent

    engine = args.engine
    top_p = args.top_p

    context = args.context
    prompt = args.prompt
    prompt_file_name = args.prompt_file_name
    json_input_file = args.json_input_file
    json_output_file = args.json_output_file

    prompts_to_score = []

    if json_input_file:
        with (PROMPTS_FOLDER / f'{json_input_file}.json').open('r') as f:
            prompts_to_score = json.load(f)
    else:
        if not prompt and prompt_file_name:
            with (PROMPTS_FOLDER / f'{prompt_file_name}.txt').open('r') as f:
                prompt = f.read().strip()

        prompts_to_score = [{'prompt': prompt, 'context': context}]

    if len(prompts_to_score) == 0:
        raise Exception("Specify a prompt via --prompt='example prompt' or a prompt file name in the prompts folder."
                        "-pfn test or a json input file (i.e. -jif 'test') with content in the form of"
                        " [{context: 'prepended to the prompt but not scored.', prompt: 'this will be scored'}]")

    outputs = []
    for idx, prompt_to_score in enumerate(prompts_to_score):
        prompt = prompt_to_score.get('prompt')
        assert prompt is not None, 'Please specify a correct prompt (none is not a correct prompt)'

        context = prompt_to_score.get('context', '')

        if not isinstance(prompt, list) and not isinstance(prompt, tuple):
            prompt = [prompt]

        if not isinstance(context, list) and not isinstance(context, tuple):
            context = [context]


        # Really the format of the json file can be
        # {context: ['abc put this before the prompt', 'another context to try'],
        # prompt: ['prompt 1 with the context', 'prompt 2 with the same context', 'prompt 3 with the...', ...]}
        # just to make things easier and compact.
        for c in context:
            all_scores = []

            if not silent:
                print(f"\n\nContext: {c}")

            curr_prompt = []
            for p in prompt:
                if isinstance(p, list) or isinstance(p, tuple):
                    curr_prompt.append(p)
                    all_scores.append(curr_prompt)
                    curr_prompt = []
                    continue
                elif len(curr_prompt) > 0 and len(curr_prompt) < 3:
                    curr_prompt.append([])
                    all_scores.append(curr_prompt)
                    curr_prompt = []

                scoring = score(
                    p,
                    context=c,
                    engine=engine,
                    top_p=top_p,
                )

                curr_prompt = [scoring, p]

            if len(curr_prompt) < 3 and len(curr_prompt) > 0:
                curr_prompt.append([])
            if len(curr_prompt) > 0:
                all_scores.append(curr_prompt)

            sorted_scores = list(sorted(all_scores, key=lambda x: x[0], reverse=True))
            for (s, p, t) in sorted_scores:
                if not silent:
                    print(f"Prompt ({s:4f}): {p} | tags {'; '.join(t)}")

            outputs.append({'context': c, 'scores': [{'prompt': p, 'score': s, 'tags': t} for (s, p, t) in sorted_scores]})

    out_file = None
    if json_output_file:
        out_file = Path(OUTPUTS_FOLDER / json_output_file)
    elif json_input_file:
        out_file = Path(OUTPUTS_FOLDER / (json_input_file + '_out.json'))
    elif prompt_file_name:
        out_file = Path(OUTPUTS_FOLDER / (prompt_file_name + "_out.json"))

    if out_file:
        json.dump(outputs, out_file.open('w'))
