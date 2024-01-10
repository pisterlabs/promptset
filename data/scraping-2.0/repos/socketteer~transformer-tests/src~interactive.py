from masks import *
import openai
from char_mode import char_tokenization, char_tokenization_2


def get_completion(prompt, engine='cushman-alpha', max_tokens=100, stop=None, temperature=0.8, n=1, masks=None):
    if masks is None:
        mask = {}
    else:
        mask = {}
        for m in masks:
            mask.update(m)
    print(f'ENGINE: {engine}')
    print(f'TEMPERATURE: {temperature}\n')
    print(f'PROMPT: \"{prompt}\"\n')
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        echo=False,
        top_p=1,
        n=n,
        stop=stop,
        timeout=15,
        logit_bias=mask)

    for i, choice in enumerate(response.choices):
        response_text = choice['text']
        print(f'RESPONSE {i}: \"{response_text}\"')


def prompt_gpt(engine='cushman-alpha', max_tokens=100, stop=None, temperature=0.8, n=1, masks=None):
    prompt = input(">")
    get_completion(prompt, engine, max_tokens, stop, temperature, n, masks)


def prompt_gpt_char(engine='cushman-alpha', max_tokens=100, stop=None, temperature=0.8, n=3, masks=None, output_char_mode=False):
    if output_char_mode:
        masks = [single_char_mask]
    prompt = char_tokenization_2(input(">"))
    get_completion(prompt, engine, max_tokens, stop, temperature, n, masks)


def single_char_output(engine='cushman-alpha', max_tokens=200, stop=None, temperature=0.6, n=5):
    mask = single_char_mask
    prompt_gpt(engine, max_tokens, stop, temperature, n, [mask])


def main():
    #single_char_output()
    #prompt_gpt()
    prompt_gpt_char(output_char_mode=True)


if __name__ == "__main__":
    main()