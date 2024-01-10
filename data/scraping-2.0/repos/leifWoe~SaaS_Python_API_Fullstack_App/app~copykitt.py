import os
import re
import openai
import argparse


MAX_INPUT_LENGTH = 32


def main():
    # parser defnition
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '--i', type=str, required=True)
    args = parser.parse_args()
    user_input = args.input

    if validating_prompt_length(user_input):
        snippet_result = generate_snippet_openai_api_call(user_input)
        keyword_result = generate_keywords_openai_api_call(user_input)
    else:
        raise ValueError(
            f'Input to long. Must be under {MAX_INPUT_LENGTH}'
        )
    return snippet_result, keyword_result


def validating_prompt_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_snippet_openai_api_call(user_input: str) -> str:
    # auth at openAI
    openai.organization = 'org-i19EoyPkRCwwIyJUGPDcqHTg'
    openai.api_key = ('')
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.Model.list()
    # API Call
    prompt = f"Generate only one upbeat branding snippet for {user_input}"
    response = openai.Completion.create(
        engine='text-davinci-003', prompt=prompt, max_tokens=32)
    # extract for first text only
    text_response = response["choices"][0]["text"].strip()
    print(f'{prompt}')
    print(text_response)
    return text_response


def generate_keywords_openai_api_call(user_input: str) -> 'list[str]':
    # auth at openAI
    openai.organization = 'org-i19EoyPkRCwwIyJUGPDcqHTg'
    openai.api_key = ('')
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.Model.list()
    # API Call
    prompt = f"Generate related upbeat branding keywords for {user_input}"
    response = openai.Completion.create(
        engine='text-davinci-003', prompt=prompt, max_tokens=32)
    # extract for first text only
    keywords_response = response["choices"][0]["text"].strip()
    # print(keywords_response)
    # strip most common chat-gpt list types and put in a list
    # else start again  # TODO make better
    if keywords_response[0] == '-':
        regex_pattern_keywords_list = r'-([^\n]+)'
    elif keywords_response[0] == '•':
        regex_pattern_keywords_list = r'• (.+)'
    elif keywords_response[0] == '1':
        regex_pattern_keywords_list = r'\d+\.\s*(.+)'
    else:
        main()
    keywords_response = re.findall(
        regex_pattern_keywords_list, keywords_response)

    print(f'{prompt}')
    print(keywords_response)
    return keywords_response


if __name__ == "__main__":
    main()
