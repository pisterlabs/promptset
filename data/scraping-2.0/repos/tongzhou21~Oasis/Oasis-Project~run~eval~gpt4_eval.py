import sys
sys.path.insert(0, sys.path[0]+"/../../")
import json
import openai
import argparse
from utils import sample_data_without_cache


def openai_auto_eval(api_key, prompt, truncation, list_text, save_path):
    openai.api_key = api_key
    list_res = []
    for idx, text in enumerate(list_text):
        chat_completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "user",
                 "content": prompt + '(if the following text is high quality output 1, if the following text is low quality, output 0)'
                 "'''" + ' '.join(text.split()[:truncation]) + "'''"}
            ]
        )
        list_res.append(chat_completion.choices[0].message.content)

    list_pos = [list_text[idx] for idx, res in enumerate(list_res) if '1' in res and '0' not in 'res']
    list_neg = [list_text[idx] for idx, res in enumerate(list_res) if '0' in res and '1' not in 'res']

    with open(save_path + '.gpt4.pos', 'w') as f_write:
        for text in list_pos:
            f_write.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')

    with open(save_path + '.gpt4.neg', 'w') as f_write:
        for text in list_neg:
            f_write.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    print('len(list_pos), len(list_neg)', len(list_pos), len(list_neg))

def openai_auto_eval_corpus(corpus_path, text_key, sample_count, api_key, prompt, truncation, save_path):
    list_dict_data = sample_data_without_cache(corpus_path, sample_count)

    list_text = [dict_data[text_key] for dict_data in list_dict_data]

    openai_auto_eval(api_key, prompt, truncation, list_text, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--text_key", type=str)
    parser.add_argument("--sample_count", type=int)

    parser.add_argument("--api_key", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--truncation", type=int)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    for arg in vars(args):
        print('{} = {}'.format(arg.lower(), getattr(args, arg)))
    print('')

    openai_auto_eval_corpus(
        args.corpus_path,
        args.text_key,
        args.sample_count,
        args.api_key,
        args.prompt,
        args.truncation,
        args.save_path,
    )