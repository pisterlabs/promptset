import locale
import os.path
import sys
import openai
from decouple import config
import csv


openai.api_key = config('OPEN_API_KEY')
csv_file = os.path.join(os.pardir, os.path.join('simple_language', os.path.join('corpus', 'wikitext_simple.csv')))


def get_text_from_json(json_obj):
    return json_obj.get('choices')[0]['text']


def summarize_2nd_grader(text):
    response_en = openai.Completion.create(engine="text-davinci-001",
                                           prompt=f'Summarize this for a second-grade student:\n\n{text}',
                                           temperature=0.7,
                                           max_tokens=300,
                                           top_p=1.0,
                                           frequency_penalty=0.0,
                                           presence_penalty=0.0
                                           )

    return get_text_from_json(response_en)


def translate_to_de(text):
    response_de = openai.Completion.create(engine="text-davinci-001",
                                           prompt=f'Translate this into German:\n\n{text}',
                                           temperature=0.7,
                                           max_tokens=300,
                                           top_p=1.0,
                                           frequency_penalty=0.0,
                                           presence_penalty=0.0)

    return get_text_from_json(response_de)


def convert_text(text):
    response_en = summarize_2nd_grader(text)

    response_de = translate_to_de(response_en)
    return response_de


def read_text(arg):
    simple_text = ""
    if not os.path.isfile(arg):
        simple_text = convert_text(arg)
        # save_text = input("Write a file path if you want to write the simplified text to a file: ")
        # csv_file = os.path.join(os.pardir, os.path.join('simple_language', os.path.join('corpus', 'simple_language_openAI.csv')))
        # csv_file = 'C:\MA\NLP_Test\simple_language\corpus\simple_language_openAI.csv'
        arg = arg.replace('\n', '')
        simple_text = simple_text.replace('\n', '')
        if os.path.isfile(csv_file):
            with open(csv_file, 'a', encoding='utf-8') as f:
                s = [arg, simple_text]
                print(s)
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(s)
    return simple_text


def create_csv(inp_arr):

    for i in inp_arr:
        for j in i:
            read_text(j)
    print('Written to the csv File')

# if __name__ == '__main__':
#     argument = sys.argv[1]
#     # print(argument)
#     read_text(argument)

