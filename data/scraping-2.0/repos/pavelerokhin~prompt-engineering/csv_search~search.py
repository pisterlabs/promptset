import json
import os
import random

from csv_search import openai, davinci1, davinci05, davinci01

# prepare prompt, we inject a list of metadata into the prompt as context
# we inject headers of the corresponding csv files (with same name)
# we inject the first line of the corresponding csv files (with same name)


def get_file_header(file):
    with open(file, 'r') as f:
        return f.readline()


base = """Questi sono le informazioni che abbiamo sul dataset:
{data}
***
Fine dataset.

Devi rispondere con il numero del dataset o -1 se non lo sai.
Qual è numero di dataset relativo alla domanda:
"""

data_prompt = """
Dataset {i}
Le metadata sono:
{metadata}
Le categorie sono:
{csv_header}
"""

file_names = ["82_20230706_eg_incarichi",
              "82_20230801_an_cittadini",
              "82_20230801_at_atti_amministrativi",
              "4701_20230508_eg_concorsi"]


def prepare_prompt_context(file_names: list):
    data = ""

    for i, file_name in enumerate(file_names):
        file_name_csv = './output/' + file_name + '.csv'
        file_name_json = './output/' + file_name + '.json'

        # read json and make it a map
        with open(file_name_json, 'r') as f:
            j = json.loads(f.read())

        topics = j.get("metadata").get("Temi del dataset")
        geo = j.get("metadata").get("Copertura geografica")
        auth = j.get("metadata").get("Titolare")
        metadata = f"Tematiche: {topics}\nCopertura geografica: {geo}\nTitolare: {auth}"

        # read csv headers
        csv_header = get_file_header(file_name_csv)

        data += data_prompt.format(i=i, metadata=metadata, csv_header=csv_header)

    return base.format(data=data)


if __name__ == '__main__':
    alpha = 150  # 1/alpha of files will be used for the prompt

    part_of_files = []

    # number of files in dir ./output
    n = len(os.listdir('./output'))

    for i, file in enumerate(os.listdir('./output')):
        if i >= n / alpha:
            break

        full_name = file.split('.')
        name = full_name[0]
        ext = full_name[-1]
        if ext == 'csv':
            part_of_files.append(name)



    # make list of files random order
    random.shuffle(file_names)
    correct_response = file_names.index("4701_20230508_eg_concorsi")

    # prepare prompt
    prompt = prepare_prompt_context(file_names)
    print(prompt)

    # ask question
    question = "Ci sono ancora dei concorsi aperti per operaio mautentore di macchine operatrici complesse?"
    print(question)

    print("*" * 100)
    # get answer on openai
    print("Risposta corretta: ", correct_response)
    print("openai")
    answer = openai(prompt + question)
    print(answer)

    # get answer on davinci1
    print("Risposta corretta: ", correct_response)
    print("davinci1")
    answer = davinci1(prompt + question)
    print(answer)

    # get answer on davinci05
    print("Risposta corretta: ", correct_response)
    print("davinci05")
    answer = davinci05(prompt + question)
    print(answer)

    # get answer on davinci01
    print("Risposta corretta: ", correct_response)
    print("davinci01")
    answer = davinci01(prompt + question)
    print(answer)
