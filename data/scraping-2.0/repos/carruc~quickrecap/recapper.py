import json
import time
import openai
from openai.error import RateLimitError


def _get_api_key():
    # Chat-GPT api key request using key stored in JSON
    with open("secrets.json") as f:
        secrets = json.load(f)
        api_key = secrets["api_key"]
    return api_key


def _splitter(transcription):
    with open(transcription, "r") as text_file:
        text = text_file.read()
        text_file.close()

    # Split in parts of 8k characters (8k tokens maximum w/ gpt4) to avoid RateLimitErrors
    n = 8000
    split_transcriptions = [(text[i:i + n]) for i in range(0, len(text), n)]
    print("Debug: Number of transcriptions = ", len(split_transcriptions))
    return split_transcriptions


def _get_recap(filename, part, subject):
    try:
        messages = [{"role": "system",
                     "content": "Sei un assistente di grande aiuto, mi dovrai aiutare a riassumere lezioni universitarie."},
                    {"role": "user",
                     "content": "Potresti riassumere il seguente testo, una trascrizione di una parte di lezione universitaria di "
                                + subject + ", schematizzandola, formattando in MarkDown il testo e aggiungendo collegamenti" +
                                "alle pagine di Wikipedia dei concetti e definizioni principali" + part}]
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=1.0
        )

        if chat_completion.choices[0].message is not None:
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(chat_completion.choices[0].message.content + '\n\n')
                f.close()
        else:
            print('Debug: Recap is empty.')

        # Add request time limit (10k tokens/min)
        time.sleep(60)
    except RateLimitError:
        print('Debug: RateLimitError.')
        return
    return


def _get_abstract(filename):
    # Summed up abstract request
    print('\t[' + filename + ']: Abstract request.')

    with open(filename, 'r') as f:
        summary = f.read()
        f.close()

    prompt = "Ti passo varie trascrizioni di una lezione universitaria, potresti unirle in modo coeso e preciso? Testo: " + summary

    try:
        messages = [{"role": "system",
                     "content": "Sei un assistente di grande aiuto, mi dovrai aiutare a riassumere lezioni universitarie."},
                    {"role": "user",
                     "content": prompt}]
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=1.0
        )

        text_file_name = filename.replace('_recap.txt', '_abstract.txt')
        with open(text_file_name, 'w') as f:
            f.write(chat_completion.choices[0].message.content)
            f.close()
    except RateLimitError:
        print('Debug: RateLimitError.')


def recap(transcription_list, subject):
    openai.api_key = _get_API_key()

    for transcription in transcription_list:
        print('\t[' + transcription + ']: Starting recap.')
        filename = transcription.replace('_transcription.txt', '_recap.txt')
        split_transcription = _splitter(transcription)
        for part in split_transcription:
            _get_recap(filename, part, subject)
        _get_abstract(filename)
    return
