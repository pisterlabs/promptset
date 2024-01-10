""" analiza biogramów PSB - relacje rodzinne - gpt-3.5-turbo
(ang. prompt, skrócone biogramy, tylko podstawowe pokrewieństwo: matka, ojciec,
córka, syn, siostra, brat)
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
#import spacy
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)


# tryb pracy, jeżeli True używa API OpenAI, jeżeli False  to tylko test
USE_API = True

# maksymalna wielkość odpowiedzi
OUTPUT_TOKENS = 500
# wielkość angielskiego promptu to ok 564 tokenow, model gpt-3.5-turbo obsługuje do 4096 tokenów
# maksymalna liczba tokenów w treści biogramu
MAX_TOKENS = 4096 - OUTPUT_TOKENS - 564 # 3032

# ceny gpt-3.5-turbo w dolarach
INPUT_PRICE_GPT =  0.0015
OUTPUT_PRICE_GPT = 0.002

# api key
env_path = Path(".") / ".env"
# api key
#env_path = Path(".") / ".env_ihpan"

load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# spacy do podziału tekstu na zdania
#nlp = spacy.load('pl_core_news_lg')


def count_tokens(text:str, model:str = "gpt-3.5-turbo") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    return openai.ChatCompletion.create(**kwargs)


def get_answer(prompt:str='', text:str='', model:str='gpt-3.5-turbo') -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''
    prompt_tokens = completion_tokens = 0

    try:
        response = get_answer_with_backoff(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Jesteś pomocnym asystentem, specjalistą w dziedzinie historii, genealogii, życiorysów znanych postaci."},
                        {"role": "user", "content": f"{prompt}\n{text}"}
                    ],
                    temperature=0.0,
                    top_p = 1.0,
                    max_tokens=OUTPUT_TOKENS)

        result = response['choices'][0]['message']['content']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']

    except Exception as request_error:
        print(request_error)
        sys.exit(1)

    return result, prompt_tokens, completion_tokens


def format_result(text: str) -> tuple:
    """ poprawianie i formatowanie wyniku zwróconego przez LLM """
    text = text.strip()
    if text.startswith("Wynik:"):
        text = text[6:].strip()
    elif text.startswith("Result:"):
        text = text[7:].strip()

    tmp = text.lower().replace('[','').replace(']','').replace('{','').replace('}','').strip()
    tmp = tmp.replace('result','').replace(':','')
    tmp = tmp.replace('.','').replace("'",'').replace('"','').strip()
    if tmp == 'brak danych':
        text = 'null'
    elif tmp == 'no data':
        text = 'null'

    if text != 'null' and not '[' in text:
        text = '[' + text + ']'

    try:
        data = json.loads(text)
        format_result = True
    except json.decoder.JSONDecodeError as json_err:
        print(json_err.msg, '\n', text)
        format_result = False
        data = text

    return format_result, data


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    if USE_API:
        print('UWAGA: uruchomiono w trybie realnego przetwarzania z wykorzystaniem API - to kosztuje!')
    else:
        print('Uruchomiono w trybie testowym, bez użycia API (to nie kosztuje).')

    total_price_gpt = 0
    total_tokens = 0

    # szablon zapytania o podstawowe informacje na temat postaci
    prompt_path = Path("..") / "prompts" / "person_relations_en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # dane z próbki testowej
    test_folder = Path("..") / "short_250" / "basic"
    test_file_list = test_folder.glob('*.txt')
    test_collection = {}
    for test_file in test_file_list:
        test_file_name = os.path.basename(test_file)
        test_collection[test_file_name] = test_file

    tom = 'tom_01'
    data_folder = Path("..") / "data_psb" / "short_relations" / tom
    data_file_list = data_folder.glob('*.txt')

    # pomiar czasu wykonania
    start_time = time.time()

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        data_file_name = os.path.basename(data_file)

        # czy nie był to plik już przetwarzany w teście
        if data_file_name in test_collection:
            source_file = Path("..") / "output_short_json_250" / "relations" / data_file_name.replace('.txt', '.relations.json')
            target_file = Path("..") / "output_psb" / "relations" / tom / data_file_name.replace('.txt', '.json')
            shutil.copyfile(source_file, target_file)
            # skoro wyniki były gotowe to skrypt przechodzi do następnego pliku
            print(f'Biogram: {data_file_name} wynik z próbki testowej')
            continue

        # wczytanie tekstu z podanego pliku
        text_from_file = ''
        with open(data_file, 'r', encoding='utf-8') as f:
            text_from_file = f.read().strip()

        output_path = Path("..") / 'output_psb' / 'relations' / tom / data_file_name.replace('.txt','.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # weryfikacja liczby tokenów
        tokens_in_data = count_tokens(text_from_file)
        if tokens_in_data > MAX_TOKENS:
            print(f'Biogram przekracza ograniczenia modelu: {data_file_name}')
            continue
            # zbyt długi tekst biogramu można podzielić na części
            #texts_from_file = get_text_parts(text_from_file, MAX_TOKENS)

        print(f'Biogram: {data_file_name}', end='')
        # przetwarzanie modelem gpt-3.5-turbo
        if USE_API:
            llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, text_from_file, model='gpt-3.5-turbo')
        else:
            # tryb testowy
            llm_prompt_tokens = count_tokens(prompt + text_from_file)
            llm_compl_tokens = 120 # przeciętna liczba tokenów w odpowiedzi
            llm_result = """Wynik:
                            {
                            "place_of_birth":"Władysławowo",
                            "place_of_death":"Łódź",
                            "place_of_burial":{"place": "Łódź", "note": "cmentarz katolicki przy ul. Ogrodowej"},
                            "date_of_birth":{"date":"1814-10-13"},
                            "date_of_death":{"date":"1904-01-17"},
                            "date_of_burial":"brak danych"
                            }
                        """

        json_ok, llm_dict = format_result(llm_result)

        # zapis do pliku json
        with open(output_path, 'w', encoding='utf-8') as f:
            if json_ok:
                json.dump(llm_dict, f, indent=4, ensure_ascii=False)
            else:
                f.write(llm_dict)

        # obliczenie kosztów
        price_gpt = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT))
        print(f' ({llm_prompt_tokens}), koszt: {price_gpt:.2f}')

        total_price_gpt += price_gpt
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
