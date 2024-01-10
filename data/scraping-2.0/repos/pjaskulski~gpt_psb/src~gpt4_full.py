""" analiza biogramów PSB - instytucje, osoby, miejsca - GPT4
(ang. prompt, nowe skrócone biogramy)
"""
import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import spacy
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

MODEL = 'gpt-4-1106-preview'

# maksymalna wielkość odpowiedzi
OUTPUT_TOKENS = 1000
# wielkość angielskiego promptu to ok 900 tokenow, model gpt-4 obsługuje do 8000 tokenów
# model gt-4-1106 - 128 tys.
MODEL_TOKENS = 128000

# ceny gpt-4-1106-preview w dolarach
INPUT_PRICE_GPT = 0.01
OUTPUT_PRICE_GPT = 0.03

# api key
env_path = Path(".") / ".env"
# api key
#env_path = Path(".") / ".env_ihpan"

load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# spacy, model polski do podziału tekstu na zdania
nlp = spacy.load('pl_core_news_lg')


def count_tokens(text:str, model:str = "gpt-4") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    client = OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response


def get_answer(prompt:str='', text:str='', model:str=MODEL) -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''
    prompt_tokens = completion_tokens = 0

    try:
        completion = get_answer_with_backoff(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant, a specialist in history, genealogy, biographies of famous figures. You return the answers in JSON format."},
                        {"role": "user", "content": f"{prompt}\n{text}"}
                    ],
                    temperature=0.0,
                    top_p = 1.0,
                    response_format={"type": "json_object"},
                    seed=2,
                    max_tokens=OUTPUT_TOKENS)

        result = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

    except Exception as request_error:
        print(request_error)
        sys.exit(1)

    return result, prompt_tokens, completion_tokens


def format_result(text: str) -> tuple:
    """ poprawianie i weryfikacja wyniku zwróconego przez LLM """
    text = text.strip()

    try:
        data = json.loads(text)
        result = True
    except json.decoder.JSONDecodeError as json_err:
        print(json_err.msg, '\n', text)
        result = False
        data = text

    return result, data


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # pomiar czasu wykonania
    start_time = time.time()

    print('UWAGA: uruchomiono w trybie realnego przetwarzania z wykorzystaniem API - to kosztuje!')

    total_price_gpt4 = 0
    total_tokens = 0

    # szablon zapytania o 5 kategorii informacji na temat postaci
    prompt_path = Path("..") / "prompts" / "person_inst_pers_place_func_rel_v1.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # prompt dla instytucji, osób i miejsc
    PROMPT_TOKENS = count_tokens(prompt)
    # maksymalna liczba tokenów w treści biogramu
    MAX_TOKENS = MODEL_TOKENS - PROMPT_TOKENS - OUTPUT_TOKENS

    # folder z tomem PSB
    tom = 'tom_01'
    data_folder = Path("..") / "data_psb" / "full" / tom
    data_file_list = data_folder.glob('*.txt')

    # utworzenie katalogu na wyniki jeżeli nie istnieje
    if not os.path.isdir(Path("..") / 'data_psb' / 'full' / tom ):
        os.mkdir(Path("..") / 'data_psb' / 'full' / tom )

    # dane z próbki testowej
    test_folder = Path("..") / "short_250" / "basic"
    test_file_list = test_folder.glob('*.txt')
    test_collection = {}
    for test_file in test_file_list:
        test_file_name = os.path.basename(test_file)
        test_collection[test_file_name] = test_file

    for data_file in data_file_list:
        # nazwa pliku bez ścieżki
        data_file_name = os.path.basename(data_file)

        # czy nie był to plik już przetwarzany w teście
        if data_file_name in test_collection:
            source_file = Path("..") / "output_json_250" / "combined_results" / data_file_name.replace('.txt', '.json')
            target_file = Path("..") / "output_psb" / "full" / tom / data_file_name.replace('.txt', '.json')
            shutil.copyfile(source_file, target_file)
            # skoro wyniki były gotowe to skrypt przechodzi do następnego pliku
            continue

        # wczytanie tekstu z podanego pliku
        text_from_file = ''
        with open(data_file, 'r', encoding='utf-8') as f:
            text_from_file = f.read().strip()

        output_path = Path("..") / 'output_psb' / 'full' / tom / data_file_name.replace('.txt','.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # weryfikacja liczby tokenów
        tokens_in_data = count_tokens(text_from_file)
        if tokens_in_data > MAX_TOKENS:
            print(f'Biogram przekracza ograniczenia modelu: {data_file_name}')
            continue

        # przetwarzanie modelem gpt-4-1106
        llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, text_from_file, model=MODEL)
        json_ok, llm_dict = format_result(llm_result)

        # zapis do pliku json
        with open(output_path, 'w', encoding='utf-8') as f:
            if json_ok:
                json.dump(llm_dict, f, indent=4, ensure_ascii=False)
            else:
                f.write(llm_dict)

        # obliczenie kosztów
        price_gpt4 = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'{current_time} Biogram: {data_file_name} ({llm_prompt_tokens}), koszt: {price_gpt4:.2f}')

        total_price_gpt4 += price_gpt4
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt4:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
