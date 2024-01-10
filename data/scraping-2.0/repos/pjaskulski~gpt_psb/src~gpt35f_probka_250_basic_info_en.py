""" analiza próbki 250 biogramów - informacje podstawowe """
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import spacy
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)


# tryb pracy, jeżeli True używa API OpenAI, jeżeli False  to tylko test
USE_API = True

# maksymalna wielkość odpowiedzi
OUTPUT_TOKENS = 400

# wielkość kontekstu dla modelu GPT-3.5-turbo
MODEL_TOKENS = 4096

# ceny gpt-3.5-turbo (fine-tuning) w dolarach
INPUT_PRICE_GPT3F = 0.012
OUTPUT_PRICE_GPT3F = 0.016

# przetwarzanie modelem gpt-3.5-turbo po fine-tuningu
MODEL = 'ft:gpt-3.5-turbo-0613:personal::7tiu4gBL'


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


def count_tokens(text:str, model:str = "gpt-3.5-turbo") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    return openai.ChatCompletion.create(**kwargs)


def get_answer(prompt:str='', text:str='', model:str=MODEL) -> str:
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
    if text.lower().strip() == 'brak danych':
        text = '{"result": "brak danych"}'
    elif text.lower().strip() == 'brak danych.':
        text = '{"result": "brak danych"}'

    if not '[' in text:
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

    total_price_gpt35 = 0
    total_tokens = 0

    # szablon zapytania o podstawowe informacje na temat postaci
    prompt_path = Path("..") / "prompts" / "person_basic_en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    PROMPT_TOKENS = count_tokens(prompt)

    # maksymalna liczba tokenów w treści biogramu
    MAX_TOKENS = MODEL_TOKENS - PROMPT_TOKENS - OUTPUT_TOKENS

    # dane z pliku tekstowego
    data_folder = Path("..") / "short_250" / "basic"
    data_file_list = data_folder.glob('*.txt')

    # pomiar czasu wykonania
    start_time = time.time()

    for data_file in data_file_list:
        # wczytanie tekstu z podanego pliku
        text_from_file = ''
        with open(data_file, 'r', encoding='utf-8') as f:
            text_from_file = f.read().strip()

        # nazwa pliku bez ścieżki
        data_file_name = os.path.basename(data_file)
        output_path = Path("..") / 'output_short_json_250' / 'basic_35_fine_tuning' / data_file_name.replace('.txt','.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # weryfikacja liczby tokenów
        tokens_in_data = count_tokens(text_from_file)
        # skracanie za długich biogramów
        if tokens_in_data > MAX_TOKENS:
            print(f'ERR: za długi tekst ({data_file_name})')
            sys.exit(1)

        if USE_API:
            llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, text_from_file, model=MODEL)
        else:
            # tryb testowy
            llm_prompt_tokens = count_tokens(text_from_file) + PROMPT_TOKENS
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
        price_gpt35 = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT3F) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT3F))
        print(f'Biogram: {data_file_name} ({llm_prompt_tokens}), koszt: {price_gpt35:.2f}')

        total_price_gpt35 += price_gpt35
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt35:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
