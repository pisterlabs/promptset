""" analiza próbki 250 biogramów - relacje rodzinne głównego bohatera/bohaterki """
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
OUTPUT_TOKENS = 1000

# maksymalna liczba tokenów w treści biogramu
MAX_TOKENS = 2000

# ceny gpt-3.5-turbo w dolarach
INPUT_PRICE_GPT = 0.0015
OUTPUT_PRICE_GPT = 0.002
# ceny po fine-tuningu
#INPUT_PRICE_GPT = 0.0120
#OUTPUT_PRICE_GPT= 0.0160

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# spacy do podziału tekstu na zdania
nlp = spacy.load('pl_core_news_lg')


def short_version_relations(text:str) -> str:
    """ skracanie biogramów do zdań zawierających informacje o pokrewieństwie lub powinowactwie """

    select_data = []
    words = ['ojciec', 'matka', 'syn', 'córka', 'brat', 'siostra', 'żona',
                'mąż', 'zięć', 'synowa', 'dziecko', "ożenić", "ożeniony", "zamężna",
                "rodzic", "rodzice", "małżeństwo", "rodzeństwo",
                "żonaty", "ożenić", "poślubić", "wyjść"]

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    if len(sentences) > 6:
        # zawsze pierwsze trzy zdania
        select_data = sentences[0:3]

        # ze środkowych zdań tylko takie wskazujące na opisy relacji rodzinnych
        for i in range(3,len(sentences) - 3):
            sent_doc = nlp(sentences[i])
            for token in sent_doc:
                if token.lemma_ in words:
                    select_data.append(sentences[i])
                    break

        # ostatnie trzy zdania
        select_data += sentences[len(sentences) - 3:]
    else:
        # wszystko
        select_data = sentences[0:]

    result = ' '.join(select_data)
    return result


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


def format_result(text: str) -> dict:
    """ poprawianie i formatowanie wyniku zwróconego przez LLM """
    text = text.strip()
    if text.startswith("Wynik:"):
        text = text[6:].strip()
    if text.startswith("Result:"):
        text = text[7:].strip()
    if text.lower().strip() == 'brak danych':
        text = '{"result": "brak danych"}'
    elif text.lower().strip() == 'brak danych.':
        text = '{"result": "brak danych"}'

    if not '[' in text:
        text = '[' + text + ']'

    try:
        data = json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        print(json_err.msg, '\n', text)
        sys.exit(1)

    return data


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    if USE_API:
        print('UWAGA: uruchomiono w trybie realnego przetwarzania z wykorzystaniem API - to kosztuje!')
    else:
        print('Uruchomiono w trybie testowym, bez użycia API (to nie kosztuje).')

    total_price_gpt = 0
    total_tokens = 0

    # szablon zapytania o relacje rodzinne postaci
    prompt_path = Path("..") / "prompts" / "person_relations_en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # dane z pliku tekstowego
    data_folder = Path("..") / "data_psb_250"
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
        # ścieżka do pliku wyjściowego
        output_path = Path("..") / 'output_short_json_250' / 'relations' / data_file_name.replace('.txt','.relations.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".relations.json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # text biogramu jest zawsze skracany
        text_from_file = short_version_relations(text_from_file)

        # weryfikacja liczby tokenów, za duże teksty (mimo skracania) są na razie pomijane
        tokens_in_data = count_tokens(prompt + text_from_file)
        if tokens_in_data > 4096 - OUTPUT_TOKENS:
            print(f'Za duży kontekst: {tokens_in_data} tokenów, biogram: {data_file_name}')
            continue

        # przetwarzanie modelem gpt-3.5
        if USE_API:
            llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, text_from_file, model='gpt-4')
        else:
            # tryb testowy
            llm_prompt_tokens = count_tokens(prompt + text_from_file)
            llm_compl_tokens = 120 # przeciętna liczba tokenów w odpowiedzi
            llm_result = """Wynik:
                            [
                                {"family_relation":"ojciec", "person":"Niccola Ricasoli"},
                                {"family_relation":"matka", "person":"Annalena Ricasoli"},
                                {"family_relation":"brat", "person":"Bernard"},
                                {"family_relation":"bratanica", "person":"Małgorzata Anna"},
                                {"family_relation":"żona", "person":"Joanna"},
                                {"family_relation":"teść", "person":"Adam Kurozwęcki"}
                            ]
                        """

        llm_dict = format_result(llm_result)

        # zapis do pliku json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_dict, f, indent=4, ensure_ascii=False)

        # obliczenie kosztów
        price_gpt = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT))
        print(f'Biogram: {data_file_name} ({llm_prompt_tokens}), koszt: {price_gpt:.2f}')

        total_price_gpt += price_gpt
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
