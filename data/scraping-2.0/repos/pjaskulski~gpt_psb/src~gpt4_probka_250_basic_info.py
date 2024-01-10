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
# maksymalna liczba tokenów w treści biogramu
MAX_TOKENS = 6250

# ceny gpt-4 w dolarach
INPUT_PRICE_GPT4 = 0.03
OUTPUT_PRICE_GPT4 = 0.06

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


def short_version(text:str, first:int=10, last:int=10) -> str:
    """ proste skracanie biogramów - określona liczba początkowych i końcowych zdań """
    result = text

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) > first + last:
        select_data = []
        # pierwsze {first} zdań
        select_data = sentences[0:first]
        # ostatnie {last} zdań
        select_data += sentences[len(sentences) - last:]
        result = ' '.join(select_data)

    return result


def count_tokens(text:str, model:str = "gpt-4") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    return openai.ChatCompletion.create(**kwargs)


def get_answer(prompt:str='', text:str='', model:str='gpt-4') -> str:
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
    data = json.loads(text)

    return data


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    if USE_API:
        print('UWAGA: uruchomiono w trybie realnego przetwarzania z wykorzystaniem API - to kosztuje!')
    else:
        print('Uruchomiono w trybie testowym, bez użycia API (to nie kosztuje).')

    total_price_gpt4 = 0
    total_tokens = 0

    # spacy do podziału tekstu na zdania
    nlp = spacy.load('pl_core_news_md')

    # szablon zapytania o podstawowe informacje na temat postaci
    prompt_path = Path("..") / "prompts" / "person_basic.txt"
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

        if not text_from_file:
            print('Brak tekstu w pliku:', data_file)
            continue

        # nazwa pliku bez ścieżki
        data_file_name = os.path.basename(data_file)
        output_path = Path("..") / 'output_json_250' / 'basic' / data_file_name.replace('.txt','.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # weryfikacja liczby tokenów
        tokens_in_data = count_tokens(prompt + text_from_file)
        # bardzo długie biogramy mogą mieć dodatkowe akapity tekstu już po
        # informacji o śmierci bohatera/bohaterki biogramu, stąd więcej ostatnich zdań
        if tokens_in_data > 50000:
            last_sent = 35
        else:
            last_sent = 15

        # text biogramu jest skracany jeżeli ma > 5000 tokenów, chyba że ma mniej zdań niż first + last
        if tokens_in_data > 5000:
            text_from_file = short_version(text_from_file, first=10, last=last_sent)

        # przetwarzanie modelem gpt-4
        if USE_API:
            llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, text_from_file, model='gpt-4')
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

        llm_dict = format_result(llm_result)

        # zapis do pliku json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(llm_dict, f, indent=4, ensure_ascii=False)

        # obliczenie kosztów
        price_gpt4 = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT4) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT4))
        print(f'Biogram: {data_file_name} ({llm_prompt_tokens}), koszt: {price_gpt4:.2f}')

        total_price_gpt4 += price_gpt4
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt4:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
