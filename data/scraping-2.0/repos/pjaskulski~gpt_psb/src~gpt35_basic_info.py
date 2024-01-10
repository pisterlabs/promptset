""" analiza próbki 250 biogramów - informacje podstawowe - nowy model gpt-3.5-turbo-1106 """
import os
import sys
import json
import time
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

# MODEL
MODEL = 'gpt-3.5-turbo-1106'

SYSTEM = """
You are a helpful assistant, a specialist in history, genealogy, biographies of famous figures. You return the answers in JSON format.
Based on the text of the biography provided by user, find the place of birth, place of death, place of burial, date of birth, date of death, and date of burial of the protagonist. Present the results in the form of a list of JSON objects containing the fields:
place_of_birth: the birthplace of the protagonist (name of the town), if more detailed information about the place of birth is known, for example: 'near Krakow', 'Brzeski county', 'by the Vistula River', record it in an additional field: note
place_of_death: the place of death of the protagonist (name of the town), if more detailed information about the place of death is known, for example: 'near Krakow', 'Brzeski county', 'by the Vistula River', record it in an additional field: note
place_of_burial: the burial place of the protagonist (name of the town), if more detailed information about the burial place is known, record it in an additional field: note
date_of_birth: the date of birth of the protagonist, only if there are additional terms next to the date, for example: around, before, after, record them in an additional field: note
date_of_death: the date of death of the protagonist, only if there are additional terms next to the date, for example: around, before, after, record them in an additional field: note
date_of_burial: the date of burial of the protagonist, only if there are additional terms next to the date, for example: around, before, after, record them in an additional field: note
Instead of the protagonist's name in the depth of the text, the initial of the surname or first name may appear, e.g. S.
Note: medieval figures are often called, for example, Peter of Wadowice, but this does not mean that Wadowice is Peter's birthplace.
The place of death is not necessarily the same as the place of burial. Include only those burial places mentioned explicitly in the biography.
If any information is missing from the given text, write: 'no data'. Besides the result in JSON format, do not add any descriptions or comments to the answer. Include only information that is found in the text provided. Provide your answers in Polish.
"""
USER_01 = """
Text:
Soderini Carlos (1557- ok. 1591), kupiec i bankier.
Był jednym z pięciu synów Niccola i Annaleny Ricasoli, młodszym
bratem Bernarda (zob.). Ur. się 1 czerwca, we wsi Andalewo koło Wyszeborga. Jego bratanicą była Małgorzata Anna, żona
Winfrida de Loeve. S. ożenił się z Joanną, córką burgrabiego
krakowskiego Adama Kurozwęckiego. Zmarł w Hurczynianach, pochowano go po 15 czerwca 1591 roku na miejscowym cmentarzu parafialnym.
"""

ASSISTANT_01 = """
{
 "place_of_birth":{"place":"Andalewo", "note":"koło Wyszeborga"},
 "place_of_death":{"place":"Hurczyniany", "note":"powiat koniński"},
 "place_of_burial":{"place": Hurczyniany", "note": "miejscowy cmentarz parafialny"},
 "date_of_birth":{"date":"1557-06-01"},
 "date_of_death":{"date":"1591", "note":"około"},
 "date_of_burial":{"date":"1591-06-15", "note":"po"}
}
"""

USER_02 = """
Text:
Kowalski Jerzy (1857- ok. 1901), nauczyciel. Ur. 17 VII. Mieszkał i uczył się w Warszawie. Zmarł w Otwocku, w sierpniu 1901 roku, pochowano go po 15 sierpnia roku na cmentarzu ewangelickim.
"""

ASSISTANT_02 = """
{
 "place_of_birth":{"place":"brak danych"},
 "place_of_death":{"place":"Otwock"},
 "place_of_burial":{"place": Otwock", "note": "cmentarz ewangelicki"},
 "date_of_birth":{"date":"1857-07-17"},
 "date_of_death":{"date":"1901-08"},
 "date_of_burial":{"date":"1901-08-15", "note":"po"}
}
"""

USER_03 = """
Text: "Marian z Górki (1357- ok. 1401), rycerz. Na dworze Władysława Jagiełły pełnił funkcję podczaszego. Zmarł w Krakowie, około 1401 roku."
"""

ASSISTANT_03 = """
{
 "place_of_birth":{"place":"brak danych"},
 "place_of_death":{"place":"Kraków"},
 "place_of_burial":{"place": brak danych"},
 "date_of_birth":{"date":"1357"},
 "date_of_death":{"date":"1401"},
 "date_of_burial":{"date":"brak danych"}
}
"""

# maksymalna wielkość odpowiedzi
OUTPUT_TOKENS = 300
# wielkość angielskiego promptu to ok 900 tokenow, model obsługuje do 16000 tokenów
# maksymalna liczba tokenów w treści biogramu
MAX_TOKENS = 14000

# ceny gpt-3.5 w dolarach
INPUT_PRICE_GPT35 = 0.001
OUTPUT_PRICE_GPT35 = 0.002

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

# spacy do podziału tekstu na zdania
nlp = spacy.load('pl_core_news_lg')


def count_tokens(text:str, model:str = "gpt-3.5-turbo") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(3))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    client = OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response


def simple_get_answer(**kwargs):
    """ simple version """
    client = OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response


def get_answer(text:str='', model:str=MODEL) -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''
    prompt_tokens = completion_tokens = 0

    try:
        completion = simple_get_answer(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_01},
                        {"role": "assistant", "content": ASSISTANT_01},
                        {"role": "user", "content": USER_02},
                        {"role": "assistant", "content": ASSISTANT_02},
                        {"role": "user", "content": USER_03},
                        {"role": "assistant", "content": ASSISTANT_03},
                        {"role": "user", "content": f"{text}"}
                    ],
                    temperature=0.0,
                    top_p = 1.0,
                    response_format={"type": "json_object"},
                    max_tokens=OUTPUT_TOKENS)

        result = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

    except Exception as request_error:
        print(request_error)
        sys.exit(1)

    return result, prompt_tokens, completion_tokens


def format_result(text: str) -> tuple:
    """ poprawianie i formatowanie wyniku zwróconego przez LLM """
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


    print('UWAGA: uruchomiono w trybie realnego przetwarzania z wykorzystaniem API - to kosztuje!')

    total_price_gpt35 = 0
    total_tokens = 0

    # dane z pliku tekstowego
    tom = 'tom_46'
    data_folder = Path("..") / "data_psb" / "short" / tom
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

        output_path = Path("..") / 'output_psb' / 'basic_35' / tom / data_file_name.replace('.txt','.json')
        if os.path.exists(output_path):
            print(f'Plik {data_file_name.replace(".txt",".json")} z wynikiem przetwarzania już istnieje, pomijam...')
            continue

        # weryfikacja liczby tokenów
        tokens_in_data = count_tokens(text_from_file)
        if tokens_in_data > MAX_TOKENS:
            print(f'Biogram przekracza ograniczenia modelu: {data_file_name}')
            continue

        # przetwarzanie nowym modelem gpt-3.5-turbo
        llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(text_from_file, model=MODEL)

        json_ok, llm_dict = format_result(llm_result)

        # zapis do pliku json
        with open(output_path, 'w', encoding='utf-8') as f:
            if json_ok:
                json.dump(llm_dict, f, indent=4, ensure_ascii=False)
            else:
                f.write(llm_dict)

        # obliczenie kosztów
        price_gpt35 = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT35) +
                      ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT35))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'{current_time} Biogram: {data_file_name} ({llm_prompt_tokens}), koszt: {price_gpt35:.2f}')

        total_price_gpt35 += price_gpt35
        total_tokens += (llm_prompt_tokens + llm_compl_tokens)

        # przerwa między requestami
        time.sleep(0.25)

    print(f'Razem koszt: {total_price_gpt35:.2f} $, tokenów: {total_tokens}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
