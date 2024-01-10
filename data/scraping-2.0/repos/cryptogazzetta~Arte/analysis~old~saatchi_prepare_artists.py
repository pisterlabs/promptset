import time
import pandas as pd
from infra import openai

def prepare_artists_df():
    file_path = './temporary-files/saatchi_artists_info.json'
    artists_df = pd.read_json(file_path)

    clean_exhibitions_system_message = {"role": "system", "content": "If user message doesnt contain any info of at least one exhibition, answer NaN. Otherwise, please organize the history of exhibitions in a single json format (without nesting) with keys 'event', 'event type', 'location', 'city/country', 'date'. 'event type' can take the following values: 'solo exhibition', 'collective exhibition', 'art fair'. In case of data absent or not clear, leave blank"}
    artists_df['ExhibitionsClean'] = artists_df['Exhibitions'].apply(lambda x: get_exhibitions_clean(x, clean_exhibitions_system_message))

    artists_df.to_json('./temporary-files/saatchi_artists_info_clean.json', orient='records')

    return artists_df


def get_exhibitions_clean(exhibition_raw, system_message):
    if exhibition_raw != exhibition_raw:
        return exhibition_raw

    time.sleep(20)
    exhibition_clean = openai.chat(system_message, exhibition_raw)[-1]['content']
    exhibition_clean = exhibition_clean.replace('\n', '').replace('  ', '')

    return exhibition_clean