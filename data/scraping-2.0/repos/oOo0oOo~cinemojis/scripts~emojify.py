import openai
import dotenv
import pandas as pd

import os
import re
import time

MOVIE_PATH = '../data/movies.csv'
EMOJI_PATH = '../data/emojis_raw/'

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

movies = pd.read_csv(MOVIE_PATH)

emoji_template = re.compile(r'(\U0001F600-\U0001F64F)')


def process_movie(metadata):
    ind, title, year = metadata

    # If file already exists, skip
    if os.path.isfile(f'{EMOJI_PATH}{ind}.txt'):
        return 0, 0

    # prompt = f'Summarize the movie "{title}" ({year}) using exclusively 20 emojis. No text explanation.'
    prompt = f'25 emojis that best explain the movie "{title}" ({year}). Only emojis, no text explanations.'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=120,
        n=5
    )

    txt = "".join([c['message']['content'] for c in response['choices']])
    
    # Rough filtering: Keep only emojis
    txt = emoji_template.sub('', txt)

    # Save to file
    with open(f'{EMOJI_PATH}{ind}.txt', 'w') as f:
        f.write(txt)
    print(f'Saved "{title}" as {ind}.txt')

    # Wait a bit
    time.sleep(2)

    # Extract number of tokens used
    tokens = response['usage']
    return tokens['prompt_tokens'], tokens['completion_tokens']


def price_gpt3(tokens):
    return round(tokens[0] * 0.001 * 0.0015 + tokens[1] * 0.001 * 0.002, 2)


# Iterate over the rows of the DataFrame
total_tokens = [0, 0]

for index, row in movies.iterrows():
    in_tokens, out_tokens = process_movie(row)
    total_tokens[0] += in_tokens
    total_tokens[1] += out_tokens

    # Print percentage done and accumulated price
    if index % 10 == 0:
        print(f'{index + 1} / {len(movies)} done. Total price: {price_gpt3(total_tokens)}$')

    


