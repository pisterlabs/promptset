import os
import openai
from constants import GPT_API_KEY
import time

openai.api_key = GPT_API_KEY
DATASET_FOLDER = './nerv_dataset'


def get_responses(song_name, genre):
    prompt_1 = f'The string on the line below is the YouTube video title of a piano cover of a(n) {genre} song.                 Given the string, write a rich musical description of the original song, not the cover.                 This description should be roughly 200 words.                 \n{song_name}'
    prompt_2 = f'Now, provide a list of aspects describing the music of the original song, using \",\" as the delimiter,                 and without spaces between aspects. Below are three examples:\n                 low quality,sustained strings melody,soft female vocal,mellow piano melody,sad,soulful,ballad\n                 sustained piano synth,lullaby,bright xylophone,mellow melody,bass guitar,dreamy,kids music,slow tempo,nostalgic,cello,no vocals,sleepy\n                 rock,passionate male vocal,harmonizing male vocals,wide electric guitar melody,groovy bass,punchy kick,punchy snare,shimmering cymbals,energetic,addictive'

    response_1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt_1}
            ]
    )
    caption = response_1['choices'][0]['message']['content'].lstrip('\n')

    response_2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt_2}
            ]
    )
    aspects = response_2['choices'][0]['message']['content'].lstrip('\n')
    
    return (caption, aspects)


def caption_all_genre_folders(overwrite=False):
    genre_list = os.listdir(DATASET_FOLDER)
    for genre in genre_list:
        if not genre.startswith('.') and os.path.isdir(f'{DATASET_FOLDER}/{genre}'):
            caption_genre_folder(genre, overwrite)


def caption_genre_folder(genre, overwrite=False):
    '''
    Takes in the name of a genre folder containing .mp3 files and generates rich text captions
    for each song in the genre folder in .txt format

            Parameters:
                    genre (str): the genre whose songs to caption
                    overwrite (bool): specify whether captioning should overwrite existing 
                                      .txt files with the same file name
    '''
    genre_path = f'{DATASET_FOLDER}/{genre}'

    song_list = os.listdir(genre_path)

    execution_times = []
    for song_name in song_list:
        if song_name.startswith('.'):
            continue

        caption_filename = song_name + '_caption.txt'
        aspects_filename = song_name + '_aspects.txt'
        song_path = os.path.join(genre_path, song_name)
        caption_path = os.path.join(genre_path, song_path, caption_filename)
        if overwrite or not os.path.isfile(caption_path):

            if not os.path.exists(song_path):
                os.mkdir(song_path)

            start_time = time.time()
            caption, aspects = get_responses(song_name, genre)
            assert(caption != None and len(caption) != 0)
            assert(aspects != None and len(aspects) != 0)

            with open(os.path.join(song_path, caption_filename), 'w', encoding='utf-8') as f:
                f.write(caption)
            with open(os.path.join(song_path, aspects_filename), 'w', encoding='utf-8') as f:
                f.write(aspects)

            end_time = time.time()
            elapsed_time = end_time - start_time
            execution_times.append(elapsed_time)

    avg_execution_time = sum(execution_times) / len(execution_times)
    print("Genre: {}, Average execution time per song: {:.2f} seconds".format(genre, avg_execution_time))
