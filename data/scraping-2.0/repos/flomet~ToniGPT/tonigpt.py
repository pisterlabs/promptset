# -*- coding: utf-8 -*-

import os
import openai
from pathlib import Path
from datetime import datetime

from create_story import get_text_for_keywords
from tty_openai import tty

from typing import Union
import os
import shutil
import json 

import logging

def create_logger(): 
    logfile_path = Path(os.getcwd())
    logfile = logfile_path.joinpath('tonigpt.log')
    fmt = "%(asctime)s %(levelname)-8s %(module)s.py %(funcName)s() %(lineno)i: %(message)-s"
    logging.basicConfig(filename=str(logfile),
                        filemode='w',
                        format=fmt, 
                        level=logging.DEBUG, 
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.debug(f'Starting Samba - Ansa {datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")}')
    print(f'Create logfile {logfile} - {logfile.exists()}')


def get_create_filename(story_folder: Path) -> Path:
    dateTimeObj = datetime.now()
    filename = f"{dateTimeObj.strftime('%Y%m%d_%H%M%S')}.txt"
    return Path(f"{story_folder}/{filename}")

def get_new_filenumber(story_folder: Path) -> Path:
    """
    Check the given path for existing files with names "{number}.txt".
    Increment the number if the file already exists.
    Return first filename which is not taken.
    
    Args:
    - path (str): The path to check for existing files.
    
    Returns:
    - Path: The first filepath which is not taken.
    """
    
    filename = get_create_filename(story_folder)
    while filename.exists():
        filename = get_create_filename(story_folder)
    return filename


def store_story_text(story:str, story_folder:Path)->Path:
    new_story_filename = get_new_filenumber(story_folder)
    with open(new_story_filename, "w") as f:
        f.write(story)
    return new_story_filename

def store_meta_info_to_story(story_meta_info_fn:Path, keywords:list, genre:str, age:tuple, wordlimit:int, voice:str, prompt:str)->Path:
    info_dict = {"keywords":keywords, 
                 "genre":genre, 
                 "age":age, 
                 "wordlimit":wordlimit, 
                 "voice":voice.lower(), 
                 "prompt":prompt
                }
    with open(str(story_meta_info_fn), 'w') as fp:
        json.dump(info_dict, fp)
    return story_meta_info_fn

def tonigpt(keywords:list, genre:str, age:tuple, wordlimit:int, voice:str, story_path:Path)->Path: 
    
    create_logger()
    logger = logging.getLogger(__name__)
    
    client = openai.OpenAI()

# Create Story 
    logger.debug(f"Create Story with keywords: {keywords}, genre: {genre}, age: {age}, wordlimit: {wordlimit}, voice: {voice}")
    my_story_str, prompt = get_text_for_keywords(client, keywords, genre, age, wordlimit)

# Store Story    
    #story_path = Path(os.getcwd()).joinpath("stories")
    my_story_file = store_story_text(my_story_str, story_path)
    logger.debug(f"Story stored in {my_story_file}")

# Store meta info to story
    story_meta_info_fn = my_story_file.parent.joinpath(Path(my_story_file).stem + ".json")
    story_meta_info_fn = store_meta_info_to_story(story_meta_info_fn, keywords, genre, age, wordlimit, voice, prompt)
    logger.debug(f"Story story_meta_info stored in {story_meta_info_fn}")

# Create Audio
    my_story_mp3_tmp = tty(client, my_story_str, voice=voice)

# Move Audio to storage story folder
    my_story_mp3_file = my_story_file.parent.joinpath(Path(my_story_file).stem + ".mp3")
    shutil.move(my_story_mp3_tmp, my_story_mp3_file)
    logger.debug(f"Audio stored in {my_story_mp3_file}")
    return (my_story_mp3_file)

if __name__ == "__main__":
# Keywords to create story
    keywords = ["Jorin", "Hund", "Maus"]
    genre = "Detektivgeschichte"
    # genre_examples = "Geschichte zum Zählen zu lernen"
    # genre_examples = "Abenteuergeschichte"
    # genre_examples = "pädagogisch wertvolle Geschichte"
    # genre_examples = "Geschichte im Reimform"
    age = (2,3) # zwei bis drei Jahre 
    wordlimit = 750
    voice = "Alloy" #["Alloy", "Echo", "Fable", "Onyx", "Nova", "Shimmer"]


    tonigpt(keywords, genre, age, wordlimit, voice)