import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re
import guidance
from datetime import datetime
from dotenv import load_dotenv
from typing import Any
from pydantic import BaseModel
from clone_collector import clean_everything
from clonr.data.parsers import FullURLParser, FandomParser
from scrapers.clone_gen_utils import find_char_url, find_links_with_same_base_url, get_all_example_dialogues

parser = FullURLParser()

class Fandom(BaseModel):
    fandom_name: str
    fandom_root: str

class FandomCharacter(BaseModel):
    name: str
    greeting: str
    short_description: str
    long_description: str
    avatar_uri: str
    fandom_char_name: str

## TODO: edit full list
def get_all_fandoms():
    total_fandoms = [
        Fandom(fandom_name='Harry Potter', fandom_root='harrypotter'),
        Fandom(fandom_name='Naruto', fandom_root='naruto'),
        Fandom(fandom_name='Marvel', fandom_root='marvel'),
        Fandom(fandom_name='DC', fandom_root='dc'),
        Fandom(fandom_name='Star Wars', fandom_root='starwars'),
        Fandom(fandom_name='Game of Thrones', fandom_root='gameofthrones'),
        Fandom(fandom_name='Genshin Impact', fandom_root='genshin-impact'),
        Fandom(fandom_name='Attack on Titan', fandom_root='attackontitan'),
    ]
    return total_fandoms

def get_fandom_top_chars(fandom):
    fandom_name = fandom.fandom_name 
    fandom_root = fandom.fandom_root

    load_dotenv()

    top_chars = """
    {{#system~}}
    You are a helpful AI assistant is knowledge about various fandoms.
    {{~/system}}
    {{#user~}}.

    Can you write down, in a list format with the list of str names only, the top 10 characters from the fandom {{fandom_name}}?

    {{~/user}}
    {{#assistant~}}
    {{gen 'result' temperature=0.1 max_tokens=500}}
    {{~/assistant}}
    """

    gpt_turbo = guidance.llms.OpenAI('gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    top_chars = guidance(top_chars, llm=gpt_turbo)

    result = top_chars(fandom_name=fandom_name)
    file_path = f'scrapers/top_fandom_chars/{fandom_root}.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'x') as f:
            f.write(result['result'])
    else:
        print(f"Fandom top chars: '{file_path}' already exists!")
    return result

def get_fandom_char(fandom, char_name):
    fandom_name = fandom.fandom_name
    fandom_root = fandom.fandom_root 

    load_dotenv()

    char_profile = """
    {{#system~}}
    You are a helpful AI assistant is knowledge about various fandoms.
    {{~/system}}
    {{#user~}}.

    Can you provide the following in JSON format:
    -greeting message
    -short description
    -long description
    
    for the character: {{char_name}} in {{fandom_name}}

    {{~/user}}
    {{#assistant~}}
    {{gen 'result' temperature=0.1 max_tokens=500}}
    {{~/assistant}}
    """

    gpt_turbo = guidance.llms.OpenAI('gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    char_profile = guidance(char_profile, llm=gpt_turbo)

    result = char_profile(char_name=char_name, fandom_name=fandom_name)
    char_path = "_".join(char_name.split(" ")) + "-" + fandom_root
    file_path = f'scrapers/fandom_char_profiles/{char_path}.json'
    if not os.path.exists(file_path):
        with open(file_path, 'x') as f:
            f.write(result['result'])
    else:
        print(f"Fandom char generation: '{file_path}' already exists!")
    return result

## TODO: add on char_img from fandom, etc. to complete the profile above

# Check for missing and sparse values in cols
def checking_missing_and_sparse_values(row, columns_to_check, sparse_threshold=5):
    missing_fields = []
    sparse_fields = []

    def check_sparse(value, column_name):
        if isinstance(value, str) and len(value) < sparse_threshold:
            sparse_fields.append(column_name)
        elif pd.isna(value):
            missing_fields.append(column_name)

    for column in columns_to_check:
        value = row[column]
        if isinstance(value, dict):
            for key, dict_value in value.items():
                check_sparse(dict_value, f'{column}.{key}')
        elif isinstance(value, list):
            for i, list_value in enumerate(value):
                check_sparse(list_value, f'{column}[{i}]')
        else:
            check_sparse(value, column)

    return {
        'missing_fields': missing_fields,
        'sparse_fields': sparse_fields
    }

# Heuristic for getting char fandom wiki
def get_char_fandom_wiki(char_name, short_description):
    char_wiki = ''

    if 'from' in char_name.lower():
        char_wiki = char_name.lower().split("from ")[1]
        char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')
    if 'of' in char_name.lower():
        char_wiki = char_name.lower().split("of")[1].strip("of ")
        char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')
    
    if 'from' in short_description.lower():
        char_wiki = short_description.lower().split("from ")[1]
        char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')
    if 'of' in short_description.lower():
        char_wiki = short_description.lower().split("of")[1].strip("of ")
        char_wiki = char_wiki.replace(" ", "-").strip('.').strip('!')

    return char_wiki

def get_fandom_content(df, name_col, wiki_col):
    parser = FullURLParser()
    df['fandom_content'] = df.progress_apply(lambda x: get_all_example_dialogues(x[name_col], x[wiki_col], parser), axis=1)
    return 

def fandom_examples(total_df):
    from_df_sample = total_df[total_df['name'].str.contains(' from ', case=False) | total_df['short_description'].str.contains(' from ', case=False)]
    return from_df_sample

def parse_fandom(char_name, char_wiki):
    base_url = find_char_url(char_name, char_wiki)
    found_links = find_links_with_same_base_url(base_url)
    get_all_example_dialogues(char_name, char_wiki, parser)
    return base_url

# Using local fandom files
def get_txt_files_in_directory(directory_path):
    txt_files = []    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Attempt to get char stories from fandom outputs
def parse_fandom_outputs_for_char_stories(input_file_path, output_directory):
    with open(input_file_path, 'r') as file:
        input_text = file.read()
    
    # split on ###
    sections = re.split(r'(?=^###)', input_text, flags=re.MULTILINE)
    
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, input_file_path.split('/')[-1])

    with open(output_file_path, 'w') as output:
        for section in sections:
            section_lines = section.strip().split('\n')
            
            if section_lines:
                if (
                    not any(line.strip().startswith(('*', '-', '1.', '2.', '3.', '4.', '5.', '###', '##', '#')) for line in section_lines[1:])
                    and any(line.strip() and not line.strip().startswith("#") for line in section_lines[1:])
                ): 
                    output.write(section)

# Parse initial fandom outputs from input dir to a cleaned output dir
def parse_fandom_outputs_dir(input_dir_path, output_dir_path):
    txt_files = get_txt_files_in_directory(input_dir_path)
    for txt_file in txt_files:
        parse_fandom_outputs_for_char_stories(txt_file, output_dir_path)
    return 

# Get fandom url from base name
def get_fandom_url_from_base_name(base_name):
    # TODO: edit
    base_name_map = {
        'overlord': 'overlordmaruyama',   
    }
    if base_name in base_name_map:
        return base_name_map[base_name]
    return base_name

def get_curated_clones():
    total_df = clean_everything()
    
    columns_to_check = ['long_description', 'short_description', 'greeting', 'avatar_uri', 'example_dialogues']
    total_df['missing_or_sparse_fields'] = total_df.apply(checking_missing_and_sparse_values, axis=1, columns_to_check=columns_to_check)
    
    clean_df = total_df[total_df['missing_or_sparse_fields'].apply(lambda x: len(x['missing_fields']) + len(x['sparse_fields']) == 0)]
    print(clean_df.shape)
    clean_df_sample = clean_df.head(2)

    # get initial fandom outputs
    for idx, row in clean_df_sample.iterrows():
        char_name = row['name']
        char_wiki = row['fandom_wiki']
        print(f"Processing char_name: {char_name}, char_wiki: {char_wiki}")
        parse_fandom(char_name, char_wiki)

    # clean fandom outputs
    input_dir_path = 'scrapers/fandom'
    output_dir_path = 'scrapers/fandom_cleaned'
    parse_fandom_outputs_dir(input_dir_path, output_dir_path)
    return 


if __name__ == '__main__':
    print("curating clones..")


