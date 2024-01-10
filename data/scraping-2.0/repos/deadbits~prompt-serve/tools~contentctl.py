#!/usr/bin/env python

import os
import sys
import uuid
import argparse
import configparser
import json
import yaml
import pandas as pd

from rich import print as rprint
from rich.prompt import Prompt
from git import Repo
from collections import defaultdict 
from pykwalify.core import Core
from langchain import PromptTemplate


statistics = defaultdict(lambda: defaultdict(int))
seen_uuids = set()
passed = 0
failed = 0


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            rprint(f'[bold][red](error)[/red][/bold] config file not found: {self.config_file}')
            sys.exit(1)
        
        rprint(f'[bold][green](status)[/green][/bold] loading config file: {self.config_file}')
        self.config.read(config_file)

    def get(self, section, key):
        answer = None

        try:
            answer = self.config.get(section, key)
        except:
            rprint(f'[bold][red](error)[/red][/bold] config file missing section: {section}')

        return answer


def init_repo(config):
    repo_path = config.get('main', 'repo_path')
    repo_name = config.get('main', 'repo_name')
    full_path = os.path.join(repo_path, repo_name)

    if os.path.exists(os.path.join(full_path, '.git')):
        rprint(f'[bold red](error)[/bold red] repository already exists: {full_path}')
        sys.exit(1)
    
    try:
        repo = Repo.init(full_path)
        rprint(f'[bold green](status)[/bold green] repository initialized: {repo.common_dir}')
    except Exception as err:
        rprint(f'[bold red](error)[/bold red] failed to initialize repository: {err}')
        sys.exit(1)


def convert_to_langchain(fpath):
    rprint(f'[bold green](status)[/bold green] converting template {fpath}')
    
    with open(fpath, 'r') as fp:
        try:
            data = yaml.safe_load(fp)
            prompt = data['prompt']
            
            # check if prompt-serve template contains input variables
            if 'input_variables' in data.keys():
                input_vars = data['input_variables']
                langchain_template = PromptTemplate(template=prompt, input_variables=input_vars)
            
            # otherwise we only use the prompt
            else:
                langchain_template = PromptTemplate(template=prompt, input_variables=[])
     
            return (data, langchain_template)
     
        except Exception as err:
            print
            rprint(f'[bold red](error)[/bold red] failed to convert prompt: {fpath} - {err}')
            return (None, None)


def ask_for_input(field_name, field_type, is_required, default_value=None):
    value = None
    
    if is_required:
        while not value:
            value = Prompt.ask(f'[bold]{field_name} ({field_type})[/bold]')
    else:
        value = Prompt.ask(f'[bold]{field_name} ({field_type}) [optional, default={default_value}][/bold]') or default_value

    if value:
        if field_type == 'int':
            value = int(value)
        elif field_type == 'float':
            value = float(value)
        elif field_type == 'bool':
            value = value.lower() in ['true', 't', 'false', 'f']
        elif field_type == 'seq':
            value = [item.strip() for item in value.split(',')]
    return value


def create_prompt():
    # create the YAML map
    map_info = {}

    map_info['title'] = ask_for_input('title', 'str', True)
    map_info['uuid'] = str(uuid.uuid4())
    map_info['description'] = ask_for_input('description', 'str', True)
    map_info['category'] = ask_for_input('category', 'str', True)
    map_info['provider'] = ask_for_input('provider', 'str', False)
    map_info['model'] = ask_for_input('model', 'str', False)
    
    # model settings with default values
    model_settings_fields = ['temperature', 'top_k', 'top_p', 'max_tokens', 'stream', 'presence_penalty', 'frequency_penalty']
    model_settings_types = ['float', 'int', 'float', 'int', 'bool', 'float', 'float']
    model_settings_defaults = [0.8, None, 1, None, False, 0.0, 0.0]
    
    model_settings = {field: ask_for_input(field, ftype, False, default) for field, ftype, default in zip(model_settings_fields, model_settings_types, model_settings_defaults)}
    
    # only add model_settings to map if it's not empty
    if any(model_settings.values()):
        map_info['model_settings'] = model_settings

    map_info['prompt'] = ask_for_input('prompt', 'str', True)
    
    # Sequence fields
    seq_fields = ['references', 'associations', 'packs', 'tags', 'input_variables']
    for field in seq_fields:
        map_info[field] = ask_for_input(field, 'seq', False)

    return map_info


def save_prompt(prompt, filename):
    try:
        with open(filename, 'w') as f:
            yaml.dump(prompt, f, sort_keys=False)
        rprint(f'[bold green](status)[/bold green] prompt saved to {filename}')
    except Exception as e:
        rprint(f'[bold red](error)[/bold red] failed to save prompt: {e}')


def collect_stats_from_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)

            if 'category' in data:
                statistics['category'][data['category']] += 1
            if 'provider' in data:
                statistics['provider'][data['provider']] += 1
            if 'model' in data:
                statistics['model'][data['model']] += 1

            for tag in data.get('tags', []):
                statistics['tags'][tag] += 1

        except yaml.YAMLError as err:
            rprint(f'[bold red](error)[/bold red] error getting stats from {file_path}: {err}')


def collect_stats_from_dir(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                collect_stats_from_file(os.path.join(root, file))


def display_stats():
    # convert dictionary to dataframe
    df = pd.DataFrame.from_dict({(i,j): statistics[i][j] 
                            for i in statistics.keys() 
                            for j in statistics[i].keys()},
                            orient='index', columns=['Count'])

    dfs = {field: pd.DataFrame(list(values.items()), columns=[field, 'Count']).sort_values('Count', ascending=False) 
        for field, values in statistics.items()}

    # Print our statistics in separate tables
    for field, df in dfs.items():
        rprint(f'[bold]{field}[/bold]')
        if field == 'tags':
            print('(top 5)')
            print(df.head(5).to_string(index=False))  # Only display top 5 for tags
        else:
            print(df.to_string(index=False))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', 
        action='store',
        help='configuration file'
    )

    parser.add_argument(
        '-i', '--init',
        action='store_true',
        help='initialize new repository from config file'
    )

    parser.add_argument(
        '-n', '--new',
        action='store', 
        help='create new prompt'
    )
    
    parser.add_argument(
        '-s', '--stats',
        action='store',
        help='show statistics for directory of prompts'
    )

    parser.add_argument(
        '-l', '--langchain',
        action='store',
        help='convert prompt-serve template to langchain PromptTemplate'
    )

    args = parser.parse_args()

    if args.init:
        if not args.config:
            rprint(f'[bold red](error)[/bold red] config file required for initialization')
            sys.exit(1)

        config = Config(args.config)
        init_repo(config)

    if args.new:
        if os.path.exists(args.new):
            rprint(f'[bold red](error)[/bold red] file already exists: {args.new}')
            sys.exit(1)

        prompt = create_prompt()
        save_prompt(prompt, args.new) 

    if args.stats:
        if not os.path.isdir(args.stats):
            rprint(f'[bold red](error)[/bold red] {args.stats} is not a valid directory')
            sys.exit(1)

        collect_stats_from_dir(args.stats)
        display_stats()
    
    if args.langchain:
        if not os.path.exists(args.langchain):
            rprint(f'[bold red](error)[/bold red] template does not exist: {args.langchain}')
            sys.exit(1)

        original, langchain_template = convert_to_langchain(args.langchain)
        if original is None or langchain_template is None:
            rprint(f'[bold red](error)[/bold red] failed to convert prompt: {args.langchain}')
            sys.exit(1)
        
        rprint(f'[bold green](status)[/bold green] successfully converted template: {args.langchain}')
        rprint(f'[bold orange3]LangChain PromptTemplate[/bold orange3]')
        print(json.dumps(langchain_template.dict(), indent=2))