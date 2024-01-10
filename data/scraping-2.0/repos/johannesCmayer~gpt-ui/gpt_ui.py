from glob import glob
import hashlib
import itertools
import math
import platform
import re
import subprocess
from pathlib import Path
import argparse
import json
import textwrap
import time
import os
import datetime
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Any
import html
from threading import Thread
import sys
from gsay import speak

import tiktoken
import yaml
import openai
from openai.error import TryAgain
import prompt_toolkit as pt
from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from xdg_base_dirs import xdg_config_home

# Basic helper functions
def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

def set_terminal_title(title):
    if platform.system() == 'Windows':
        os.system(f'title {title}')
    else:
        sys.stdout.write(f"\x1b]0;{title}\x07")
        sys.stdout.flush()

# Setting up Paths and looading config 1/2
project_dir = Path(__file__).parent.absolute()

config_dir = xdg_config_home() / 'gpt-ui'
config_file = config_dir / "config.yaml"
config_file_local = config_dir / "config_local.yaml"

# Loading config
config = yaml.load(config_file.open(), yaml.FullLoader)
# Load local config file and overwrite default config
if config_file_local.exists():
    config.update(yaml.load(config_file_local.open(), yaml.FullLoader))
openai.api_key = yaml.load((config_dir / 'api_key.yaml').open(), yaml.FullLoader).get('api_key')

model = config['default_model']
user = config['user']
models_dict = yaml.load((project_dir / 'models_metadata.yaml').open(), yaml.FullLoader)
max_tokens = models_dict[model]['max_tokens']
speak_default = config['speak']

# Setting up paths 2/2
if config["chat_dir"] is not None:
    chat_dir = Path(config["chat_dir"]).expanduser()
    if not chat_dir.exists():
        raise FileNotFoundError(f"Chat directory {chat_dir} does not exist.")
else:
    chat_dir = project_dir / "chats"
    chat_dir.mkdir(exist_ok=True)

chat_backup_file = chat_dir / f".backup_{timestamp()}.json"

prompt_history_dir = project_dir / "prompt_history"
prompt_history_dir.mkdir(exist_ok=True)

prompt_dir = project_dir / 'prompts'
prompt_dir.mkdir(exist_ok=True)

voice_precache_dir = project_dir / 'voice_precache'
voice_precache_dir.mkdir(exist_ok=True)

chat_dir.mkdir(exist_ok=True)
prompt_history_dir.mkdir(exist_ok=True)

obsidian_vault_dir = Path(config['obsidian_vault_dir']).expanduser()
if not obsidian_vault_dir.exists():
    raise FileNotFoundError(f"Obsidian vault directory {obsidian_vault_dir} does not exist.")

enc = tiktoken.encoding_for_model(model)

# Parsing Arguments
parser = argparse.ArgumentParser(description=
    "Press CTRL+C to stop generating the message. "
    "In user role press CTRL+D to exit the chat. You will first be asked to save the chat. "
    "Press CTRL+D in the safe dialog to exit the program without saving the chat. "
    "\n\n"
    "During a chat session there are a number of runtime commands you can enter as the user role. Enter 'help' as the "
    "user role to see all available runtime commands."
    "\n\n"
    "You can use :file:FILENAME: to show the contents of FILENAME to GPT, while in the UI the text will "
    "not be expanded. Similarly you can use :obsidian:FILENAME: in order to search the obsidian vault "
    "(needs to be configured in config.yaml) for the file FILENAME and show the contents to GPT.")
parser.add_argument('--chat-name', type=str, help='Name of the chat')
parser.add_argument('--load-chat', type=str, help='Name of the chat to load')
parser.add_argument('--load-last-chat', action='store_true', help='Name of the chat to load')
parser.add_argument('--list-chats', action='store_true', help='List all chats')
parser.add_argument('--list-all-chats', action='store_true', help='List all chats including hidden backup chats')
parser.add_argument('--list-models', action='store_true', help='List all models')
parser.add_argument('--list-models-full', action='store_true', help='List all models and their details')
parser.add_argument('--speak', default=speak_default, action='store_true', help='Speak the messages.')
parser.add_argument('-p', '--personality', default='helpful_assistant', type=str, choices=[x.stem for x in prompt_dir.iterdir()], help='Set the system prompt based on predefined file.')
parser.add_argument('--config', action='store_true', help='Open the config file.')
parser.add_argument('--debug', action='store_true', help='Run with debug settings. Includes notifications.')
parser.add_argument('--export-chats-to-markdown', action='store_true', help='Re export all named chats as markdown files into the chat directory.')
parser.add_argument('user_input',  type=str, nargs='*', help='Initial input the user gives to the chat bot.')
args = parser.parse_args()
if args.user_input == []:
    args.user_input = None
else:
    args.user_input = " ".join(args.user_input)
    if args.user_input == "":
        args.user_input = None

assistant_name = 'assistant'
def GET_DEFAULT_CHAT(): 
    prompt_path = prompt_dir / (args.personality + ".yaml")
    if not prompt_path.exists():
        print(f"Prompt file {prompt_path} does not exist.")
        print("The foolowing prompt files are available:")
        for prompt_file in (project_dir / 'prompts').iterdir():
            print(prompt_file.stem)
        exit(0)
    return yaml.load(prompt_path.open(), yaml.FullLoader)

class Command:
    def __init__(self, str_matches, description):
        self.str_matches = str_matches
        self.description = description
    def __str__(self) -> str:
        return f"{'/'.join(self.str_matches)}: {self.description}"

# TODO: Find a better way to get the same functionality, without needing to repeat the command names in __str__
class Commands:
    exit = Command(['exit'], 'Exit the program')
    pass_ = Command(['pass'], 'Pass the turn of the current role?')
    restart = Command(['restart'], 'Clear the entire chat (excluding system message)')
    restart_hard = Command(['restart hard'], 'Clear the entire chat (including system message)')
    list = Command(['list', 'ls'], 'List all saved chats')
    list_all = Command(['list all', 'ls all'], 'List all saved chats including hidden backup chats')
    load = Command(['load'], 'Load a chat')
    save = Command(['save'], 'Save the chat')
    edit = Command(['vi', 'vim', 'nvim'], 'Edit the chat')
    regenerate = Command(['regenerate'], 'Regenerate the chat')
    speak = Command(['speak', 's'], 'Speak the messages')
    speak_last = Command(['speak last', 'sl'], 'Speak the last messages')
    help = Command(['help', 'h'], 'Show this help message')
    def __str__(self) -> str:
        return '\n'.join([str(x) for x in [Commands.exit, Commands.pass_, Commands.restart, Commands.restart_hard, Commands.list, \
                                            Commands.list_all, Commands.load, Commands.save, Commands.edit, \
                                            Commands.regenerate, Commands.speak, Commands.speak_last, \
                                            Commands.help]])

commands = Commands()

def HTML_color(text, color):
    return f'<style fg="ansi{color}">{text}</style>'

def HTML_bold(text):
    return f'<b>{text}</b>'

def color_by_role(role, text=None):
    ret = None
    if "system" in role:
        ret = HTML_color(text if text else role, "blue")
    elif "user" in role:
        ret = HTML_color(text if text else role, "green")
    else:
        ret = HTML_color(text if text else role, "red")
    return HTML_bold(ret)

def next_role(chat):
    if len(chat) == 0:
        return "system"
    elif chat[-1]["role"] == "user":
        return "assistant"
    else:
        return "user"

def print_chat(chat):
    for m in chat:
        name = m['model'] if m['role'] == 'assistant' \
                          else (m['user'] if m['role'] == 'user' else 'system')
        prompt = f'{name}:'
        prompt += config['prompt_postfix']
        pt.print_formatted_text(HTML(f"{color_by_role(m['role'], prompt)}"))
        pt.print_formatted_text(f"{m['content']}")

def append_to_chat(chat, role, content, l_date=None, l_model=None, l_user=None):
    date = timestamp()
    chat.append({"role": role, "model": l_model if l_model else model, 'user': l_user if l_user else user, 'date': l_date if l_date else date, "content": content})
    backup_chat(chat)

def number_of_tokens(chat):
    length = 0
    for c in chat:
        length += len(enc.encode(c['content']))
    return length

def trim_chat(chat):
    num_tokens = len(enc.encode(chat[0]['content']))
    new_chat = []
    for i, e in enumerate(reversed(chat[1:])):
        num_tokens += len(enc.encode(e['content']))
        if num_tokens > max_tokens:
            break
        new_chat.append(e)
    new_chat = list(reversed(new_chat))
    return [chat[0]] + new_chat, num_tokens

def backup_chat(chat, name=None, prompt_name=None):
    if len(chat) == 0:
        return
    # Always backup chat first, even if we are prompting for a name
    with ensure_extension(chat_backup_file, ".json").open("w") as f:
        json.dump(chat, f, indent=4)
    if prompt_name:
        try:
            user_input_name = pt.prompt("Save name: ")
            with (chat_dir / ensure_extension(user_input_name, "json")).open("w") as f:
                json.dump(chat, f, indent=4)
            return user_input_name
        except EOFError as e:
            pass
    elif name:
        with (chat_dir / ensure_extension(name, '.json')).open("w") as f:
            json.dump(chat, f, indent=4)
        return name
    else:
        return chat_backup_file

def edit_chat(chat, user_input):
    backup_chat(chat)
    meta_data_prefix = f"###>>>"
    with (chat_dir / 'temp').open("w") as f:
        for m in chat:
            meta_data = json.dumps({k: v for k, v in m.items() if k != 'content'})
            f.write(f"{meta_data_prefix}{meta_data}\n{m['content']}\n\n")
        meta_data = json.dumps({'role': next_role(chat), 'model': model, 'user': user, 'date': timestamp()})
        f.write(f"{meta_data_prefix}{meta_data}\n\n")
    os.system(f"{user_input} {chat_dir / 'temp'}")
    with (chat_dir / 'temp').open() as f:
        chat = []
        role = None
        text = ""
        last_r = None
        for line in f:
            r = None
            if line.startswith(meta_data_prefix):
                r = json.loads(line[len(meta_data_prefix):])
                last_r = r
            if r:
                if role:
                    append_to_chat(
                        chat, 
                        role, 
                        text.strip(), 
                        l_date=r['date'] if r and 'date' in r else None, 
                        l_user=r['user'] if r and 'user' in r else None,
                        l_model=r['model'] if r and 'model' in r else None)
                backup_chat(chat)
                text = ""
                role = r['role']
            else:
                text += line
        if text.strip() != "":
            append_to_chat(
                chat, 
                role, 
                text.strip(), 
                l_date=last_r['date'] if 'date' in last_r else None, 
                l_user=last_r['user'] if 'user' in last_r else None,
                l_model=last_r['model'] if 'model' in last_r else None)
        backup_chat(chat)
    (chat_dir / 'temp').unlink()
    print('\n\n')
    print_chat(chat)
    return chat

# def speak_all_as_sentences(text):
#     hash = hashlib.md5(text.encode('utf-8')).hexdigest()
#     last_hash = None
#     while hash != last_hash:
#         text = speak_first_sentence(text)
#         last_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
#         hash = hashlib.md5(text.encode('utf-8')).hexdigest()
#     speak(text)

def list_chats(hide_backups=True):
    for chats in sorted(chat_dir.iterdir()):
        if chats.is_dir() or chats.name.endswith('.md'):
            continue
        color = 'green'
        if hide_backups and chats.name.startswith('.'):
            continue
        if chats.name.startswith('.backup'):
            color = 'magenta'
        pt.print_formatted_text(HTML(HTML_color(html.escape(chats.name), color)))
        with chats.open() as f:
            chat = json.load(f)
            print(textwrap.shorten(chat[-1]['content'], width=100))
        print()

def get_file_content_embeding(path):
    if not path.exists():
        return f"Error: The file {path} does not exist. Tell this to the user very briefly, telling them the path that does not exsist, ignoring the rest of the prompt."
    with path.open() as f:
        text = f.read()
    return f"\n{path}>>>\n{text}\n<<<{path}\n"

def search_file(start_path: Path, target_file: str) -> Optional[List[Path]]:
    matches = list(start_path.rglob(target_file))  # Search for target file
    if matches:
        return matches # Returns list of all matches file paths
    else:
        return None

def search_single_file(start_path: Path, target_file: str) -> Path:
    matches = list(start_path.rglob(target_file))  # Search for target file
    if matches:
        if len(matches) == 1:
            return matches[0] # Returns list of all matches file paths
        else:
            completer = WordCompleter([str(x) for x in matches])
            result = pt.prompt('Please enter your choice:', completer=completer)
            return Path(result)
    else:
        return None

def explode_file_links(chat):
    for c in chat:
          c.update({'content': re.sub(':file:(.*):',
                 lambda match: get_file_content_embeding(Path(match.group(1))), 
                 c['content'])})
    return chat

def ensure_extension(string: str, ext: str) -> str:
    """Ensure that the text ends with a particular extension."""
    path = False
    if isinstance(string, Path):
        string = str(string)
        path = True
    return_value = None
    if string.endswith(ext):
        return_value = string
    else:
        return_value = f"{string}{ext}"
    if path:
        return_value = Path(return_value)
    return return_value


def resolve_obsidian_links(chat):
    for c in chat:
          c.update({'content': re.sub(':obsidian:(.*):',
                 lambda match: f":file:{search_single_file(obsidian_vault_dir, ensure_extension(match.group(1), '.md'))}:",
                 c['content'])})
    return chat

def explode_chat(chat):
    chat = deepcopy(chat)
    chat = resolve_obsidian_links(chat)
    chat = explode_file_links(chat)
    return chat

def get_summary(chat):
    try:
        exploded_chat = explode_chat(chat)
        summarize_instuctions = (
            'Please give a summary of the conversation so far in 5 words or less. You do not need to make a complete sentence. '
            'Be as brief and descriptive as possible. Ideally do not leave out any topics discussed. If there are too many '
            'topics (and only then) it is ok to write a longer than 5 words summary, but still keep it as brief as possible. '
            'Do not use the following characters in your output: "*", ":", "!", "?", "/", "\\"')
        summary_response = openai.ChatCompletion.create(
            model=model,
            messages=[{k: v for k, v in y.items() if k in ['role', 'content']} for y in exploded_chat] + [{'role': 'user', 'content': summarize_instuctions}],
        )
        summary = summary_response["choices"][0].message["content"]
    # TODO make this exception more specific
    except Exception as e:
        summary = ''
        print('Error: ', e)
    return summary

class Toolbar:
    def __init__(self):
        self.n_tokens = 0
        self.summary = ''
        self.worker = None

    def __str__(self):
        return f'{model} | {int(self.n_tokens/max_tokens*100)}% {self.n_tokens}/{max_tokens} | {args.personality} | {self.summary}'

    def background_update(self, chat):
        if self.worker is None or not self.worker.is_alive():
            t = Thread(target=self._update, args=[chat])
            self.worker = t
            t.start()

    def _update(self, chat):
        self._update_num_tokens(chat)
        self._update_summary(chat)

    def _update_num_tokens(self, chat):
        self.n_tokens = number_of_tokens(explode_chat(chat))

    def _update_summary(self, chat):
        self.summary = get_summary(chat)
        set_terminal_title(f"GPT {self.summary}")

bottom_toolbar_session = Toolbar()

class Speaker:
    def __init__(self):
        self.arg_queue = []
        self.queue_processor = None
        self.stop_flag = False
        self.speak_subprocess = None

    def stop(self):
        self.stop_flag = True
        if self.speak_subprocess is not None:
            self.speak_subprocess.kill()
        self.arg_queue = []

    def speak(self, cmd, text) -> None:
        text = self._prepare_text(text)
        text = text.strip()
        if text == "":
            return
        speak_touple = self._precompute_speech(cmd, text)
        self.arg_queue.append(speak_touple)
        if self.queue_processor is None or not self.queue_processor.is_alive():
            tp = Thread(target=self._process_queue)
            self.queue_processor = tp
            tp.start()

    def _process_queue(self):
        while len(self.arg_queue) > 0 and not self.stop_flag:
            t = Thread(target=self._speak)
            t.start()
            t.join()

    def _precompute_speech(self, cmd, text) -> Tuple[subprocess.Popen, Path]:
        if cmd == "say" and platform.system() == "Darwin":
            extension = ".aiff"
        else:
            extension = ".mp3"
        cache_file = voice_precache_dir / f"{timestamp()}{extension}"
        debug_notify(text)
        proc = subprocess.Popen([cmd, "--output-file", cache_file, "--speed", str(1), "--", text], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return proc, Path(cache_file), text

    def _prepare_text(self, reading_buffer):
        reading_buffer = re.sub('`', '', reading_buffer)
        reading_buffer = re.sub('"', '', reading_buffer)
        # Filter out python interpreter prompt
        reading_buffer = re.sub('>>> ', '', reading_buffer)
        # Filter out underscores
        reading_buffer = re.sub('_', ' ', reading_buffer)
        reading_buffer = reading_buffer.strip()
        return reading_buffer

    def _speak(self):
        proc, cache_file, text = self.arg_queue.pop(0)
        proc.wait()
        mpv_speed = 1.4 #min(sum([len(s[2]) for s in self.arg_queue]) * 0.0001  + 1, 1.1)
        if cache_file.exists():
            self.speak_subprocess = subprocess.Popen(["mpv", "--really-quiet", f"--speed={mpv_speed}", cache_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            self.speak_subprocess.wait()
            cache_file.unlink()


def get_first_sentence(text: str) -> Tuple[str, str]:
    """Split the first sentence out of a text blob.
       @return: Rutrns a touple of (first sentence, rest of text)
    """
    end_chars = ['.', '?', '!', ':', '。', '？', '！']
    end_markers = []
    for c in end_chars:
        end_markers.append(c + ' ')
        end_markers.append(c + '\n')
        end_markers.append(c + '"')
        end_markers.append(c + "'")

    first_sentence = ''
    remaining_text = text
    for i in range(len(text)):
        if text[i] == '\n' or (len(text) >= 2 and text[i:i+2] in end_markers):
            first_sentence = text[:i+1]
            remaining_text = text[i+1:]
            break
    return first_sentence, remaining_text

def chat_to_markdown(chat):
    markdown = '%% Auto geneterated file, do not edit %%\n\n'
    for m in chat:
        if m['role'] == 'assistant':
            speaker = m.get('model', 'assistant')
        elif m['role'] == 'user':
            speaker = m.get('user', 'user')
        else:
            speaker = m['role']
        markdown += f"**{speaker}:** {m['content']}\n"
    return markdown

def save_chat_as_markdown(chat, name):
    with (chat_dir / f"{name}.md").open("w") as f:
        f.write(chat_to_markdown(chat))

def sanetize_filename(filename):
    """Sanetize a filename to be safe to use on most filesystems, as well as work with the Obsidian sync plugin."""
    filename = re.sub(':', ' - ', filename)
    filename = re.sub('[<>"/\|?*]+', '', filename)
    filename = re.sub('\.^', '', filename)
    return filename.strip()
    
def main():
    speak_cmd = 'gsay'
    save_name_session = PromptSession(history=FileHistory(prompt_history_dir /'saveing.txt'), auto_suggest=AutoSuggestFromHistory())
    user_prompt_session = PromptSession(history=FileHistory(project_dir /'user_prompt.txt'), auto_suggest=AutoSuggestFromHistory())
    def bottom_toolbar():
        return str(bottom_toolbar_session)
    
    if args.list_models_full:
        for m in sorted(openai.Model.list()['data'], key=lambda x: x['id']): 
            print(m)
        exit(0)

    if args.list_models:
        print('available models:')
        for m in sorted(openai.Model.list()['data'], key=lambda x: x['id']): 
            print(m['id'])
        exit(0)

    chat_name = args.chat_name

    if args.list_chats:
        list_chats()
        exit(0)
    if args.list_all_chats:
        list_chats(hide_backups=False)
        exit(0)
    elif args.config:
        subprocess.run([os.environ['EDITOR'], config_file])
        exit(0)
    if args.export_chats_to_markdown:
        for chat_file in chat_dir.iterdir():
            if chat_file.is_file() and not chat_file.name.startswith('.'):
                with chat_file.open() as f:
                    try:
                        chat = json.load(f)
                        save_chat_as_markdown(chat, chat_file.stem)
                    except Exception as e:
                        print(f"Error while exporting chat {chat_file}: {e}")
        exit(0)

    if args.user_input:
        chat = GET_DEFAULT_CHAT()
        chat.append({'role': 'user', 'content': args.user_input, 'user': config['user']})
    elif args.load_chat:
        with (chat_dir / ensure_extension(args.load_chat, ".json")).open() as f:
            chat = json.load(f)
    elif args.load_last_chat:
        chat_path = [x for x in sorted(chat_dir.iterdir()) if x.is_file() and x.name.startswith('.backup')][-1]
        with chat_path.open() as f:
            chat = json.load(f)
    else:
        chat = GET_DEFAULT_CHAT()

    print_chat(chat)

    active_role = next_role(chat)
    prompt_postfix = config['prompt_postfix']

    while True:
        try:
            if active_role in ['user', 'system']:
                ctrl_d = 0
                user_name = user if active_role == "user" else "system"
                try:
                    prompt = f'{user_name}:{prompt_postfix}'
                    prompt = color_by_role(active_role, prompt)
                    user_input = user_prompt_session.prompt(
                        HTML(prompt), 
                        bottom_toolbar=bottom_toolbar, 
                        auto_suggest=AutoSuggestFromHistory(),
                        multiline=True)
                except EOFError as e:
                    ctrl_d += 1
                if ctrl_d > 0 or user_input in commands.exit.str_matches:
                    backup_chat_name = backup_chat(chat)
                    while not chat_name or chat_name == '':
                        abort = False
                        try:
                            chat_name = save_name_session.prompt('Save name: ', default=sanetize_filename(bottom_toolbar_session.summary), bottom_toolbar=bottom_toolbar, auto_suggest=AutoSuggestFromHistory())
                        except EOFError as e:
                            ctrl_d += 1
                        except KeyboardInterrupt as e:
                            abort = True
                        if abort or ctrl_d > 1 or chat_name in commands.exit.str_matches:
                            pt.print_formatted_text(f"Chat saved as: {backup_chat_name}")
                            exit(0)
                        if (chat_dir / chat_name).exists() and pt.prompt('Chat already exists. Overwrite? ', bottom_toolbar=bottom_toolbar).lower() != 'y':
                                chat_name = ''
                                continue
                        time.sleep(0.1)
                    chat_name = sanetize_filename(chat_name)
                    chat_save_name = backup_chat(chat, chat_name)
                    save_chat_as_markdown(chat, chat_name)
                    pt.print_formatted_text(f"Chat saved as: {chat_save_name}")
                    exit(0)
                elif user_input in commands.pass_.str_matches:
                    active_role = "assistant"
                    continue
                elif user_input in commands.restart.str_matches:
                    backup_chat(chat)
                    chat = GET_DEFAULT_CHAT()
                    pt.print_formatted_text('\n\n')
                    pt.print_formatted_text(chat)
                    active_role = next_role(chat)
                    continue
                elif user_input in commands.restart_hard.str_matches:
                    backup_chat(chat)
                    print('\n\n')
                    chat = []
                    active_role = next_role(chat)
                    continue
                elif user_input in commands.list.str_matches:
                    list_chats()
                    continue
                elif user_input in commands.list_all.str_matches:
                    list_chats(hide_backups=False)
                    continue
                elif user_input in commands.load.str_matches:
                    for chars in chat_dir.iterdir():
                        if not chars.name.startswith('.'):
                            print(f"{chars.name}")
                    chat_name = pt.prompt('Name of chat to load: ')
                    if chat_name == 'exit':
                        continue
                    with chat_dir.joinpath(ensure_extension(chat_name, ".json")).open() as f:
                        backup_chat(chat)
                        chat = json.load(f)
                    print('\n\n')
                    print_chat(chat)
                    continue 
                elif user_input in commands.save.str_matches:
                    while not chat_name or (chat_dir / chat_name).exists() or chat_name == '':
                        chat_name = pt.prompt('Name chat: ').strip()
                        if chat_name == 'exit':
                            continue
                        time.sleep(0.1)
                    with (chat_dir / ensure_extension(chat_name, ".json")).open("w") as f:
                        json.dump(chat, f, indent=4)
                    continue
                elif user_input in commands.edit.str_matches:
                    chat = edit_chat(chat, user_input)
                    continue
                elif len(chat) >= 3 and user_input in commands.regenerate.str_matches:
                    backup_chat(chat)
                    chat = chat[:-1]
                    active_role = next_role(active_role)
                    print('\n\n')
                    print_chat(chat)
                    continue
                elif user_input in commands.speak.str_matches:
                    args.speak = not args.speak
                    print(f"Speak messages: {args.speak}")
                    continue
                elif user_input in commands.help.str_matches:
                    print(commands)
                    continue
                elif user_input in commands.speak_last.str_matches:
                    Speaker().speak(speak_cmd, chat[-1]['content'])
                    continue

                # Check if the user input starts with a model identifier, and if so,
                # set the model appropriately
                for n_model in models_dict.values():
                    for name in [n_model['name']] + n_model['aliases']:
                        if user_input.startswith(name):
                            global model
                            global max_tokens
                            model = n_model['name']
                            max_tokens = n_model['max_tokens']
                            user_input = user_input[len(name):].strip()
                        continue

                append_to_chat(chat, active_role, user_input)
                backup_chat(chat)
                active_role = next_role(chat)
            elif active_role == 'assistant':
                # Get the content iterator
                chat, num_tokens = trim_chat(chat)
                max_retries = 5
                for try_idx in itertools.count(1):
                    try:
                        exploded_chat = explode_chat(chat)
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=[{k: v for k, v in y.items() if k in ['role', 'content']} for y in exploded_chat],
                            stream = True,
                        )
                        break
                    except TryAgain as e:
                        if try_idx > max_retries:
                            backup_chat(chat)
                            raise e
                        pt.print_formatted_text(HTML(HTML_color(f"Error. Retrying {try_idx}/{max_retries}", 'red')))
                        if args.debug:
                            pt.print_formatted_text(HTML(HTML_color(f"Error: {e}", 'red')))
                        time.sleep(1)
                complete_response = []
                pt.print_formatted_text(HTML(color_by_role(f'{model}:{prompt_postfix}')), end='', flush=True)

                # Process the content
                speaker = Speaker()
                read_buffer = ''
                try:
                    for chunk in response:
                        c = None
                        try:
                            c = chunk.choices[0].delta.content
                            complete_response.append(c)
                            read_buffer += c
                        except AttributeError as e:
                            pass
                        except Exception as e:
                            print("\nAn error occured.")
                            backup_chat(chat, prompt_name=True)
                            raise e

                        if c is None:
                            continue

                        print(c, end='', flush=True)

                        # if speak_subproc is None or speak_subproc.poll() is not None:
                        first_sentence, read_buffer = get_first_sentence(read_buffer)
                        if first_sentence != '':
                            speaker.speak(speak_cmd, first_sentence)
                except KeyboardInterrupt as e:
                    speaker.stop()

                # Speak the remaning buffer
                speaker.speak(speak_cmd, read_buffer)
                complete_response = ''.join(complete_response)
                append_to_chat(chat, 'assistant', complete_response)
                active_role = next_role(chat)
                print()
        except KeyboardInterrupt:
            speaker.stop()

        bottom_toolbar_session.background_update(chat)
    
def debug_notify(msg):
    if args.debug:
        os.system(f"notify-send '{msg}'")

if __name__ == "__main__":
    main()