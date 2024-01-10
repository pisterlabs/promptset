#!/usr/bin/env python3

import os
import sys
import math
import time
import datetime
import openai
import argparse
import tiktoken
import pkg_resources
from rich.prompt import Prompt
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from collections import namedtuple
from urllib import request
from gpterm.config import Config
from gpterm.shell import ShellHandler
from gpterm.enums import ThemeColors, Colors, ThemeMode, VoiceStop
from gpterm.utils import alias_for_model


class GptTerminal:
    def __init__(self, debug=False, theme=None, api_key=None, api_key_path='~/.openai-api-key'):
        self.debug = debug
        self.api_key = api_key
        self.api_key_path = os.path.abspath(os.path.expanduser(os.path.expandvars(api_key_path)))
        self.shell = None
        self.conversation = ""
        self.conversation_formatted = ""
        self.prompt_idx = 0
        self.resp_line = ""
        self.resp_sentence = ""
        self.resp_start = True
        self.in_code_block = False
        self.prompt = ""
        self.prompt_input = ""
        self.first_sentence = True
        self.in_gpt_response = False
        self.abort_response = False
        self.after_reset = True
        self.config_file_path = Config.DEFAULT_CONFIG_PATH
        self.cfg = Config(self.config_file_path)
        self.cfg.load()
        if theme:
            self.cfg.color_theme = ThemeMode[theme]
        self.colors = self.get_term_colors()
        self._setup_openai()
        self._setup_gpt()
        self._setup_console()
        self._setup_code_format()
        self._setup_voice()

    def get_commands(self, advanced=False):
        Command = namedtuple('Command', ['advanced', 'setting', 'nargs', 'description'])
        commands = {
            '/help':  Command(False, None, 0, "List available commands"),
            '/exit': Command(False, None, 0, "Exit GPTerm"),
            '/save': Command(False, None, 0, "Save current settings"),
            '/context': Command(False, None, 0, "Print the current chat context (conversation) to screen"),
            '/reset': Command(False, None, 0, "Reset the chat context"),
            '/block': Command(False, None, 0, "Enter a multi-line input"),
            '/image': Command(False, None, 0, "Generate an image from a description using a DALL·E model"),
            '/theme': Command(False, self.cfg.color_theme, 0, "Toggle color theme to match background: light or dark"),
            '/code': Command(False, self.cfg.use_code_format, 0, "Toggle code format on/off (toggling will reset context)"),
            '/voice': Command(False, self.cfg.use_voice, 0, "Toggle voice on/off"),
            '/advanced': Command(False, self.cfg.display_advanced, 0, "Toggle display of advanced commands"),
            '/image-size': Command(True, self.cfg.image_size, 1, "Set the size (pixels x pixels) of generated images. options: 256, 512, 1024"),
            '/image-view': Command(True, self.cfg.image_view, 0, "Toggle to display image after generation in default viewer or not"),
            '/image-store': Command(True, None, 1, "Set the path to store generated images"),
            '/voice-name': Command(True, self.cfg.voice_name, 1, "Set the voice to be used"),
            '/voice-over': Command(True, self.cfg.voice_over, 0, "Toggle voice over highlighting"),
            '/voice-stop': Command(True, self.cfg.voice_stop, 0, "Toggle voice stop: period or newline"),
            '/model': Command(True, self.cfg.model, 1, "GPT Models. Possible options: chatgpt, davinci, curie, babbage, ada (or any custom trained model)"),
            '/temperature': Command(True, self.cfg.temperature, 1, "Provide a value between 0 and 1. Higher for more diverse responses. Lower for more deterministic")
        }
        if advanced:
            return commands
        else:
            return {key: commands[key] for key in commands if not commands[key].advanced}

    def _setup_gpt(self):
        self.stream = True
        self.max_tokens = self.tokens_per_model()

    def _setup_code_format(self):
        self.code_format_directive = "\nAny code snippet in your responses must be inside a code block. respond yes if you will comply"
        self.code_lang = "python"
        self.code_syntax_theme = 'github-dark'  # rich.syntax.DEFAULT_THEME
        self.code_line_numbers = False
        self.code_block_idx = 0

    def _setup_voice(self):
        self.stream_voiced_text = True
        self.stream_text_delay = 0.005

    def toggle_advanced(self):
        self.cfg.display_advanced = not self.cfg.display_advanced
        return self.cfg.display_advanced

    def toggle_voice(self):
        self.cfg.use_voice = not self.cfg.use_voice
        return self.cfg.use_voice

    def toggle_voice_over(self):
        self.cfg.voice_over = not self.cfg.voice_over
        return self.cfg.voice_over

    def toggle_voice_stop(self):
        self.cfg.voice_stop = VoiceStop.period if self.cfg.voice_stop == VoiceStop.newline else VoiceStop.newline
        return self.cfg.voice_stop

    def toggle_image_view(self):
        self.cfg.image_view = not self.cfg.image_view
        return self.cfg.image_view

    def toggle_theme(self):
        self.cfg.color_theme = ThemeMode.dark if self.cfg.color_theme == ThemeMode.light else ThemeMode.light
        self.colors = self.get_term_colors()
        return self.cfg.color_theme

    def toggle_code(self):
        self.cfg.use_code_format = not self.cfg.use_code_format
        return self.cfg.use_code_format

    def _setup_openai(self):
        if self.api_key:
            openai.api_key = self.api_key
        elif os.path.exists(self.api_key_path):
            openai.api_key_path = self.api_key_path

    def _setup_console(self):
        Prompt.prompt_suffix = ''
        self.console = Console()

    def get_term_colors(self):
        dark_theme = ThemeColors(title=Colors.magenta.value, info=Colors.blue.value, cinfo="#6973f6",
                                 prompt=Colors.white.value, cmessage="#C7E9FF", cinput="bold cyan", cresponse="#FFE8F3")
        light_theme = ThemeColors(title=Colors.magenta.value, info=Colors.blue.value, cinfo="bold #0000ff",
                                  prompt=Colors.black.value, cmessage="#191846", cinput="bold #000099", cresponse="bold #ad1f98")
        return dark_theme if self.cfg.color_theme == ThemeMode.dark else light_theme

    def tokens_per_model(self):
        safety_gap = 10
        model_alias = alias_for_model(self.cfg.model)
        if model_alias in ['chatgpt', 'davinci']:
            model_max_tokens = 4096
        else:
            model_max_tokens = 2048
        return model_max_tokens - safety_gap

    def is_chat_model(self):
        model_alias = alias_for_model(self.cfg.model)
        return model_alias == 'chatgpt'

    def calc_max_tokens(self):
        total = self.tokens_per_model()
        self.max_tokens = total - self.text_to_tokens(self.prompt_input)

    def text_to_tokens(self, text):
        num_tokens = 0
        try:
            encoding = tiktoken.encoding_for_model(self.cfg.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if self.cfg.model.startswith('gpt-3.5-') or self.cfg.model.startswith('gpt-4-'):
            num_tokens += 4 + 2  # tokens for message header ("role": "user", "content": ) + response header (assistant)

        num_tokens += len(encoding.encode(text))
        return num_tokens

    def update_max_tokens(self):
        self.calc_max_tokens()
        if self.max_tokens < 0:
            # reset context
            self.console.print(f"[bold red]*** reached max tokens. resetting chat context ***[/]")
            self.reset_context(prompt=self.prompt, submit=False)
        return self.max_tokens

    def reset_context(self, prompt, submit):
        self.after_reset = True
        self.prompt = prompt
        current_prompt = f"\n{self.prompt}\n" if self.prompt else ""
        self.add_to_conversation(current_prompt, is_response=False, reset=True)
        if self.cfg.use_code_format:
            self.apply_code_format_directive(prompt=prompt, submit=submit)
        self.prompt_input = self.conversation
        self.calc_max_tokens()
        self.update_shell_prompt()

    def run(self):
        self.shell = ShellHandler()
        gpterm_version = pkg_resources.get_distribution('gpterm').version
        self.gpterm_intro = f"{self.colors.title}## GPTerm {gpterm_version} - Interact with a GPT model via a terminal\n" \
                            f"Type {self.colors.info}/help{self.colors.title} to list available commands.{self.colors.end} " \
                            f"{self.colors.info}/exit{self.colors.title} or {self.colors.info}^C{self.colors.title} to quit.{self.colors.end}\n"
        self.shell.set_gpt_terminal(self)
        self.update_shell_prompt()
        while True:
            self.run_cmdloop()

    def run_cmdloop(self):
        try:
            self.shell.cmdloop()
        except KeyboardInterrupt:
            try:
                if not self.in_gpt_response:
                    response = Prompt.ask("\nExit GPTerm (y/n)? ")
                    if response.lower().startswith('y'):
                        print()
                        sys.exit(0)
                    elif not response:
                        print()
                else:
                    self.in_gpt_response = False
                    print()
                print()
            except KeyboardInterrupt:
                print("\n")

    def update_shell_prompt(self):
        shell_prompt = f"{self.colors.info}🌴 {self.max_tokens} {self.colors.prompt}>{self.colors.end} "
        self.shell.set_shell_prompt(shell_prompt)

    def apply_code_format_directive(self, prompt="", submit=True):
        if not self.cfg.use_code_format:
            return
        self.console.print(f"[{self.colors.cmessage}]{self.code_format_directive}[/]")
        self.prompt = self.code_format_directive + "\n"
        if prompt:
            self.prompt += f"\n{prompt}\n"
        if submit:
            self.submit_prompt(self.prompt)
            self.add_to_conversation("\n", is_response=True, reset=False)
        else:
            self.add_to_conversation(f"{self.prompt}", is_response=False, reset=False)

    def get_completion(self):
        if self.is_chat_model():
            return self.get_chat_completion()
        else:
            return self.get_text_completion()

    def get_text_completion(self):
        completion = openai.Completion.create(
            headers={"source": "gpterm"},
            engine=self.cfg.model,
            prompt=self.prompt_input,
            max_tokens=self.max_tokens,
            n=1,
            temperature=self.cfg.temperature,
            stop=None,
            stream=self.stream,
            # presence_penalty=0,
            # frequency_penalty=0,
        )
        return completion

    def get_chat_completion(self):
        completion = openai.ChatCompletion.create(
            headers={"source": "gpterm"},
            model=self.cfg.model,
            messages=[
                {"role": "user", "content": self.prompt_input},
            ],
            max_tokens=self.max_tokens,
            n=1,
            temperature=self.cfg.temperature,
            stop=None,
            stream=self.stream,
            # presence_penalty=0,
            # frequency_penalty=0,
        )
        return completion

    def submit_prompt(self, prompt):
        try:
            self.prompt = prompt
            self.add_to_conversation(f"\n{self.prompt}\n", is_response=False)
            self.prompt_input = self.conversation

            if self.debug:
                print(f"[green]{self.prompt_input}[/]", end='')

            self.update_max_tokens()
            self.update_shell_prompt()
            completion = self.get_completion()
            self.handle_completion(completion)
            self.prompt_idx += 1
        except Exception as e:
            self.print_error(e)

    def handle_response_line(self, response, end=False):
        self.resp_line += response
        self.resp_sentence += response
        if self.resp_line.strip() == '```':
            self.code_block_idx = 0
            self.print_chat_response(response.strip() + "\n", force=True)
            self.in_code_block = not self.in_code_block
            self.resp_line = ""
            return

        if not self.in_code_block or response == '``':
            self.print_chat_response(response)
            if self.cfg.use_voice and self.stream_voiced_text and not self.first_sentence:
                time.sleep(self.stream_text_delay)

        has_nl = '\n' in response
        has_fs = '.' in response
        # has_fs = '.' in response or ',' in response
        is_eol = end or has_nl  # end of line
        is_eos = is_eol or has_fs  # end of sentence

        if is_eos:
            self.first_sentence = False
            if is_eol:
                if self.in_code_block:
                    self.print_code_response()
                else:
                    if self.cfg.voice_stop == VoiceStop.newline:
                        self.voice(self.resp_line)
                self.resp_line = ""
            if self.cfg.voice_stop == VoiceStop.period:
                self.voice(self.resp_sentence)
            self.resp_sentence = ""

    def add_to_conversation(self, text, is_response=False, reset=False):
        if reset:
            self.conversation = ""
            self.conversation_formatted = ""
        self.conversation += text
        if is_response:
            if self.resp_start:
                text = self.add_gpt_prefix(text)
            self.conversation_formatted += f"[{self.colors.cresponse}]{text}[/]"
        else:
            if text.strip() != '':
                self.conversation_formatted += f"\n\n[{self.colors.cinput}][Me]: {text.lstrip()}[/]"

    def voice(self, text):
        if not text:
            return
        if not self.cfg.use_voice:
            return
        if self.in_code_block or text == '```' or text == '``':
            return
        line_to_speak = text
        line_to_speak = line_to_speak.replace("\"", "\\\"").replace("`", "")
        if line_to_speak.startswith('-'):
            line_to_speak = f"\\{line_to_speak}"
        try:
            highlight = '-i' if self.cfg.voice_over else ''
            rc = os.system(f'say {highlight} -v {self.cfg.voice_name} \"{line_to_speak}\"')
            if rc != 0:
                self.abort_response = True
        except Exception as e:
            err_msg = f"Failed to produce voiced text: {e}\nYou can disable this feature with /voice"
            self.print_error(err_msg)

    def print_error(self, e):
        self.console.print(f"[bold red]*** got error: {str(e)} ***[/]")

    def reset_response_state(self):
        self.resp_line = ""
        self.resp_sentence = ""
        self.resp_start = True
        self.in_gpt_response = False

    def handle_completion(self, completion):
        if self.stream:
            self.reset_response_state()
            self.first_sentence = True
            self.code_block_idx = 0
            self.abort_response = False
            for idx, obj in enumerate(completion):
                self.in_gpt_response = True
                if self.abort_response:
                    completion.close()
                    self.add_to_conversation("\n")
                    self.reset_response_state()
                    break
                response = self.get_response(obj)
                if self.debug:
                    print(f"[yellow]{response}[/]", end='')

                if idx == 0 and response == '\n':  # happens at any response from text completion
                    continue

                if self.after_reset:
                    if idx == 1 and response == '\n\n':  # happens at first response from chat completion
                        continue

                if response is not None:
                    self.add_to_conversation(response, is_response=True)
                    self.handle_response_line(response)
                    self.resp_start = False

            if self.resp_line:
                self.handle_response_line('', end=True)

            self.in_gpt_response = False
            self.after_reset = False
            print("\n")
        else:
            print(completion.choices[0].text)

    def get_response(self, obj):
        if self.is_chat_model():
            return self.get_chat_response(obj)
        else:
            return self.get_text_response(obj)

    def get_text_response(self, obj):
        return obj.choices[0].text

    def get_chat_response(self, obj):
        response = obj.choices[0].delta.content if 'content' in obj.choices[0].delta else None
        return response

    def print_code_response(self):
        if self.resp_line == '\n' and self.code_block_idx == 0:
            return
        self.code_block_idx += 1
        if self.resp_line.endswith('\n'):
            self.resp_line = self.resp_line[:-1]
        syntax = Syntax(self.resp_line, self.code_lang, line_numbers=self.code_line_numbers, theme=self.code_syntax_theme)
        self.console.print(syntax, end='')

    def print_chat_response(self, response, force=False):
        if self.cfg.voice_over and not force:
            return
        if self.resp_start:
            response = self.add_gpt_prefix(response, include_prefix=False)
        self.console.print(f"[{self.colors.cresponse}]{response}[/]", end='')

    def print_info(self, message):
        self.console.print(f"{message}\n")

    def add_gpt_prefix(self, text, include_prefix=True):
        starting_nl = '\n' if text == '``' else ''
        prefix = "[GPT]: " if include_prefix else ""
        text = f"{prefix}{starting_nl}{text}"
        return text

    def submit_image_gen_request(self, prompt):
        size_px = self.cfg.image_size
        self.console.print(f"[{self.colors.cresponse}]Generating Image...[/]")
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=f"{size_px}x{size_px}"
            )
            image_url = response['data'][0]['url']
            self.download_image(image_url=image_url)
        except Exception as e:
            self.print_error(e)

    def download_image(self, image_url):
        if not os.path.exists(self.cfg.image_store):
            os.makedirs(self.cfg.image_store)
        image_filename = "img-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3] + ".png"
        image_path = os.path.join(self.cfg.image_store, image_filename)
        request.urlretrieve(image_url, image_path)
        if os.path.exists(image_path):
            self.console.print(f"[{self.colors.cresponse}]Image saved at: {image_path}[/]\n")
            if self.cfg.image_view:
                os.system(f"open {image_path}")
        else:
            self.console.print(f"[{self.colors.cresponse}]Failed to retrieve image[/]\n")


def gpterm_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="An OpenAI API key")
    parser.add_argument("--api_key_path", type=str, default='~/.openai-api-key',
                        help="Path to file containing an OpenAI API key")
    parser.add_argument("--theme", type=str, choices=['light', 'dark'],
                        help="Set theme to match background. light or dark")
    args = parser.parse_args()
    debug = False
    gpterm = GptTerminal(debug=debug, theme=args.theme, api_key=args.api_key, api_key_path=args.api_key_path)
    gpterm.run()


if __name__ == "__main__":
    def main():
        gpterm_main()

    main()
