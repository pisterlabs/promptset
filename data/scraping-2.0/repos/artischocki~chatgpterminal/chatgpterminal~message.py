from typing import Optional

import sys
import shutil

import openai
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import Terminal256Formatter


class Message():
    def __init__(self,
                 role: Optional[str] = None,
                 content: Optional[str] = None
                 ) -> None:
        if role is None and content is None:
            # the first message is the system instruction of the role
            self._role = "system"
            self._content = "You are a helpful assistant."
            return
        self._role = role
        self._content = content

    @property
    def role(self):
        return self._role

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value: str):
        self._content = value

class Prompt(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="user", content=content)

    def __str__(self) -> str:
        print("\r")
        print("_")
        print("╲")

class Response(Message):
    def __init__(self, model, max_tokens, messages) -> None:

        response = openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                stream = True
                )

        self._variable_mode = False
        self._code_mode = False

        self._found_first_backtick = False
        self._found_second_backtick = False
        
        self._reading_language = False
        self._language = ""

        self._current_code_line = ""

        collected_content = []

        for chunk in response:
            try:
                content = chunk.choices[0].delta.content
            except:
                content = "\n"
            chopped_content = [*content]
            for char in chopped_content:
                collected_content.append(char)
                self._print_char(char)
                sys.stdout.flush()

        super().__init__(role="assistant", content="".join(collected_content))


    def _toggle_variable_mode(self) -> None:
        self._variable_mode = not self._variable_mode


    def _toggle_code_mode(self) -> None:
        self._code_mode = not self._code_mode


    def _set_language(self, language: str) -> None:
        self._language = language


    def _print_char(self, char: str) -> None:
        def _print(char: str) -> None:
            if char == "`":
                return
            if self._reading_language:
                return
            if self._code_mode:
                return
            if self._variable_mode:
                # print with inverted colors
                print(f"\033[7m{char}\033[0m", end="")
                return
            print(char, end="")

        if not char == "`":

            if self._code_mode and not self._reading_language:
                # check if end of line -> highlight accumulated line
                if char != "\n":
                    self._current_code_line += char
                else:
                    formatter = Terminal256Formatter(style="material")
                    highlighted_code_line = highlight(self._current_code_line,
                                                      self._lexer, formatter) 
                    print(highlighted_code_line, end="")
                    self._current_code_line = ""

            if self._found_first_backtick:
                # in this case it is at least a variable
                self._toggle_variable_mode()

            if self._reading_language:
                # read the language name char by char until \n
                if char == "\n":
                    if self._language != "":
                        print(" ╭" + (len(self._language)+2) * "─" + "╮")
                        print(" │ " + self._language + " │")
                    try:
                        self._lexer = get_lexer_by_name(self._language)
                    except:
                        self._lexer = guess_lexer("")
                    self._reading_language = False
                    terminal_width = shutil.get_terminal_size().columns
                    seperators = list("─" * terminal_width)
                    if self._language != "":
                        seperators[1] = "┴"
                        seperators[4+len(self._language)] = "┴"
                    print("".join(seperators))
                    return

                self._language += char

            self._found_first_backtick = False
            self._found_second_backtick = False
            _print(char)
            return

        if not self._found_first_backtick:
            self._found_first_backtick = True
            _print(char)
            return

        if not self._found_second_backtick:
            self._found_second_backtick = True
            _print(char)
            return

        self._toggle_code_mode()

        if self._code_mode:
            self._language = ""
            self._reading_language = True
        else:
            terminal_width = shutil.get_terminal_size().columns
            seperators = "─" * terminal_width
            print(seperators)

        _print(char)
        self._found_first_backtick = False
        self._found_second_backtick = False
