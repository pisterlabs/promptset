import json
import threading
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import openai

from environments.shell.gptshelltask.gptshellhistory_htmlviz import save_shell_task_history_as_html
from environments.shell.terminal_provider import TerminalProvider


class StdinListener:

    def __init__(self, listener: Callable[[str], None], initial_stdin: Optional[str] = None):
        self.listener = listener
        if initial_stdin is not None:
            self.notify(initial_stdin)

    def notify(self, stdin: str):
        self.listener(stdin)


class ShellStdinStreamingResponse:

    def __init__(self, response, initial_stdin: Optional[str] = None):
        self.open_ai_response = response
        if initial_stdin is not None:
            self.stdin_buffer = initial_stdin
        else:
            self.stdin_buffer = ""
        self.stdin_listeners: List[StdinListener] = []
        self.completion_listeners: List[Callable[[], None]] = []
        self.is_complete = False
        threading.Thread(target=self._read_open_ai_response, daemon=True, name="OpenAiResponseThread").start()

    def _read_open_ai_response(self):
        for chunk in self.open_ai_response:
            chunk_message = chunk['choices'][0]['delta']

            if 'content' in chunk_message:
                new_stdin = chunk_message['content']
                self.stdin_buffer += new_stdin

                # notify stdin listeners
                for listener in self.stdin_listeners:
                    listener.notify(new_stdin)

        self.is_complete = True

        # notify completion listeners
        for listener in self.completion_listeners:
            listener()

    def add_stdin_listener(self, listener: Callable[[str], None]):
        initial_stdin = None
        if len(self.stdin_buffer) > 0:
            initial_stdin = self.stdin_buffer
        self.stdin_listeners.append(StdinListener(listener, initial_stdin))

    def add_finish_listener(self, listener: Callable[[], None]):
        if self.is_complete:
            listener()
        self.completion_listeners.append(listener)


@dataclass
class ShellHistoryEntry:
    terminal_context: str
    observation: Optional[str]
    response_stdin: Optional[str]
    thought: Optional[str] = None
    erroneous_completion: Optional[str] = None
    error_message: Optional[str] = None
    ctx_is_few_shot_learning_helper: bool = True
    is_final: bool = False
    delay: float = 0.0


PROMPT = """You are a GPT, an autoregressive language mode controlling a terminal.

Execute the following task: ###TASK###
Execute the necessary commands to complete the task. You have access to a terminal inside a docker container.

You will be iteratively prompted to provide the next shell command to execute,
while receiving updates on the current state of the terminal.
Respond in the following format:
{"observation": "The ping command was not found.", "thought": "I need to install the package for ping", "delay": 10.0}\nstdin:<your input here>
The "delay" field controls how many seconds the system will wait before prompting you for the next command.
Specify "0.0" for the delay if you expect the command to execute instantly.
Only specify a delay if you expect the command to take a while to execute.
Make a best gues for how long the command will take to execute. Consider ETA estimates from running programs.

Respond with what you wish to type into the terminal as input. Eg:
"apt-get install inetutils-ping\r"
ALWAYS USE "\r" instead of "\n" to indicate the Enter key!

When an application blocks for input (eg. apt, nano, less, etc.), respond with the input you wish to provide.
If you want to respond with eg. Ctrl+C, respond with the xterm sequence for Ctrl+C: "\x03".
If command execution is not yet finished and it not accepting input, you wait by simply omitting the "stdin" field.
When using text editors, "stdin" may contain multiple lines separated by "\r".
Eg. {"observation": ..., "thought": ...}\nstdin:#include<stdio.h>\rint main() {\rprintf("Hello World!");\rreturn 0;\r}\r

IMPORTANT:
Exit the text editor after you are done editing.
Eg. to exit nano, respond with the corresponding xterm sequence. Eg. for Control+X, y + Enter respond with "\x18y\r".

IMPORTANT: Note that as an autoregressive language model you cannot meaningfully interact with a cursor-based text 
editor, when the file is not empty. If you wish to edit a file, first delete it with "rm <filename>" and then 
recreate it with "nano <filename>" and rewrite it from scratch. 

CAREFULLY READ THE TERMINAL OUTPUT AND CHOSE YOUR ACTIONS WISELY. DO NOT BLINDLY EXECUTE COMMANDS.

If the task is complete, run the exit command.
"""


class GptShellTaskExecutor:

    def __init__(self, model: str, task: str, terminal_provider: TerminalProvider):
        self.model = model
        self.task = task
        self.prompt = PROMPT.replace("###TASK###", task)
        self.term_provider = terminal_provider
        self.history = []

    def capture_term_ctx_as_hist_entry(self):
        term_context = self.term_provider.get_terminal_context().copy()

        # visualize cursor position
        x, y = self.term_provider.get_terminal_cursor_position()
        term_context[y] = term_context[y][:x] + "█" + term_context[y][x + 1:]

        new_ctx_window = ""
        for line in term_context:
            # if line is not white, add it to the new context window
            if new_ctx_window:
                new_ctx_window += "\n"
            new_ctx_window += line.rstrip()
        # strip trailing newlines
        while new_ctx_window and new_ctx_window[-1] == "\n":
            new_ctx_window = new_ctx_window[:-1]

        self.history.append(ShellHistoryEntry(new_ctx_window, None, None))

    def __make_few_shot_learning_example(self, stdin: str, observation: str, thought: str, delay: float):
        self.capture_term_ctx_as_hist_entry()
        self.history[-1].response_stdin = stdin
        self.history[-1].thought = thought
        self.history[-1].observation = observation
        self.history[-1].delay = delay
        self.history[-1].ctx_is_few_shot_learning_helper = True
        self.history[-1].is_final = True

        self.term_provider.send_input(stdin)

    num_fsl_examples = 5

    def make_fsl_example(self, i: int):
        """
        This adds an initial ls to the history, so that the GPT model thinks it has already executed an ls.
        This will hopefully make it not write random english words into the terminal.
        """
        if i == 0:
            self.__make_few_shot_learning_example("ls\r",
                                                  "I am root and in the root's home directory.",
                                                  "Let's first see what files are in the root's home directory before we "
                                                  "start the task.",
                                                  0.0)

        if i == 1:
            self.__make_few_shot_learning_example("nano test.txt\r",
                                                  "Operating a text editor is difficult.",
                                                  "I will create a new file called test.txt to test how to operate a text "
                                                  "editor.",
                                                  0.0)

        if i == 2:
            self.__make_few_shot_learning_example("This is the first line\rThis is the second line\rThis is the third line",
                                                  "The text editor has been opened.",
                                                  "I will write three lines of text without a trailing newline.",
                                                  1.0)

        if i == 3:
            self.__make_few_shot_learning_example("\x18y\r",
                                                  "I have finished editing the file.",
                                                  "I will now save and exit the file.",
                                                  0.0)
        if i == 4:
            self.__make_few_shot_learning_example("clear\r",
                                                  "The screen is cluttered with text.",
                                                  "I need to save space. I will clear the screen.",
                                                  0.0)

    def get_next_completion(self) -> Tuple[Optional[ShellStdinStreamingResponse], float]:
        while True:
            self.capture_term_ctx_as_hist_entry()

            while True:
                messages = [
                    {"role": "system", "content": self.prompt},
                ]
                for entry in self.history:
                    if entry.erroneous_completion:
                        messages.append({"role": "user", "content": entry.terminal_context})
                        messages.append({"role": "assistant", "content": entry.erroneous_completion})
                        assert entry.error_message is not None
                        messages.append({"role": "user", "content": entry.error_message})
                    else:
                        if entry.ctx_is_few_shot_learning_helper:
                            messages.append({"role": "user", "content": entry.terminal_context})

                        if entry.is_final:
                            content_obj = {}
                            if entry.observation:
                                content_obj['observation'] = entry.observation
                            if entry.thought:
                                content_obj['thought'] = entry.thought

                            content_obj['delay'] = entry.delay

                            content_str = json.dumps(content_obj)
                            if entry.response_stdin:
                                content_str += "\nstdin:" + entry.response_stdin
                            messages.append({"role": "assistant", "content": content_str})
                save_shell_task_history_as_html(self.history, 'history.html')
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                    )
                    break
                except openai.error.OpenAIError:
                    idx = self.num_fsl_examples
                    for entry in self.history[self.num_fsl_examples:-3]:
                        if entry.ctx_is_few_shot_learning_helper:
                            entry.ctx_is_few_shot_learning_helper = False
                            print(f"Omitting terminal context of history entry {idx}")
                            break
                        idx += 1
                    pass

            response_str = ""

            json_obj_started = False
            for chunk in response:
                chunk_message = chunk['choices'][0]['delta']
                response_str += chunk_message.get('content', '')

                if not json_obj_started and '{' in response_str:
                    json_obj_started = True

                if json_obj_started and '}' in response_str:
                    break

            response_json = response_str[response_str.find('{'):response_str.find('}') + 1]

            try:
                response_obj = json.loads(response_json)
                break
            except json.decoder.JSONDecodeError:
                self.history[-1].erroneous_completion = response_str
                self.history[
                    -1].error_message = 'Message from the system that lets you interact with the terminal: I could not parse your response as json.'
                print("Erroneous response: ", response_str)
                pass

        if 'observation' in response_obj:
            observation = response_obj['observation']
            self.history[-1].observation = observation
            print("GPT observation: ", observation)

        if 'thought' in response_obj:
            thought = response_obj['thought']
            self.history[-1].thought = thought
            print("GPT thought: ", thought)

        delay = 0.0

        if 'delay' in response_obj:
            delay = response_obj['delay']
            self.history[-1].delay = delay
            print("GPT delay: ", delay)

        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            response_str += chunk_message.get('content', '')

            if '\nstdin:' in response_str:
                initial_stdin = response_str[response_str.find('\nstdin:') + len('\nstdin:'):]
                self.history[-1].response_stdin = ""
                self.history[-1].is_final = False  # streaming response not complete yet

                # update_history_entry_stdin will be invoked with initial_stdin
                streaming_response = ShellStdinStreamingResponse(response,
                                                                 initial_stdin)

                def finalize_history_entry():
                    self.history[-1].is_final = True

                def update_history_entry_stdin(new_stdin):
                    self.history[-1].response_stdin += new_stdin

                streaming_response.add_finish_listener(lambda: finalize_history_entry())
                streaming_response.add_stdin_listener(lambda new_stdin: update_history_entry_stdin(new_stdin))
                return streaming_response, delay

        return None, delay
