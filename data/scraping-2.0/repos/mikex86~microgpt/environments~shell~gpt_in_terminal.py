import threading
import time

import openai
import pyglet

from docker_terminal_provider import DockerTerminalProvider
from environments.shell.gptshelltask.gptshelltask_executor import GptShellTaskExecutor
from terminal_gui import TerminalGui

TERMINAL_WIDTH = 60
TERMINAL_HEIGHT = 30

INTER_GPT_DELAY = 0.2
INTER_STDIN_DELAY = 0.025

TASK = "Write a calculator console app in C that supports arbitrary expressions with +, -, *, /, parentheses and order of operations"


openai.api_base = "http://localhost:5000/api/v1"
openai.api_key = "sk-1234"

def main():
    term_provider = DockerTerminalProvider('test_ubuntu', (TERMINAL_WIDTH, TERMINAL_HEIGHT))
    # term_provider.send_input("apt-get update && apt-get upgrade -y\r")
    # term_provider.send_input("apt-get install git -y\r")
    # term_provider.send_input("git clone https://github.com/mikex86/terminal_recorder.git\r")
    term_gui = TerminalGui('Terminal', term_provider, True)

    # gpt_shell_task_executor = GptShellTaskExecutor("gpt-3.5-turbo", TASK, term_provider)

    gpt_shell_task_executor = GptShellTaskExecutor("replit-3b", TASK, term_provider)

    last_action_time = time.time()
    last_stdin_time = time.time()
    response_underway = False
    few_shot_learning_examples_performed = False

    stdin_to_send = []

    def make_response():
        nonlocal last_action_time, response_underway
        next_completion, delay = gpt_shell_task_executor.get_next_completion()  # get next action from GPT

        def finish_response():
            nonlocal response_underway, last_action_time
            response_underway = False
            last_action_time = time.time()

        if next_completion is not None:
            def send_to_terminal(new_stdin: str):
                print(new_stdin, end='')
                if '\n' in new_stdin:
                    lines = new_stdin.split('\n')

                    if new_stdin.endswith('\n'):
                        lines = lines[:-1]

                    for line in lines:
                        stdin_to_send.append(line + '\r')
                else:
                    stdin_to_send.append(new_stdin)

            print("stdin from GPT:\n")
            next_completion.add_finish_listener(lambda: pyglet.clock.schedule_once(lambda dt: finish_response(), delay))
            next_completion.add_stdin_listener(lambda new_stdin: send_to_terminal(new_stdin))
        else:
            print("No action from GPT")
            pyglet.clock.schedule_once(lambda dt: finish_response(), delay)

    current_fsl_idx = 0

    def update():
        nonlocal last_action_time, last_stdin_time, response_underway, stdin_to_send, \
            few_shot_learning_examples_performed, current_fsl_idx

        term_provider.update()

        if time.time() - last_stdin_time > INTER_STDIN_DELAY:
            if len(stdin_to_send) > 0:
                term_provider.send_input(stdin_to_send.pop(0))
                last_stdin_time = time.time()
                return

        if time.time() - last_action_time < INTER_GPT_DELAY:
            return

        if not term_provider.is_open():
            pyglet.app.exit()

        if not few_shot_learning_examples_performed:
            gpt_shell_task_executor.make_fsl_example(current_fsl_idx)
            last_stdin_time = time.time()
            current_fsl_idx += 1
            if current_fsl_idx == gpt_shell_task_executor.num_fsl_examples:
                few_shot_learning_examples_performed = True
            return

        if response_underway or len(stdin_to_send) > 0:
            return

        response_underway = True
        threading.Thread(target=make_response, daemon=True, name="MakeResponseThread").start()

    pyglet.clock.schedule(lambda dt: update())

    pyglet.app.run()


if __name__ == '__main__':
    main()
