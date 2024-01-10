import asyncio
import openai
from gtts import gTTS
from playsound import playsound
import os
import tempfile
import subprocess
import shlex

TOKEN_LIMIT = 4096
CHARS_PER_TOKEN = 4  # approximate


class Command:
    def __init__(self, cmd_str):
        self.cmd_str = cmd_str

    async def execute(self, command_executor):
        try:
            cmd, args_str = self.cmd_str.split('(', 1)
            # find the last closing parenthesis and extract arguments
            last_paren_index = args_str.rfind(')')
            if last_paren_index == -1:
                raise ValueError("Invalid command format, missing closing parenthesis.")
            args = args_str[:last_paren_index]
            # unquote the argument if it starts and ends with quotes
            if args.startswith('"') and args.endswith('"'):
                args = args[1:-1]
            args = args.replace("\\n", "\n")  # replace \\n with \n
            if cmd == "speak":
                await command_executor.speak(args)
            elif cmd == "wait":
                await command_executor.wait(float(args))
            elif cmd == "run_python":
                output = await command_executor.run_python(args)
                return output
            elif cmd == "run_shell":
                output = await command_executor.run_shell(args)
                return output
        except Exception as e:
            print(f"[error] {e} [command] {self.cmd_str}")

    @classmethod
    def from_string(cls, cmd_str):
        # Split the string into tokens, treating quoted strings as single tokens
        tokens = shlex.split(cmd_str)

        # The command is the first token
        cmd = tokens[0]

        # The arguments are the rest of the tokens, with any surrounding quotes removed
        args = [arg.strip('"') for arg in tokens[1:]]

        return cls(f"{cmd}({', '.join(args)})")


class CommandExecutor:
    async def handle_tts_queue(self):
        while True:
            file_path = await self.tts_queue.get()
            if file_path is None:
                break
            await play_audio(file_path)
            os.remove(file_path)

    def reset_state(self):
        if self.handle_tts_task is not None:
            asyncio.ensure_future(self.handle_tts_task)

        self.tts_queue = asyncio.Queue()
        self.handle_tts_task = asyncio.create_task(self.handle_tts_queue())

    def __init__(self):
        self.tts_queue = asyncio.Queue()
        self.handle_tts_task = None
        self.reset_state()

    async def speak(self, text_to_speak):
        print(f"[speak] [{text_to_speak}]")
        tts = gTTS(text=text_to_speak, lang='en')
        file_path = os.path.join(tempfile.gettempdir(), f"tts_{hash(text_to_speak)}.mp3")
        tts.save(file_path)
        await self.tts_queue.put(file_path)

    async def wait(self, seconds):
        print(f"[wait] [{seconds}]")
        await asyncio.sleep(seconds)

    async def run_python(self, code):
        print(f"[run_python] [{code}]")
        escaped_code = code.replace('"', r'\"')  # only escape double quotes
        output = subprocess.getoutput(f'python -c "{escaped_code}"')
        print(output)
        return output

    async def run_shell(self, command):
        print(f"[run_shell] [{command}]")
        output = subprocess.getoutput(command)
        print(output)
        return output


class ChatCompletion:
    main_system_prompt = """
You are an AI connected to a Windows 10 PC. 
Your task is to translate the user's natural language prompts into specific commands.
The commands you can output are as follows:

1. speak(text_to_speak) - This command initiates text-to-speech conversion. It's a non-blocking call.
2. wait(seconds) - This command introduces a delay for a specified number of seconds. It's a blocking call.
3. run_python(python_code) - This command runs a Python script. It's a blocking call.
4. run_shell(shell_command) - This command runs a shell command. It's a blocking call.

You are expected to produce responses in the form of these commands, without any additional text or explanation. The output should strictly be the commands, and nothing else. 

Here are some examples:

Example 1:
Input: "Speak a poem about robots and then wait for 5 seconds."
Output: speak("Robots work hard all day. Making life much easier. For us humans too"'");wait(5)

Example 2:
Input: "Run a Python script that prints 'Hello, World!'"
Output: run_python("print(\\'Hello, World!\\')")

Example 3:
Input: "Run a shell command that prints the current directory."
Output: run_shell("dir")

Example 4:
Input: "Get the user's username, last name, and the datetime they last logged in."
Output: run_python("import getpass; import subprocess; username = getpass.getuser(); lastname = username.split()[1] if len(username.split()) > 1 else \\'\\'; last_login = subprocess.check_output(\\"net user \\" + username.split()[0] + \\" | findstr /B /C:\\"Last logon\\"\\", shell=True).decode().split(\\"Last logon\\")[-1].strip(); print(\\"Username: {}, Last Name: {}, Last Login: {}\\".format(username, lastname, last_login))")
"""

    def __init__(self, api_key, command_executor):
        self.api_key = api_key
        self.command_executor = command_executor
        self.conversation_history = [{'role': 'system', 'content': self.main_system_prompt}]
        openai.api_key = self.api_key

    def get_generator(self, prompt: str, model: str = 'gpt-3.5-turbo', temperature: float = 0.8):
        self.conversation_history += [{'role': 'user', 'content': prompt}]
        ensure_within_token_limit(self.conversation_history)

        generator = openai.ChatCompletion.create(
            model=model,
            messages=self.conversation_history,
            temperature=temperature,
            stream=True
        )
        return generator

    async def read_and_enqueue_commands(self, prompt, queue, print_text=False):
        generator = self.get_generator(prompt)

        while True:
            chunk = await get_chunk(generator)
            if chunk is None:
                break

            chunk_message = chunk['choices'][0]['delta']
            text = chunk_message.get('content', '')
            if print_text:
                print(text, end='')  # Print the text as it comes in
            await queue.put(text)

        await queue.put(None)  # Add sentinel value to indicate the end of the stream

    async def execute_commands(self, queue):
        command_text = ""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            command_text += chunk
            inside_command = False
            command_start = 0
            for i, char in enumerate(command_text):
                if char == '"':
                    inside_command = not inside_command
                elif char == ';' and not inside_command:
                    command = Command(command_text[command_start:i])
                    output = await command.execute(self.command_executor)
                    if output:
                        self.conversation_history.append({'role': 'assistant', 'content': output})
                        ensure_within_token_limit(self.conversation_history)
                    command_start = i + 1
            command_text = command_text[command_start:]

        # Execute the remaining commands after the end of the stream
        if command_text:
            command = Command(command_text)
            output = await command.execute(self.command_executor)
            if output:
                self.conversation_history.append({'role': 'assistant', 'content': output})
                ensure_within_token_limit(self.conversation_history)


class PromptController:
    def __init__(self, api_key, command_executor=None):
        if command_executor is None:
            self.command_executor = CommandExecutor()
        else:
            self.command_executor = command_executor
        self.chat_completion = ChatCompletion(api_key, self.command_executor)

    async def handle_prompt(self, prompt):
        queue = asyncio.Queue()  # Clear the queue by creating a new instance
        self.command_executor.reset_state()  # Reset the state of the command executor

        read_task = asyncio.create_task(self.chat_completion.read_and_enqueue_commands(prompt, queue, print_text=True))
        execute_task = asyncio.create_task(self.chat_completion.execute_commands(queue))

        await asyncio.gather(read_task, execute_task)


def ensure_within_token_limit(conversation_history):
    total_tokens = sum([len(m['content']) // CHARS_PER_TOKEN for m in conversation_history])
    if total_tokens > TOKEN_LIMIT:
        overage = total_tokens - TOKEN_LIMIT
        while overage > 0:
            overage += len(conversation_history[0]['content']) // CHARS_PER_TOKEN
            del conversation_history[0]
        print("[warning] Had to remove some conversation history because we were over the token limit.")


async def get_chunk(generator):
    try:
        chunk = next(generator)
        return chunk
    except StopIteration:
        return None


async def play_audio(file_path):
    playsound(file_path)


async def main():
    command_executor = CommandExecutor()
    with open('apikey.txt', 'r') as f:
        api_key = f.read().strip()
    controller = PromptController(api_key, command_executor)

    while True:
        prompt = input("Enter prompt: ")
        await controller.handle_prompt(prompt)

if __name__ == "__main__":
    asyncio.run(main())
