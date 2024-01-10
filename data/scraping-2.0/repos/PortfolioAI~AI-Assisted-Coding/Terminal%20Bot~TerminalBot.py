#OpenAI Console Chat Application
#Created with GPT-4
#Date: June 15, 2023
import os
import openai
import json
import pyttsx3
import re
import markdown2
from rich.console import Console
from rich.pretty import Pretty
from rich.theme import Theme
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
import asyncio
import atexit
from functools import partial
custom_theme = Theme({
    "info": "cyan",
    "warning": "magenta",
    "user": "green",
    "assistant": "blue",
    "error": "red",
})
console = Console(theme=custom_theme)
openai.api_key = "INSERT_YOUR_API_KEY"
openai.organization = "INSERT_YOUR_OPENAI_ORG"
engine = pyttsx3.init()
tts_enabled = False
def speak(text):
    if tts_enabled:
        engine.say(text)
        engine.runAndWait()
async def chat_with_gpt35_turbo_async(messages, multi_turn, temperature, n=1, max_tokens=None):
    loop = asyncio.get_event_loop()
    if not multi_turn:
        messages = [messages[-1]]
    chat_fn = partial(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    future = loop.run_in_executor(None, chat_fn)
    response = await future
    return [choice.message.content.strip() for choice in response.choices]
def load_file(path):
    try:
        with open(path, 'r') as file:
            content = f"File '{path}' contents:\n" + file.read()
            console.print(f"Successfully loaded file: {path}", style="info")
            console.print(Pretty(json.loads(content)))
            return content
    except Exception as e:
        console.print(f"An error occurred while loading the file: {str(e)}", style="error")
        return ""
def exit_handler():
    # Add your clean-up code here
    console.print("Cleaning up resources...", style="info")
    # For example, stop TTS engine
    if tts_enabled:
        engine.stop()
    console.print("Goodbye!", style="assistant")
atexit.register(exit_handler)
def fit_into_model_context_limit(messages, max_chars=15000*1.5*0.95):
    total_chars = sum([len(message['content']) for message in messages])
    while total_chars > max_chars:
        removed_message = messages.pop(0)
        total_chars -= len(removed_message['content'])
    if total_chars > max_chars:
        messages[0]['content'] = messages[0]['content'][-(max_chars-len(messages[0]['content'])):]
    return messages
def save_conversation(conversation, filename):
    path = f"ENTER_DIRECTORY\\{filename}.txt"
    try:
        with open(path, 'w') as file:
            for message in conversation:
                role = message["role"]
                content = message["content"]
                file.write(f"{role}: {content}\n")
        console.print(f"Conversation saved to {path}", style="info")
    except Exception as e:
        console.print(f"An error occurred while saving the conversation: {str(e)}", style="error")
def load_conversation(filename):
    path = f"ENTER_DIRECTORY\\{filename}.txt"
    try:
        with open(path, 'r') as file:
            conversation = []
            for line in file:
                role, content = line.split(": ", 1)
                conversation.append({"role": role, "content": content.strip()})
            console.print(f"Successfully loaded conversation from: {path}", style="info")
            return conversation
    except Exception as e:
        console.print(f"An error occurred while loading the conversation: {str(e)}", style="error")
        return []
def parse_response_and_render_code(response):
    if re.search(r'```python(.|\n)*?```', response):
        code = re.findall(r'```python\n((.|\n)*?)\n```', response)[0][0]
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        return re.sub(r'```python(.|\n)*?```', "", response)
    else:
        return response
async def main():
    console.print("Welcome to the GPT-3.5 Turbo Chatbot!", style="info")
    console.print(" * Type 'quit', 'exit', or 'bye' to leave the session.", style="warning")
    console.print(" * Type 'multi' to toggle multi-turn conversations.", style="warning")
    console.print(" * Type 'set temperature X' to adjust between 0 and 1.", style="warning")
    console.print(" * Type 'set system message' to change the bot behavior.", style="warning")
    console.print(" * Type 'load path/to/file' to load a file.", style="warning")
    console.print(" * Type 'clear history' to reset the conversation.", style="warning")
    console.print(" * Type 'save conversation [filename]' to save the conversation.", style="warning")
    console.print(" * Type 'toggle tts' to toggle Text-to-Speech.", style="warning")
    console.print(" * Type 'load conversation [filename]' to load a saved conversation.", style="warning")
    console.print("="*40, style="info")
    system_message = "You're a helpful bot!"
    messages = [{"role": "system", "content": system_message}]
    multi_turn = True
    temperature = 0.8
    n = None
    max_tokens = None
    global tts_enabled
    while True:
        prompt = input("You: ").strip()
        if not prompt:
            console.print("Assistant: Sorry, I didn't catch that.", style="assistant")
            continue
        lower_prompt = prompt.lower()
        if lower_prompt in ["quit", "exit", "bye"]:
            console.print("Assistant: Goodbye!", style="assistant")
            speak("Goodbye!")
            break
        elif lower_prompt == "multi":
            multi_turn = not multi_turn
            status = "enabled" if multi_turn else "disabled"
            console.print(f"Assistant: Multi-turn conversations {status}.", style="assistant")
            speak(f"Multi-turn conversations {status}.")
            continue
        elif lower_prompt == "toggle tts":
            tts_enabled = not tts_enabled
            status = "enabled" if tts_enabled else "disabled"
            console.print(f"Assistant: Text-to-Speech is now {status}.", style="assistant")
            speak(f"Text-to-Speech is now {status}.")
            continue
        elif lower_prompt == "clear history":
            messages = [{"role": "system", "content": system_message}]
            console.print("Assistant: Conversation history cleared.", style="assistant")
            speak("Conversation history cleared.")
            continue
        elif lower_prompt.startswith("set temperature"):
            try:
                new_temp = float(prompt.split()[2])
                if 0 <= new_temp <= 1:
                    temperature = new_temp
                    console.print(f"Assistant: Temperature set to {temperature}.", style="assistant")
                    speak(f"Temperature set to {temperature}.")
                else:
                    console.print("Assistant: Temperature value should be between 0 and 1.", style="error")
                    speak("Temperature value should be between 0 and 1.")
            except (IndexError, ValueError):
                console.print("Assistant: Invalid temperature value.", style="error")
                speak("Invalid temperature value.")
            continue
        elif lower_prompt.startswith("set system message"):
            new_message = prompt[len("set system message "):]
            if new_message:
                system_message = new_message
                messages[0]["content"] = system_message
                console.print(f"Assistant: System message set to '{system_message}'.", style="assistant")
                speak(f"System message set to '{system_message}'.")
            else:
                console.print("Assistant: System message cannot be empty.", style="error")
                speak("System message cannot be empty.")
            continue
        elif lower_prompt.startswith("load "):
            path = prompt.split(" ", 1)[1]
            content = load_file(path)
            messages.append({"role": "user", "content": path})
            messages.append({"role": "assistant", "content": content})
            console.print("Assistant: ", end="", style="assistant")
            speak("Here are the contents of the file.")
            console.print(content, style="assistant")
            continue
        elif lower_prompt.startswith("save conversation"):
            filename = prompt.split(" ", 2)[2]
            save_conversation(messages, filename)
            continue
        elif lower_prompt.startswith("load conversation"):
            filename = prompt.split(" ", 2)[2]
            messages = load_conversation(filename)
            console.print("Here is the loaded conversation: ", style="info")
            console.print(Panel.fit(Pretty(messages)))
            continue
        messages.append({"role": "user", "content": prompt})
        messages = fit_into_model_context_limit(messages)
        responses = await chat_with_gpt35_turbo_async(messages, multi_turn, temperature, n, max_tokens)
        for response in responses:
            response = parse_response_and_render_code(response)
            messages.append({"role": "assistant", "content": response})
            console.print("Assistant: ", end="", style="assistant")
            speak(response)
            console.print(response, style="assistant")
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
