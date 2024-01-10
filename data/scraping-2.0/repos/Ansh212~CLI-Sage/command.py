import openai
import subprocess
import os
import sys
import signal
import config

from colorama import Fore, Back, Style


terminal_width = os.get_terminal_size().columns
continuous_line = u'\u2500' * terminal_width

def signal_handler(signal, frame):
    print()
    print(Style.BRIGHT + Fore.YELLOW + "Bye" + Style.RESET_ALL)
    sys.exit()


openai.api_key = config.api_key


def chat_with_gpt(prompt: str, conversation_history: list) -> str:
    generated_text = ""

    messages = [
        {"role": "system", "content": "You are a Linux command assistant, respond only with the command without explanation"}
    ]

    for turn in conversation_history:
        messages.append({"role": "user", "content": turn[0]})
        messages.append({"role": "assistant", "content": turn[1]})

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    try:
        for chunk in response:
            delta_content = chunk['choices'][0]['delta'].get('content', '')
            if delta_content:
                generated_text += delta_content
                print(Style.BRIGHT + Fore.LIGHTCYAN_EX + delta_content + Style.RESET_ALL, end='', flush=True)

    except Exception as e:
        print(f"Error: {e}")

    print()
    return generated_text

# Rest of your code remains unchanged

if __name__ == "__main__":
    conversation_history = []  # List to store (prompt, response) tuples
    response = ""

    print(Style.BRIGHT + Fore.YELLOW + "Press (E)xecute to run commands and CTRL-C to exit" + Style.RESET_ALL)


    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            user_input: str = input(">>> ")

            if user_input.lower() in ['quit', 'exit', 'bye', 'q', 'x', 'b']:
                break

            if user_input == "E":
                # Execute the generated command
                os.system(response)
                print(continuous_line)                
                conversation_history.append((user_input, response))
                continue 

            response = chat_with_gpt(user_input, conversation_history)
            conversation_history.append((user_input, response))

    except KeyboardInterrupt:
        print("Okay")
