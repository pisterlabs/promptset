#!/usr/bin/env python3
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import openai
except ImportError:
    print("Installing module: openai")
    install('openai')
    import openai

try:
    import tiktoken
except ImportError:
    print("Installing module: tiktoken")
    install('tiktoken')
    import tiktoken

try:
    from termcolor import colored
except ImportError:
    print("Installing module: termcolor")
    install('termcolor')
    from termcolor import colored

print("")
openai.api_key = input("Please enter your OpenAI API key: ")
MAX_TOKENS = 512

def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def openai_chat_completion(messages):
    """Returns the response from the OpenAI API given an array of messages."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message['content']

def truncate_by_removing(conversation_history):
    """Removes the oldest messages until the conversation history is short enough."""
    print(colored("Removing oldest messages.", 'red'))
    while num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        conversation_history.pop(0)
    return conversation_history

def truncate_by_summarizing(conversation_history):
    """Summarizes the oldest messages until the conversation history is short enough."""
    print(colored("Removing oldest messages.", 'red'))
    messages_to_remove = []
    while num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        messages_to_remove.append(conversation_history.pop(0))
    if messages_to_remove:
        messages_to_remove.append({"role": "system", "content": "You are responsible for summarizing the previous conversation."})
        summary = openai_chat_completion(messages_to_remove)
        print(colored(f"\nSummary of removed messages: \n{summary}", 'yellow'))
        conversation_history.insert(0, {"role": "assistant", "content": f'Summary of Removed Messages: {summary}'})
    return conversation_history

conversation_history = []
while True:
    if num_tokens_from_messages(conversation_history) > MAX_TOKENS:
        trunc_method = input(colored("\nConversation history is too long. Would you like to remove the oldest messages until it fits or summarize the conversation? ", 'red'))
        if trunc_method.lower() in ['summarize', 's']:
            conversation_history = truncate_by_summarizing(conversation_history)
        elif trunc_method.lower() in ['remove', 'r']:
            conversation_history = truncate_by_removing(conversation_history)
    
    user_input = input("\nPlease enter your request (type 'q' or 'quit' to exit): ")
    if user_input.lower() in ['q', 'quit']:
        break

    new_message = {"role": "user", "content": user_input}
    conversation_history.append(new_message)

    print(colored(f"\n{num_tokens_from_messages(conversation_history)} tokens", 'green'))
    print(colored("\nFull conversation history:", 'blue'))
    for message in conversation_history:
        print(message)
    response_content = openai_chat_completion(conversation_history)
    conversation_history.append({"role": "assistant", "content": response_content})
    print(f"\nResponse: \n{response_content}")