import os
from openai import OpenAI
from colorama import init, Fore, Style
client=OpenAI()

#openai.organization = "org-placeholder"
client.api_key = os.getenv("OPENAI_API_KEY")
model="gpt-4"     #"gpt-4"

session_tokens = 0
# Set up initial conversation context
conversation = []

# init colorama
init()

# Create an instance of the ChatCompletion API
def chatbot(conversation):
    max_tokens=4096
    completion= client.chat.completions.create(model=model, messages=conversation, max_tokens=max_tokens)

    message = completion.choices[0].message.content
    total_tokens = completion.usage.total_tokens
    return message, total_tokens

# Print welcome message and instructions
print(Fore.GREEN + "Welcome to the chatbot! To start, enter your message below.")
print("To reset the conversation, type 'reset' or 'let's start over'.")
print("To stop, say 'stop','exit', or 'bye'" + Style.RESET_ALL)

# Loop to continuously prompt user for input and get response from OpenAI
while True:
    user_input = input(Fore.CYAN + "User: " + Style.RESET_ALL)
    if user_input.lower() in ["reset", "restart", "let's start over"]:
        conversation = []
        print(Fore.YELLOW + "Bot: Okay, let's start over." + Style.RESET_ALL)
        
    elif user_input.lower() in ["stop", "exit", "bye", "quit", "goodbye"]:
        print(Fore.RED + Style.BRIGHT + "Bot: Okay, goodbye!\n  Session Tokens Used: {})".format(session_tokens)+"\n" + Style.RESET_ALL)
        break
    else:
        # Append user message to conversation context
        conversation.append({"role": "user", "content": user_input})
        # Generate chat completion
        chat, total_tokens = chatbot(conversation)
        
        # Append bot message to conversation context
        conversation.append({"role": "assistant", "content": chat})

        # Print response
        print(Fore.YELLOW + "Bot: " + Style.RESET_ALL + chat + Fore.MAGENTA + Style.BRIGHT + "\n(tokens: {})".format(total_tokens) + Style.RESET_ALL)
        session_tokens += total_tokens