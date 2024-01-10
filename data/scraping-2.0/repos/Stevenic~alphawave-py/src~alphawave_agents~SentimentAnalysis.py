from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.UserMessage import UserMessage
from promptrix.promptrixTypes import Message
from alphawave.OpenAIClient  import OpenAIClient
from alphawave.AlphaWave import AlphaWave
from alphawave.JSONResponseValidator import JSONResponseValidator
import os
import jsonschema
import readline

# Read in .env file.

# Create an OpenAI or AzureOpenAI client
client = OpenAIClient(api_key=os.getenv("OpenAIKey"))

# Define expected response schema and create a validator
ResponseSchema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
    },
    "required": ["answer", "sentiment"]
}

validator = JSONResponseValidator(ResponseSchema)

# Create a wave
wave = AlphaWave(
    client=client,
    prompt=Prompt([
        SystemMessage(
            "Answers the user but also analyze the sentiment of their message.\n"
            "Return your answer using this JSON structure:\n"
            '{"answer":"<answer>","sentiment":"positive|neutral|negative"}'
        ),
        ConversationHistory('history'),
        UserMessage('{{$input}}', 200)
    ]),
    prompt_options={
        "completion_type": "chat",
        "model": "gpt-3.5-turbo",
        "temperature": 0.9,
        "max_input_tokens": 2000,
        "max_tokens": 1000,
    },
    validator=validator,
    logRepairs=True
)

# Define main chat loop
def chat(botMessage=None):
    # Show the bots message
    if botMessage:
        if "sentiment" in botMessage:
            print(f"[{botMessage['sentiment']}]")
        print(f"{botMessage['answer']}")

    # Prompt the user for input
    input = input('User: ')
    # Check if the user wants to exit the chat
    if input.lower() == 'exit':
        # Close the readline interface and exit the process
        exit()
    else:
        # Route users message to wave
        result = wave.completePrompt(input)
        if result.status == 'success':
            chat(result.message.content)
        else:
            if result.message:
                print(f"{result.status}: {result.message}")
            else:
                print(f"A result status of '{result.status}' was returned.")
            exit()

# Start chat session
chat({"answer": "Hello, how can I help you?"})
