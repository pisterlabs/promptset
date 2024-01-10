from openai import OpenAI
from os import getenv as env

# API key is stored in .env file
OPEN_AI_API_KEY = env("OPEN_AI_API_KEY_BM")

client = OpenAI(
    api_key=OPEN_AI_API_KEY,
    # organization="org-xxx",
    # project="proj-xxx",

)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a the virtual assistant for the company that sells tomatoes."},
        {"role": "user", "content": "Tell me about your company."},
        {"role": "assistant", "content": "Tell them about the company, and what it does, giving an example."},
    ],

)

# print(completion.choices[0].message)
print(completion.choices[0].message.content) # <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>

# Path: app/turbo.py
# completion = client.chat.completions(
#     engine="davinci",
#     prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
#     temperature=0.9,
#     max_tokens=150,
#     top_p=1,
#     frequency_penalty=0.0,
#     presence_penalty=0.6,
#     stop=["\n", " Human:", " AI:"],
# )
