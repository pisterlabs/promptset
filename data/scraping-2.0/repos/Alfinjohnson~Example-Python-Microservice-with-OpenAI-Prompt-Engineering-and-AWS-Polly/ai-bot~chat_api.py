import openai
from const_chat import OPENAPI_API_KEY
# org not need since user not in multiple org
openai.api_key = OPENAPI_API_KEY
openai.Model.list()


def chat_def(query):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Connecting to chat, {query}')  # Press Ctrl+F8 to toggle the breakpoint.
    user_input = query
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input,
        temperature=0.5,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=1.0,
        presence_penalty=0.0
    )
    print(response.choices[0].text)
    return response.choices[0].text
