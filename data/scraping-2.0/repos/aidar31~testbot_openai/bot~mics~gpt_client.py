
from openai import OpenAI

def start_chat_gpt(client: OpenAI, request, messages):
    message = str(request)
    messages.append({'role': 'user', 'content': message})
    
    chat = client.completions.create(
        model="gpt-3.5-turbo-0613",
        prompt=messages
    )
    
    answer = chat['choices'][0]['message']['content']
    messages.append({'role': 'assistant', 'content': answer})
    return answer