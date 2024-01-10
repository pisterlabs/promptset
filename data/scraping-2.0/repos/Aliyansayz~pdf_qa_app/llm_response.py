from openai import OpenAI
import os

def initialize_llm():
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client

def generate_response(client, user_input):
    message = client.beta.threads.messages.create(thread_id=client.beta.threads.create().id, role="user", content=user_input)
    response = openai.ChatCompletion.create(model="davinci", messages=[message], max_tokens=1000, temperature=0.5)
    return response.choices[0].text.strip()

